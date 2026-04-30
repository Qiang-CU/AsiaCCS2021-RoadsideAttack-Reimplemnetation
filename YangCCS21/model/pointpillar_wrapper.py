"""
PointPillar wrapper for adversarial attack evaluation.

Handles voxelization (spconv) + model forward pass.
Used in black-box genetic algorithm attack (paper Section 3.3).
"""

import os
import numpy as np
import torch
from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils


class PointPillarWrapper:
    def __init__(self, config_path, ckpt_path, device='cuda:0'):
        self.device = torch.device(device)
        config_path = os.path.abspath(config_path)
        ckpt_path = os.path.abspath(ckpt_path)

        tools_dir = os.path.dirname(os.path.dirname(os.path.dirname(config_path)))
        orig_dir = os.getcwd()
        try:
            os.chdir(tools_dir)
            cfg_from_yaml_file(config_path, cfg)
        finally:
            os.chdir(orig_dir)
        self.cfg = cfg

        pcr = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        voxel_size = [0.16, 0.16, 4.0]

        class _PFE:
            num_point_features = 4

        class _DS:
            def __init__(self):
                self.class_names = cfg.CLASS_NAMES
                self.point_feature_encoder = _PFE()
                self.point_cloud_range = np.array(pcr, dtype=np.float32)
                self.voxel_size = np.array(voxel_size, dtype=np.float32)
                self.grid_size = ((self.point_cloud_range[3:] -
                                   self.point_cloud_range[:3]) /
                                  self.voxel_size).astype(np.int64)
                self.depth_downsample_factor = None

        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=_DS()
        )
        self.model.load_params_from_file(
            filename=ckpt_path,
            logger=common_utils.create_logger(),
            to_cpu=False
        )
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.voxel_gen = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=pcr,
            num_point_features=4,
            max_num_voxels=40000,
            max_num_points_per_voxel=32
        )

    def _voxelize(self, pts_np):
        """Convert (N,4) float32 numpy array to voxel batch_dict."""
        tv_pts = tv.from_numpy(pts_np.astype(np.float32))
        vox, coords, npts = self.voxel_gen.point_to_voxel(tv_pts)
        voxels = torch.from_numpy(vox.numpy()).float().to(self.device)
        coords_np = coords.numpy().astype(np.int32)
        coords_batch = np.pad(coords_np, ((0, 0), (1, 0)), constant_values=0)
        npts_t = torch.from_numpy(npts.numpy().astype(np.int32)).to(self.device)
        return {
            'batch_size': 1,
            'voxels': voxels,
            'voxel_coords': torch.from_numpy(coords_batch).int().to(self.device),
            'voxel_num_points': npts_t,
        }

    def detect(self, points, score_thresh=0.3):
        """
        Run detection on a point cloud.

        Args:
            points: (N,4) numpy array or torch tensor [x,y,z,intensity]
            score_thresh: confidence threshold

        Returns:
            pred_boxes:  (D,7) numpy array
            pred_scores: (D,) numpy array
        """
        if isinstance(points, torch.Tensor):
            pts_np = points.cpu().numpy()
        else:
            pts_np = points

        batch_dict = self._voxelize(pts_np)
        with torch.no_grad():
            pred_dicts, _ = self.model(batch_dict)

        pred = pred_dicts[0]
        scores = pred['pred_scores'].cpu().numpy()
        boxes = pred['pred_boxes'].cpu().numpy()
        mask = scores >= score_thresh
        return boxes[mask], scores[mask]

    def detect_score(self, points, target_pos, radius=3.0, score_thresh=0.0):
        """
        Run detection and return maximum score near target_pos.

        Args:
            points: (N,4) numpy array
            target_pos: (3,) object position [x,y,z]
            radius: proximity threshold for matching
            score_thresh: minimum score to consider

        Returns:
            max_score: float (0.0 if no detection near target)
            n_det: int (number of detections near target)
        """
        if isinstance(points, torch.Tensor):
            pts_np = points.cpu().numpy()
        else:
            pts_np = points

        batch_dict = self._voxelize(pts_np)
        with torch.no_grad():
            pred_dicts, _ = self.model(batch_dict)

        pred = pred_dicts[0]
        scores = pred['pred_scores'].cpu().numpy()
        boxes = pred['pred_boxes'].cpu().numpy()

        if len(scores) == 0:
            return 0.0, 0

        mask = scores >= score_thresh
        if not mask.any():
            return 0.0, 0

        boxes_f = boxes[mask]
        scores_f = scores[mask]
        dists = np.sqrt((boxes_f[:, 0] - target_pos[0])**2 +
                        (boxes_f[:, 1] - target_pos[1])**2)
        near = dists < radius
        if not near.any():
            return 0.0, 0
        return float(scores_f[near].max()), int(near.sum())
