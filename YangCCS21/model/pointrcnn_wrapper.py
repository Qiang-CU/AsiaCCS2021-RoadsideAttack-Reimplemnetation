"""
PointRCNN wrapper for adversarial attack using OpenPCDet.

Provides:
- Model loading from OpenPCDet config + checkpoint
- Forward hook on RCNN classification head's penultimate layer
- Batch construction from raw point clouds
- Proposal extraction with pre-NMS scores
- STE (Straight-Through Estimator) patch for ROI pooling gradient flow
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.roipoint_pool3d import roipoint_pool3d_cuda
from pcdet.utils import box_utils


# ── STE patch for RoIPointPool3d ────────────────────────────────────────────

class _RoIPointPool3dSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for RoIPointPool3d.

    Forward: runs the real CUDA kernel (discrete gather by point index).
    Backward: distributes output gradients back to the *input* features
              at the gathered indices, as if the gather were differentiable.

    The CUDA kernel writes pooled_features as [xyz_coords | point_feats]
    for the sampled points.  We need to track which input points were
    selected so we can scatter gradients back.  Since the CUDA kernel
    doesn't expose indices, we reconstruct them by matching pooled xyz
    against input xyz (exact float match within each ROI).
    """

    @staticmethod
    def forward(ctx, points, point_features, boxes3d,
                pool_extra_width, num_sampled_points):
        """
        Args:
            points:         (B, N, 3)
            point_features: (B, N, C)
            boxes3d:        (B, M, 7)
        Returns:
            pooled_features: (B, M, S, 3+C)
            pooled_empty_flag: (B, M)
        """
        B, N, _ = points.shape
        M = boxes3d.shape[1]
        C = point_features.shape[2]
        S = num_sampled_points

        pooled_boxes3d = box_utils.enlarge_box3d(
            boxes3d.view(-1, 7), pool_extra_width
        ).view(B, -1, 7)

        pooled_features = point_features.new_zeros((B, M, S, 3 + C))
        pooled_empty_flag = point_features.new_zeros((B, M)).int()

        roipoint_pool3d_cuda.forward(
            points.contiguous(), pooled_boxes3d.contiguous(),
            point_features.contiguous(), pooled_features, pooled_empty_flag
        )

        # Reconstruct gather indices by matching pooled xyz to input xyz.
        pooled_xyz = pooled_features[:, :, :, 0:3].detach()  # (B, M, S, 3)
        pooled_xyz_flat = pooled_xyz.reshape(B, M * S, 3)

        # Chunked cdist to avoid OOM (M*S × N can be huge)
        chunk = 4096
        indices = points.new_zeros((B, M * S), dtype=torch.long)
        for b_idx in range(B):
            pts_b = points[b_idx]  # (N, 3)
            for s in range(0, M * S, chunk):
                e = min(s + chunk, M * S)
                d = torch.cdist(pooled_xyz_flat[b_idx, s:e], pts_b)  # (chunk, N)
                indices[b_idx, s:e] = d.argmin(dim=1)
                del d

        indices = indices.reshape(B, M, S)
        ctx.save_for_backward(indices, point_features)
        ctx.shape_info = (B, N, M, S, C)

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_pooled, grad_empty):
        indices, point_features = ctx.saved_tensors
        B, N, M, S, C = ctx.shape_info

        # grad_pooled: (B, M, S, 3+C)
        # We only pass gradient for the feature part (last C dims),
        # not for the xyz part (first 3 dims are coordinates, not features).
        grad_feat_part = grad_pooled[:, :, :, 3:]  # (B, M, S, C)

        grad_point_features = point_features.new_zeros((B, N, C))
        idx_flat = indices.reshape(B, M * S)
        grad_flat = grad_feat_part.reshape(B, M * S, C)

        for b in range(B):
            grad_point_features[b].scatter_add_(
                0, idx_flat[b].unsqueeze(-1).expand(-1, C), grad_flat[b]
            )

        # No gradient for points (xyz) or boxes3d
        return None, grad_point_features, None, None, None


def _ste_roipool3d_gpu(self, batch_dict):
    """
    STE-patched replacement for PointRCNNHead.roipool3d_gpu.

    Removes torch.no_grad() and .detach() so gradients flow through
    ROI pooling via the STE approximation.
    """
    batch_size = batch_dict['batch_size']
    batch_idx = batch_dict['point_coords'][:, 0]
    point_coords = batch_dict['point_coords'][:, 1:4]
    point_features = batch_dict['point_features']
    rois = batch_dict['rois']
    batch_cnt = point_coords.new_zeros(batch_size).int()
    for bs_idx in range(batch_size):
        batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

    assert batch_cnt.min() == batch_cnt.max()

    # KEY CHANGE: do NOT detach point_cls_scores
    point_scores = batch_dict['point_cls_scores']
    point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
    point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
    point_features_all = torch.cat(point_features_list, dim=1)
    batch_points = point_coords.view(batch_size, -1, 3)
    batch_point_features = point_features_all.view(
        batch_size, -1, point_features_all.shape[-1]
    )

    # KEY CHANGE: use STE instead of torch.no_grad()
    pooled_features, pooled_empty_flag = _RoIPointPool3dSTE.apply(
        batch_points, batch_point_features, rois,
        self.roipoint_pool3d_layer.pool_extra_width,
        self.roipoint_pool3d_layer.num_sampled_points,
    )

    # Canonical transformation (now WITH gradient)
    roi_center = rois[:, :, 0:3]
    pooled_features_xyz = pooled_features[:, :, :, 0:3] - roi_center.unsqueeze(dim=2)
    pooled_features = torch.cat([pooled_features_xyz, pooled_features[:, :, :, 3:]], dim=-1)

    pooled_features = pooled_features.view(
        -1, pooled_features.shape[-2], pooled_features.shape[-1]
    )
    pooled_features_rot = common_utils.rotate_points_along_z(
        pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
    )
    pooled_features = torch.cat([pooled_features_rot, pooled_features[:, :, 3:]], dim=-1)
    pooled_features[pooled_empty_flag.view(-1) > 0] = 0

    return pooled_features


def apply_ste_patch(model):
    """
    Monkey-patch PointRCNNHead.roipool3d_gpu with the STE version.

    Call this after loading the model to enable gradient flow through
    the ROI pooling layer for white-box attacks.

    Args:
        model: OpenPCDet model (nn.Module) containing a .roi_head attribute
    """
    import types
    roi_head = model.roi_head
    roi_head.roipool3d_gpu = types.MethodType(_ste_roipool3d_gpu, roi_head)
    return model


class PointRCNNWrapper:
    """
    Wraps OpenPCDet's PointRCNN for adversarial attack use.

    Key features:
    - Hooks cls_layers penultimate layer to capture RCNN features
    - Provides differentiable forward pass (no NMS) for white-box attack
    - Builds input batch_dict from raw (N, 4) point clouds
    """

    def __init__(self, config_path, ckpt_path, device='cuda:0',
                 enable_ste=False):
        """
        Args:
            config_path: path to OpenPCDet PointRCNN YAML config
            ckpt_path:   path to pretrained .pth weights
            device:      torch device
            enable_ste:  if True, apply STE patch to ROI pooling for
                         gradient flow through the RCNN head
        """
        self.device = torch.device(device)
        self.ste_enabled = enable_ste

        # Load config — temporarily chdir to OpenPCDet/tools so that
        # _BASE_CONFIG_ relative paths resolve correctly
        config_path = os.path.abspath(config_path)
        ckpt_path = os.path.abspath(ckpt_path)
        # config is at …/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml
        # _BASE_CONFIG_ resolves relative to cwd, needs to be …/OpenPCDet/tools/
        tools_dir = os.path.dirname(os.path.dirname(os.path.dirname(config_path)))
        orig_dir = os.getcwd()
        try:
            os.chdir(tools_dir)
            cfg_from_yaml_file(config_path, cfg)
        finally:
            os.chdir(orig_dir)
        self.cfg = cfg

        # Build model — provide a minimal dummy dataset with all attributes
        # that detector3d_template.py requires
        class _PointFeatureEncoder:
            num_point_features = 4  # x, y, z, intensity
        class _DummyDataset:
            def __init__(self, class_names, pc_range, voxel_size):
                self.class_names = class_names
                self.point_feature_encoder = _PointFeatureEncoder()
                self.point_cloud_range = np.array(pc_range, dtype=np.float32)
                self.voxel_size = np.array(voxel_size, dtype=np.float32)
                self.grid_size = ((self.point_cloud_range[3:] - self.point_cloud_range[:3])
                                  / self.voxel_size).astype(np.int64)
                self.depth_downsample_factor = None
        # Use KITTI standard ranges
        pc_range = [0, -40, -3, 70.4, 40, 1]
        voxel_size = [0.05, 0.05, 0.1]
        if hasattr(cfg, 'DATA_CONFIG') and hasattr(cfg.DATA_CONFIG, 'POINT_CLOUD_RANGE'):
            pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        if hasattr(cfg, 'DATA_CONFIG') and hasattr(cfg.DATA_CONFIG, 'get'):
            pass  # voxel_size only needed for voxel-based models
        dummy_ds = _DummyDataset(cfg.CLASS_NAMES, pc_range, voxel_size)
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dummy_ds
        )
        self.model.load_params_from_file(
            filename=ckpt_path, logger=common_utils.create_logger(), to_cpu=False
        )
        self.model.to(self.device)
        self.model.eval()

        # Apply STE patch if requested (before freezing, so the patch is in place)
        if self.ste_enabled:
            apply_ste_patch(self.model)

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Hook state
        self.last_feature = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on RCNN cls_layers penultimate layer."""
        roi_head = self.model.roi_head
        # In OpenPCDet PointRCNN, the cls_layers is a Sequential module.
        # We hook the second-to-last layer to capture features before final logit.
        cls_layers = roi_head.cls_layers
        if isinstance(cls_layers, nn.Sequential):
            # Hook the layer before the final Linear
            hook_layer = None
            layers_list = list(cls_layers.children())
            for i in range(len(layers_list) - 1, -1, -1):
                if isinstance(layers_list[i], nn.Linear):
                    # Found the final linear; hook the one before it
                    if i > 0:
                        hook_layer = layers_list[i - 1]
                    break
            if hook_layer is None and len(layers_list) >= 2:
                hook_layer = layers_list[-2]

            if hook_layer is not None:
                self._hook_handle = hook_layer.register_forward_hook(
                    self._capture_feature
                )
            else:
                print("WARNING: Could not find suitable hook layer in cls_layers")
        else:
            print(f"WARNING: cls_layers is {type(cls_layers)}, not Sequential")

    def _capture_feature(self, module, input, output):
        """Hook callback: capture penultimate feature activations."""
        self.last_feature = output  # shape: (N_rois, D)

    def build_batch_dict(self, points_tensor):
        """
        Build an OpenPCDet-compatible batch_dict from a raw point cloud tensor.

        Args:
            points_tensor: (N, 4) float tensor [x, y, z, intensity] on self.device

        Returns:
            batch_dict: dict compatible with PointRCNN forward pass
        """
        if points_tensor.dim() == 2 and points_tensor.shape[1] >= 4:
            pts = points_tensor[:, :4]
        elif points_tensor.dim() == 2 and points_tensor.shape[1] == 3:
            # Add dummy intensity
            pts = torch.cat([
                points_tensor,
                torch.ones(points_tensor.shape[0], 1, device=points_tensor.device)
            ], dim=1)
        else:
            raise ValueError(f"Expected (N, 3+) points, got shape {points_tensor.shape}")

        # Add batch index column (batch_size=1)
        batch_idx = torch.zeros(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
        points_with_batch = torch.cat([batch_idx, pts], dim=1)  # (N, 5)

        batch_dict = {
            'batch_size': 1,
            'points': points_with_batch,
        }
        return batch_dict

    def forward_no_nms(self, batch_dict, rpn_only=False, run_post=True):
        """
        Run PointRCNN forward pass, capturing outputs for white-box attack.

        Args:
            batch_dict:  from build_batch_dict()
            rpn_only:    if True, stop after PointHeadBox (skip RCNN head).
                         ~2x faster. Use for Stage 1 optimization.
            run_post:    if True, run post_processing for monitoring metrics.
                         Set False to skip (saves ~10% per step).

        Returns:
            dict with RPN outputs, and RCNN outputs if rpn_only=False.
        """
        self.last_feature = None

        # OpenPCDet custom CUDA ops (e.g. pointnet2_utils) use
        # torch.cuda.*Tensor() which allocates on current_device.
        # Must match self.device to avoid cross-device errors.
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)

        for key in batch_dict:
            if isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to(self.device)

        rpn_result = {}
        with torch.set_grad_enabled(True):
            for cur_module in self.model.module_list:
                batch_dict = cur_module(batch_dict)
                mod_name = cur_module.__class__.__name__
                if mod_name == 'PointHeadBox':
                    rpn_result['point_cls_scores'] = batch_dict.get('point_cls_scores')
                    rpn_result['point_features'] = batch_dict.get('point_features')
                    rpn_result['rpn_cls_preds'] = batch_dict.get('batch_cls_preds')
                    rpn_result['rpn_box_preds'] = batch_dict.get('batch_box_preds')
                    pf = batch_dict['point_features']
                    logits = cur_module.cls_layers(pf)  # (N, num_class)
                    logits_max, _ = logits.max(dim=-1)  # (N,)
                    rpn_result['point_cls_logits'] = logits_max
                    if rpn_only:
                        break

        result = {
            **rpn_result,
            'features': self.last_feature,
        }

        if not rpn_only:
            if 'batch_cls_preds' in batch_dict:
                result['rcnn_cls_preds'] = batch_dict['batch_cls_preds']
            if 'batch_box_preds' in batch_dict:
                result['rcnn_box_preds'] = batch_dict['batch_box_preds']
            if 'rois' in batch_dict:
                result['rois'] = batch_dict['rois']

        if run_post and not rpn_only:
            with torch.no_grad():
                pred_dicts, _ = self.model.post_processing(batch_dict)
            result['pred_dicts'] = pred_dicts

        return result

    def forward_with_grad(self, points_tensor, rpn_only=False, run_post=True):
        """
        Convenience: build batch + forward, returning raw predictions.

        Args:
            points_tensor: (N, 4) float tensor with requires_grad on relevant portion
            rpn_only:      skip RCNN head for faster Stage 1
            run_post:      run post_processing for monitoring

        Returns:
            same as forward_no_nms
        """
        batch_dict = self.build_batch_dict(points_tensor)
        return self.forward_no_nms(batch_dict, rpn_only=rpn_only, run_post=run_post)

    def detect_batch(self, points_list, score_thresh=0.3):
        """
        Run batched detection on multiple point clouds in a single forward pass.

        PointNet2 backbone requires equal point counts per batch element,
        so we pad shorter clouds with duplicated points (or subsample longer
        ones) to match the maximum length in the batch.

        Args:
            points_list: list of (N_i, 4) float tensors on self.device
            score_thresh: confidence threshold

        Returns:
            list of (pred_boxes, pred_scores) tuples, one per input point cloud.
            pred_boxes: (D, 7) numpy array, pred_scores: (D,) numpy array.
        """
        B = len(points_list)
        if B == 0:
            return []
        if B == 1:
            return [self.detect(points_list[0], score_thresh)]

        # Normalise each pc to (N_i, 4)
        normed = []
        for pts_t in points_list:
            if pts_t.dim() == 2 and pts_t.shape[1] >= 4:
                normed.append(pts_t[:, :4])
            elif pts_t.dim() == 2 and pts_t.shape[1] == 3:
                normed.append(torch.cat([pts_t, torch.ones(pts_t.shape[0], 1, device=pts_t.device)], dim=1))
            else:
                raise ValueError(f"Expected (N, 3+) points, got shape {pts_t.shape}")

        # Equalise point counts: pad shorter clouds by repeating random points
        target_n = max(p.shape[0] for p in normed)
        all_pts = []
        for bi, pts in enumerate(normed):
            n = pts.shape[0]
            if n < target_n:
                pad_idx = torch.randint(0, n, (target_n - n,), device=pts.device)
                pts = torch.cat([pts, pts[pad_idx]], dim=0)
            elif n > target_n:
                pts = pts[:target_n]
            batch_idx = torch.full((target_n, 1), bi, device=pts.device, dtype=pts.dtype)
            all_pts.append(torch.cat([batch_idx, pts], dim=1))

        points_with_batch = torch.cat(all_pts, dim=0)  # (B * target_n, 5)
        batch_dict = {
            'batch_size': B,
            'points': points_with_batch,
        }

        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            for key in batch_dict:
                if isinstance(batch_dict[key], torch.Tensor):
                    batch_dict[key] = batch_dict[key].to(self.device)
            pred_dicts, _ = self.model(batch_dict)

        results = []
        for bi in range(B):
            pred = pred_dicts[bi]
            scores = pred['pred_scores'].cpu().numpy()
            boxes = pred['pred_boxes'].cpu().numpy()
            mask = scores >= score_thresh
            results.append((boxes[mask], scores[mask]))
        return results

    def detect(self, points_tensor, score_thresh=0.3):
        """
        Run full detection pipeline (with NMS) for evaluation.

        Args:
            points_tensor: (N, 4) float tensor
            score_thresh: confidence threshold

        Returns:
            pred_boxes:  (D, 7) numpy array of detected boxes
            pred_scores: (D,) numpy array of confidence scores
        """
        batch_dict = self.build_batch_dict(points_tensor)

        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            for key in batch_dict:
                if isinstance(batch_dict[key], torch.Tensor):
                    batch_dict[key] = batch_dict[key].to(self.device)
            pred_dicts, _ = self.model(batch_dict)

        pred = pred_dicts[0]
        scores = pred['pred_scores'].cpu().numpy()
        boxes = pred['pred_boxes'].cpu().numpy()

        mask = scores >= score_thresh
        return boxes[mask], scores[mask]

    def remove_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
