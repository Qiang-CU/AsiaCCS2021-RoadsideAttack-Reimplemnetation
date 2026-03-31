"""KITTI 3D Object Detection data loader and coordinate utilities."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def read_calib(calib_path):
    """Parse a KITTI calib file into a dict of numpy arrays."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])
    calib = {}
    calib['P2'] = data['P2'].reshape(3, 4)
    calib['R0_rect'] = data['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(3, 4)
    return calib


def cam_to_lidar(pts_cam, Tr_velo_to_cam, R0_rect):
    """
    Transform points from rectified camera coordinates to LiDAR coordinates.

    Args:
        pts_cam: (N, 3) in camera rect frame
        Tr_velo_to_cam: (3, 4)
        R0_rect: (3, 3)
    Returns:
        pts_lidar: (N, 3)
    """
    R0_ext = np.eye(4)
    R0_ext[:3, :3] = R0_rect
    Tr_ext = np.eye(4)
    Tr_ext[:3, :] = Tr_velo_to_cam
    transform = np.linalg.inv(R0_ext @ Tr_ext)
    pts_homo = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    return (transform @ pts_homo.T).T[:, :3]


def lidar_to_cam(pts_lidar, Tr_velo_to_cam, R0_rect):
    """Transform LiDAR points to rectified camera frame."""
    R0_ext = np.eye(4)
    R0_ext[:3, :3] = R0_rect
    Tr_ext = np.eye(4)
    Tr_ext[:3, :] = Tr_velo_to_cam
    transform = R0_ext @ Tr_ext
    pts_homo = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1))])
    return (transform @ pts_homo.T).T[:, :3]


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def read_label(label_path):
    """Parse a KITTI label_2 file.  Returns list of dicts."""
    objects = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox2d': np.array([float(x) for x in parts[4:8]]),
                'dimensions': np.array([float(x) for x in parts[8:11]]),   # h, w, l
                'location': np.array([float(x) for x in parts[11:14]]),    # cam rect x,y,z
                'rotation_y': float(parts[14]),
            }
            objects.append(obj)
    return objects


def get_bbox_corners_lidar(obj, calib):
    """
    Compute the 8 corners of a 3D bbox in LiDAR coordinates.

    Args:
        obj: dict from read_label
        calib: dict from read_calib
    Returns:
        corners_lidar: (8, 3)
    """
    h, w, l = obj['dimensions']
    x, y, z = obj['location']
    ry = obj['rotation_y']

    # Camera-frame corners (before rotation)
    # Object bottom-centre at (x, y, z); y-axis points down
    corners_cam_local = np.array([
        [ l/2,  0,  w/2],
        [-l/2,  0,  w/2],
        [-l/2,  0, -w/2],
        [ l/2,  0, -w/2],
        [ l/2, -h,  w/2],
        [-l/2, -h,  w/2],
        [-l/2, -h, -w/2],
        [ l/2, -h, -w/2],
    ])

    # Rotation around camera y-axis
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    corners_cam = (R @ corners_cam_local.T).T + np.array([x, y, z])
    corners_lidar = cam_to_lidar(corners_cam,
                                  calib['Tr_velo_to_cam'],
                                  calib['R0_rect'])
    return corners_lidar


def get_bbox_lidar(obj, calib):
    """
    Return (cx, cy, cz, l, w, h, yaw) in LiDAR coordinates.
    yaw is rotation around LiDAR z-axis.
    """
    h, w, l = obj['dimensions']
    x, y, z = obj['location']
    ry = obj['rotation_y']

    # Camera-frame centre (mid-height)
    centre_cam = np.array([[x, y - h / 2, z]])
    centre_lidar = cam_to_lidar(centre_cam,
                                 calib['Tr_velo_to_cam'],
                                 calib['R0_rect'])[0]

    # yaw conversion: camera ry → LiDAR yaw
    yaw_lidar = -ry - np.pi / 2

    return np.array([centre_lidar[0], centre_lidar[1], centre_lidar[2],
                     l, w, h, yaw_lidar])


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def load_pointcloud(bin_path):
    """Load a Velodyne .bin file.  Returns (N, 4) float32 array."""
    pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pc


def count_points_in_bbox(pc, bbox_lidar):
    """Count LiDAR points inside the 3D bbox (approximate via BEV rectangle)."""
    cx, cy, cz, l, w, h, yaw = bbox_lidar
    pts = pc[:, :3]

    # Rotate points into bbox frame
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy
    local_z = pts[:, 2] - cz

    mask = (
        (np.abs(local_x) < l / 2) &
        (np.abs(local_y) < w / 2) &
        (np.abs(local_z) < h / 2)
    )
    return mask.sum()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

FILTER_CLASSES = {'Car'}


def filter_objects(objects, pc, calib,
                   max_occluded=2, max_truncated=0.5, min_points=10):
    """Apply KITTI filtering criteria."""
    valid = []
    for obj in objects:
        if obj['type'] not in FILTER_CLASSES:
            continue
        if obj['occluded'] > max_occluded:
            continue
        if obj['truncated'] > max_truncated:
            continue
        bbox = get_bbox_lidar(obj, calib)
        n_pts = count_points_in_bbox(pc, bbox)
        if n_pts < min_points:
            continue
        valid.append(obj)
    return valid


class KITTIDataset(Dataset):
    """KITTI 3D Object Detection dataset for LiDAR-based detection."""

    def __init__(self, root, split='train', pc_range=None, filter_objs=True):
        """
        Args:
            root: path to data/kitti/
            split: 'train' or 'val'
            pc_range: [x_min,y_min,z_min,x_max,y_max,z_max]
            filter_objs: apply Car/occlusion/truncation/points filter
        """
        self.root = root
        self.split = split
        self.filter_objs = filter_objs
        if pc_range is None:
            pc_range = [0.0, -40.0, -2.5, 70.4, 40.0, 1.0]
        self.pc_range = np.array(pc_range, dtype=np.float32)

        split_file = os.path.join(root, 'ImageSets', f'{split}.txt')
        with open(split_file) as f:
            self.sample_ids = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.sample_ids)

    def _paths(self, sid):
        base = os.path.join(self.root, 'training')
        return {
            'pc': os.path.join(base, 'velodyne', f'{sid}.bin'),
            'label': os.path.join(base, 'label_2', f'{sid}.txt'),
            'calib': os.path.join(base, 'calib', f'{sid}.txt'),
        }

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        paths = self._paths(sid)

        pc = load_pointcloud(paths['pc'])       # (N, 4)
        calib = read_calib(paths['calib'])
        objects = read_label(paths['label'])

        if self.filter_objs:
            objects = filter_objects(objects, pc, calib)

        # Clip point cloud to configured range
        pc_xyz = pc[:, :3]
        lo, hi = self.pc_range[:3], self.pc_range[3:]
        mask = np.all((pc_xyz >= lo) & (pc_xyz <= hi), axis=1)
        pc = pc[mask]

        gt_bboxes = np.array(
            [get_bbox_lidar(o, calib) for o in objects], dtype=np.float32
        ) if objects else np.zeros((0, 7), dtype=np.float32)

        return {
            'sample_id': sid,
            'pointcloud': pc,               # (N, 4) float32
            'gt_bboxes': gt_bboxes,         # (M, 7) [cx,cy,cz,l,w,h,yaw]
            'gt_objects': objects,
            'calib': calib,
        }

    def collate_fn(self, batch):
        """Simple collate that keeps variable-length arrays as lists."""
        return {k: [b[k] for b in batch] for k in batch[0]}
