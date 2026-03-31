"""
BEV occupancy grid and blank region injection pipeline for appearing attack.

Provides:
- BEV occupancy map construction from point clouds and GT boxes
- Blank region sampling (finds free cells for adversarial object placement)
- Point cloud injection (merge adversarial points into scene)
"""

import numpy as np
import torch


def build_bev_occupancy(points, gt_boxes,
                        x_range=(2, 35), y_range=(-15, 15),
                        resolution=0.5, margin=0.5):
    """
    Build a 2D BEV occupancy grid for a single frame.

    Cells are marked occupied if:
    - Any GT bounding box (with `margin` padding) overlaps the cell
    - The cell contains more than 2 raw LiDAR points

    Args:
        points:     (N, 3+) numpy array, LiDAR coordinates
        gt_boxes:   (M, 7) numpy array [cx, cy, cz, l, w, h, yaw]
        x_range:    (x_min, x_max) in LiDAR frame
        y_range:    (y_min, y_max) in LiDAR frame
        resolution: grid cell size in meters
        margin:     extra padding around GT boxes in meters

    Returns:
        occupancy: 2D bool numpy array (nx, ny), True = occupied
        grid_info: dict with x_range, y_range, resolution, nx, ny
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    nx = int((x_max - x_min) / resolution)
    ny = int((y_max - y_min) / resolution)
    occupancy = np.zeros((nx, ny), dtype=bool)

    grid_info = {
        'x_range': x_range, 'y_range': y_range,
        'resolution': resolution, 'nx': nx, 'ny': ny,
    }

    # Mark cells occupied by raw point cloud (cells with > 2 points)
    pts_xy = points[:, :2]
    ix = ((pts_xy[:, 0] - x_min) / resolution).astype(int)
    iy = ((pts_xy[:, 1] - y_min) / resolution).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid], iy[valid]

    point_count = np.zeros((nx, ny), dtype=int)
    np.add.at(point_count, (ix, iy), 1)
    occupancy |= (point_count > 2)

    # Mark cells occupied by GT bounding boxes (with margin)
    for box in gt_boxes:
        cx, cy, _, l, w, _, yaw = box
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        half_l = l / 2 + margin
        half_w = w / 2 + margin

        # Corners of expanded box
        corners_local = np.array([
            [ half_l,  half_w],
            [-half_l,  half_w],
            [-half_l, -half_w],
            [ half_l, -half_w],
        ])
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = (R @ corners_local.T).T + np.array([cx, cy])

        # Rasterize: find grid cells inside the rotated box
        c_ix = ((corners[:, 0] - x_min) / resolution).astype(int)
        c_iy = ((corners[:, 1] - y_min) / resolution).astype(int)
        bx_min = max(0, c_ix.min())
        bx_max = min(nx - 1, c_ix.max())
        by_min = max(0, c_iy.min())
        by_max = min(ny - 1, c_iy.max())

        for gx in range(bx_min, bx_max + 1):
            for gy in range(by_min, by_max + 1):
                # Cell centre in world coordinates
                cell_x = x_min + (gx + 0.5) * resolution
                cell_y = y_min + (gy + 0.5) * resolution
                # Transform to box-local frame
                dx = cell_x - cx
                dy = cell_y - cy
                local_x = cos_y * dx + sin_y * dy
                local_y = -sin_y * dx + cos_y * dy
                if abs(local_x) <= half_l and abs(local_y) <= half_w:
                    occupancy[gx, gy] = True

    return occupancy, grid_info


def sample_injection_position(occupancy, grid_info,
                              n_candidates=20, min_clearance=1.5,
                              fallback_pos=(4.0, -2.0, 0.0),
                              rng=None):
    """
    Sample a random free cell from the BEV occupancy grid.

    Args:
        occupancy:     2D bool array from build_bev_occupancy
        grid_info:     dict with grid parameters
        n_candidates:  number of random candidates to try
        min_clearance: minimum free radius (meters) around chosen cell
        fallback_pos:  (x, y, z) fallback if no free cell found
        rng:           numpy RandomState (optional)

    Returns:
        pos:   (3,) numpy array [x, y, z=0] in LiDAR frame
        valid: bool, True if a free cell was found
    """
    if rng is None:
        rng = np.random.RandomState()

    x_min, _ = grid_info['x_range']
    y_min, _ = grid_info['y_range']
    res = grid_info['resolution']
    nx, ny = grid_info['nx'], grid_info['ny']
    clearance_cells = int(np.ceil(min_clearance / res))

    free_cells = np.argwhere(~occupancy)  # (K, 2) indices
    if len(free_cells) == 0:
        return np.array(fallback_pos, dtype=np.float32), False

    # Try n_candidates random free cells
    indices = rng.choice(len(free_cells), size=min(n_candidates, len(free_cells)),
                         replace=False)
    for idx in indices:
        gx, gy = free_cells[idx]
        # Check clearance: all cells in radius must be free
        x_lo = max(0, gx - clearance_cells)
        x_hi = min(nx, gx + clearance_cells + 1)
        y_lo = max(0, gy - clearance_cells)
        y_hi = min(ny, gy + clearance_cells + 1)
        patch = occupancy[x_lo:x_hi, y_lo:y_hi]
        if not patch.any():
            cell_x = x_min + (gx + 0.5) * res
            cell_y = y_min + (gy + 0.5) * res
            # z = typical KITTI car center height in Velodyne frame
            # (ground ≈ -1.82, car height ≈ 1.5m → center ≈ -1.07)
            return np.array([cell_x, cell_y, -1.0], dtype=np.float32), True

    return np.array(fallback_pos, dtype=np.float32), False


def inject_points(scene_points, adv_points, injection_pos,
                  remove_overlap=True):
    """
    Inject adversarial points into a scene point cloud.

    Args:
        scene_points: (N, 4) numpy or tensor, original scene [x,y,z,intensity]
        adv_points:   (M, 3) numpy or tensor, adversarial points (local frame)
        injection_pos: (3,) numpy or tensor, world position to place object
        remove_overlap: if True, remove original points inside adversarial AABB

    Returns:
        merged: (N+M, 4) combined point cloud
        n_adv:  int, number of adversarial points injected
    """
    is_tensor = isinstance(scene_points, torch.Tensor)

    if is_tensor:
        device = scene_points.device
        if not isinstance(adv_points, torch.Tensor):
            adv_points = torch.tensor(adv_points, dtype=torch.float32, device=device)
        if not isinstance(injection_pos, torch.Tensor):
            injection_pos = torch.tensor(injection_pos, dtype=torch.float32, device=device)

        # Translate adversarial points to injection position
        adv_world = adv_points + injection_pos.unsqueeze(0)
        n_adv = adv_world.shape[0]

        if n_adv == 0:
            return scene_points, 0

        if remove_overlap:
            aabb_min = adv_world.min(dim=0).values - 0.1
            aabb_max = adv_world.max(dim=0).values + 0.1
            pts_xyz = scene_points[:, :3]
            inside = ((pts_xyz >= aabb_min) & (pts_xyz <= aabb_max)).all(dim=1)
            scene_points = scene_points[~inside]

        # Add dummy intensity to adversarial points
        adv_4 = torch.cat([adv_world,
                           torch.ones(n_adv, 1, device=device)], dim=1)
        merged = torch.cat([scene_points, adv_4], dim=0)
    else:
        adv_points = np.asarray(adv_points)
        injection_pos = np.asarray(injection_pos)

        adv_world = adv_points + injection_pos[np.newaxis, :]
        n_adv = adv_world.shape[0]

        if n_adv == 0:
            return scene_points, 0

        if remove_overlap:
            aabb_min = adv_world.min(axis=0) - 0.1
            aabb_max = adv_world.max(axis=0) + 0.1
            pts_xyz = scene_points[:, :3]
            inside = np.all((pts_xyz >= aabb_min) & (pts_xyz <= aabb_max), axis=1)
            scene_points = scene_points[~inside]

        adv_4 = np.hstack([adv_world,
                           np.ones((n_adv, 1), dtype=np.float32)])
        merged = np.vstack([scene_points, adv_4])

    return merged, n_adv


def get_injection_metadata(frame_id, injection_pos, valid, gt_box_count):
    """
    Create injection metadata record for a frame.

    Args:
        frame_id:      str, e.g. '000042'
        injection_pos: (3,) array
        valid:         bool, True if position was freely sampled
        gt_box_count:  int, number of GT boxes in frame

    Returns:
        dict with injection metadata
    """
    return {
        'frame_id': frame_id,
        'injection_pos': injection_pos.tolist(),
        'injection_valid': bool(valid),
        'gt_box_count': int(gt_box_count),
    }
