"""
Visualisation utilities for the LiDAR adversarial attack.

- Open3D point cloud + GT bbox visualisation
- Recall-IoU curve plots
- Loss curve plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-safe default
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Open3D helpers (imported lazily to avoid hard dependency at import time)
# ---------------------------------------------------------------------------

def _o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        raise ImportError('open3d is required for 3D visualisation. '
                          'Run: pip install open3d')


def pc_to_o3d(pc_np, color=None):
    """Convert (N, 3+) numpy array to Open3D PointCloud."""
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
    if color is not None:
        cols = np.tile(np.array(color, dtype=np.float64), (len(pc_np), 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def bbox_to_o3d_lineset(bbox, color=(1, 0, 0)):
    """
    Convert a 7-DoF bbox [cx,cy,cz,l,w,h,yaw] to an Open3D LineSet wireframe.
    """
    o3d = _o3d()
    cx, cy, cz, l, w, h, yaw = bbox
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    corners_local = np.array([
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
    ])
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]], dtype=np.float64)
    corners = (R @ corners_local.T).T + np.array([cx, cy, cz])

    lines = [[0,1],[1,2],[2,3],[3,0],
             [4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def visualise_pointcloud_with_bboxes(pc_np, gt_bboxes=None, adv_pts=None,
                                      rooftop=None, window_title='LiDAR scene',
                                      save_path=None):
    """
    Visualise a point cloud scene with GT bboxes and optional adversarial points.

    Args:
        pc_np:       (N, 3+) original point cloud
        gt_bboxes:   (M, 7) or None
        adv_pts:     (K, 3) adversarial points or None
        rooftop:     (3,)   rooftop centre marker or None
        window_title: str
        save_path:   if set, save screenshot instead of interactive window
    """
    o3d = _o3d()
    geoms = []

    # Original point cloud – grey
    geoms.append(pc_to_o3d(pc_np, color=[0.6, 0.6, 0.6]))

    # GT bboxes – green
    if gt_bboxes is not None:
        for bb in gt_bboxes:
            geoms.append(bbox_to_o3d_lineset(bb, color=(0, 1, 0)))

    # Adversarial points – red
    if adv_pts is not None and len(adv_pts) > 0:
        geoms.append(pc_to_o3d(adv_pts, color=[1, 0, 0]))

    # Rooftop marker – yellow sphere
    if rooftop is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(rooftop[:3])
        sphere.paint_uniform_color([1, 1, 0])
        geoms.append(sphere)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, visible=False,
                          width=1280, height=720)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f'Saved visualisation → {save_path}')
    else:
        o3d.visualization.draw_geometries(geoms, window_name=window_title)


# ---------------------------------------------------------------------------
# Recall-IoU curve
# ---------------------------------------------------------------------------

def plot_recall_iou(iou_thresholds, recall_clean, recall_adv=None,
                    save_path='recall_iou.png', title='Recall-IoU Curve'):
    """
    Plot recall as a function of IoU threshold (clean vs. adversarial).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iou_thresholds, recall_clean, 'b-o', markersize=3,
            label='Clean')
    if recall_adv is not None:
        ax.plot(iou_thresholds, recall_adv, 'r-o', markersize=3,
                label='Adversarial')
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Recall')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved Recall-IoU plot → {save_path}')


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(history, save_path='attack_loss.png'):
    """
    Plot adversarial optimisation loss curves.

    Args:
        history: dict mapping label → list of values.
                 Accepts both old-style keys ('loss', 'loss_adv', 'loss_lap')
                 and new-style keys ('L_total', 'L_cls', etc.).
    """
    # Filter to keys that have non-empty list values
    plot_items = [(k, v) for k, v in history.items()
                  if isinstance(v, list) and len(v) > 0]
    if not plot_items:
        print(f'WARNING: no plottable data in history, skipping {save_path}')
        return

    n = len(plot_items)
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple',
              'brown', 'gray']
    fig, axes = plt.subplots(1, min(n, 4), figsize=(4.5 * min(n, 4), 4),
                             squeeze=False)
    axes = axes.flatten()
    for i, (key, vals) in enumerate(plot_items[:4]):
        ax = axes[i]
        c = colors[i % len(colors)]
        ax.plot(vals, color=c, linewidth=0.8, alpha=0.5)
        # Smoothed
        if len(vals) >= 20:
            kernel = np.ones(20) / 20
            smoothed = np.convolve(vals, kernel, mode='valid')
            ax.plot(range(19, len(vals)), smoothed, color=c,
                    linewidth=2.0, label='Smoothed')
        ax.set_title(key)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved loss curve → {save_path}')
