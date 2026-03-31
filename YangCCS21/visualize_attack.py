"""
Visualize point-opt adversarial attack results using matplotlib (no Open3D needed).

Generates per-sample figures:
  attack_3d_{id}.png — 3D close-up: scene LiDAR + adv points + reconstructed mesh

Usage:
    python visualize_attack.py --ckpt results/adv_points_pointopt_final.pth \
        --config configs/attack_config.yaml --gpu 0 --n-samples 5

    # With detection overlay
    python visualize_attack.py --ckpt results/adv_points_pointopt_final.pth \
        --config configs/attack_config.yaml --gpu 0 --n-samples 5 --detect
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(__file__))

from attack.inject import (
    build_bev_occupancy, sample_injection_position, inject_points,
)
from attack.whitebox_pointopt import precompute_injections
from utils.kitti_utils import KITTIDataset


# ── geometry helpers ────────────────────────────────────────────────────────

def bbox_corners_3d(bbox):
    """8 corners of a 7-DoF bbox [cx,cy,cz,l,w,h,yaw]."""
    cx, cy, cz, l, w, h, yaw = bbox
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = np.array([
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
    ])
    return (R @ local.T).T + np.array([cx, cy, cz])


def points_in_bbox(pc, bbox, margin=0.15):
    """Mask of points inside oriented bbox (with margin)."""
    cx, cy, cz, l, w, h, yaw = bbox
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy, dz = pc[:, 0] - cx, pc[:, 1] - cy, pc[:, 2] - cz
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return ((np.abs(lx) < l / 2 + margin) &
            (np.abs(ly) < w / 2 + margin) &
            (np.abs(dz) < h / 2 + margin))


def draw_bbox_3d(ax, bbox, color='green', lw=1.5, label=None):
    """Draw 3D wireframe bbox."""
    corners = bbox_corners_3d(bbox)
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    for i, (a, b) in enumerate(edges):
        ax.plot3D(*zip(corners[a], corners[b]), color=color, lw=lw,
                  label=label if i == 0 else None)


def draw_mesh_3d(ax, verts, faces_np, color, alpha=0.3, label=None):
    """Draw 3D triangle mesh."""
    triangles = verts[faces_np]
    col = Poly3DCollection(triangles, alpha=alpha, facecolor=color,
                           edgecolor=color, linewidth=0.3)
    ax.add_collection3d(col)
    if label:
        ax.plot([], [], [], color=color, label=label)


# ── mesh reconstruction (lightweight, scipy fallback) ──────────────────────

def reconstruct_mesh(points, method='alpha'):
    """Reconstruct a triangle mesh from point cloud. Returns (verts, faces)."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        if method == 'alpha':
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=0.15
            )
        elif method == 'ball_pivoting':
            dists = pcd.compute_nearest_neighbor_distance()
            avg = np.mean(dists)
            radii = [avg * f for f in [0.5, 1.0, 1.5, 2.0, 3.0]]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=0.15
            )
        mesh.compute_vertex_normals()
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    except ImportError:
        pass

    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return points[hull.vertices], hull.simplices


# ── injection position ─────────────────────────────────────────────────────

def compute_injection_pos(sample, inj_cfg):
    """Compute injection position for a sample."""
    occ, gi = build_bev_occupancy(
        sample['pointcloud'], sample['gt_bboxes'],
        x_range=tuple(inj_cfg['x_range']),
        y_range=tuple(inj_cfg['y_range']),
        resolution=inj_cfg['resolution'],
        margin=inj_cfg['margin'],
    )
    pos, valid = sample_injection_position(
        occ, gi,
        min_clearance=inj_cfg['min_clearance'],
        fallback_pos=tuple(inj_cfg['fallback_pos']),
    )
    return pos, valid


# ── 3-panel 3D close-up ───────────────────────────────────────────────────

def visualize_3d_closeup(sample, adv_pts_local, inj_pos, cfg,
                         wrapper=None, device=None, save_path='attack_3d.png',
                         mesh_method='alpha'):
    """
    3-panel 3D figure for point-opt adversarial attack:
      (a) Clean scene around injection position
      (b) Adversarial points injected
      (c) Reconstructed mesh + adversarial points + detections
    """
    pc_np = sample['pointcloud']
    gt_bboxes = sample['gt_bboxes']

    # Adversarial points in world frame
    adv_world = adv_pts_local + inj_pos[np.newaxis, :]
    n_adv = len(adv_world)

    # Extract nearby scene points (within radius of injection pos)
    radius = 6.0
    dxy = pc_np[:, :2] - inj_pos[:2]
    near_mask = np.sqrt((dxy ** 2).sum(axis=1)) < radius
    near_pts = pc_np[near_mask, :3]

    # Reconstruct mesh from adversarial points
    mesh_verts, mesh_faces = None, None
    if len(adv_pts_local) >= 4:
        try:
            mesh_verts, mesh_faces = reconstruct_mesh(adv_world, method=mesh_method)
        except Exception as e:
            print(f'    Mesh reconstruction failed: {e}')

    # Phantom bbox around adv points
    adv_center = adv_world.mean(axis=0)
    adv_extent = adv_world.max(axis=0) - adv_world.min(axis=0)
    phantom_bbox = np.array([
        adv_center[0], adv_center[1], adv_center[2],
        max(adv_extent[0], 0.5), max(adv_extent[1], 0.5),
        max(adv_extent[2], 0.5), 0.0
    ])

    # Run detection if wrapper provided
    det_boxes, det_scores = None, None
    if wrapper is not None and device is not None:
        merged_pc, _ = inject_points(pc_np, adv_pts_local, inj_pos,
                                     remove_overlap=True)
        score_thresh = cfg.get('model', {}).get('score_thresh', 0.3)
        pc_t = torch.tensor(merged_pc.astype(np.float32), device=device)
        det_boxes, det_scores = wrapper.detect(pc_t, score_thresh=score_thresh)

    # View angle
    azim = np.degrees(np.arctan2(inj_pos[1], inj_pos[0])) - 90
    elev = 25
    pad = 4.0

    fig = plt.figure(figsize=(22, 7))
    fig.suptitle(
        f'Sample {sample["sample_id"]} — Point-Opt Adversarial Attack  '
        f'({n_adv} adv pts, inject @ [{inj_pos[0]:.1f}, {inj_pos[1]:.1f}, {inj_pos[2]:.1f}])',
        fontsize=13, fontweight='bold'
    )

    panels = [
        ('(a) Clean Scene', False, False),
        ('(b) + Adversarial Points', True, False),
        ('(c) + Reconstructed Mesh & Detections', True, True),
    ]

    for idx, (title, show_adv, show_mesh) in enumerate(panels):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_title(title, fontsize=10, pad=10)

        # Nearby scene LiDAR points, colored by height
        if len(near_pts) > 0:
            z_vals = near_pts[:, 2]
            z_norm = (z_vals - z_vals.min()) / max(np.ptp(z_vals), 0.01)
            ax.scatter(near_pts[:, 0], near_pts[:, 1], near_pts[:, 2],
                       s=1.5, c=z_norm, cmap='coolwarm', alpha=0.5, zorder=1)

        # GT bboxes
        for i, bb in enumerate(gt_bboxes):
            draw_bbox_3d(ax, bb, color='green', lw=1.5,
                         label='GT bbox' if i == 0 else None)

        # Injection position marker
        ax.scatter([inj_pos[0]], [inj_pos[1]], [inj_pos[2]],
                   s=120, c='yellow', edgecolors='black', marker='*',
                   zorder=10, label='Inject pos')

        # Adversarial points
        if show_adv:
            ax.scatter(adv_world[:, 0], adv_world[:, 1], adv_world[:, 2],
                       s=8, c='red', alpha=0.9, zorder=5,
                       label=f'Adv pts ({n_adv})')

        # Reconstructed mesh
        if show_mesh and mesh_verts is not None and mesh_faces is not None:
            if len(mesh_faces) > 0:
                draw_mesh_3d(ax, mesh_verts, mesh_faces,
                             color=(1.0, 0.2, 0.8, 0.35),
                             alpha=0.3, label='Recon mesh')

        # Phantom bbox (dashed)
        if show_adv:
            draw_bbox_3d(ax, phantom_bbox, color='red', lw=1.0,
                         label='Phantom bbox')

        # Detection boxes
        if show_mesh and det_boxes is not None and len(det_boxes) > 0:
            near_dets = []
            for di, db in enumerate(det_boxes):
                d = np.sqrt((db[0] - inj_pos[0]) ** 2 +
                            (db[1] - inj_pos[1]) ** 2)
                if d < radius * 1.5:
                    near_dets.append((db, det_scores[di]))
            for di, (db, sc) in enumerate(near_dets):
                draw_bbox_3d(ax, db, color='orange', lw=1.8,
                             label=f'Det (conf≥{sc:.2f})' if di == 0 else None)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(inj_pos[0] - pad, inj_pos[0] + pad)
        ax.set_ylim(inj_pos[1] - pad, inj_pos[1] + pad)
        ax.set_zlim(inj_pos[2] - 2.0, inj_pos[2] + 2.0)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    return True


# ── BEV scene comparison ──────────────────────────────────────────────────

def visualize_bev_scene(sample, adv_pts_local, inj_pos,
                        wrapper, device, cfg, save_path):
    """
    2-panel BEV figure:
      Left:  clean point cloud + clean detections
      Right: adversarial point cloud + adversarial detections
    """
    pc_np = sample['pointcloud']
    gt_bboxes = sample['gt_bboxes']
    st = cfg['model']['score_thresh']

    pc_t = torch.tensor(pc_np.astype(np.float32), device=device)
    pred_c, scores_c = wrapper.detect(pc_t, score_thresh=st)

    merged_pc, n_adv = inject_points(pc_np, adv_pts_local, inj_pos,
                                     remove_overlap=True)
    adv_world = adv_pts_local + inj_pos[np.newaxis, :]
    pc_adv_t = torch.tensor(merged_pc.astype(np.float32), device=device)
    pred_a, scores_a = wrapper.detect(pc_adv_t, score_thresh=st)

    cx, cy = inj_pos[0], inj_pos[1]
    r = 25

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f'Sample {sample["sample_id"]}  |  GT: {len(gt_bboxes)} cars  |  '
        f'Clean dets: {len(pred_c)}  →  Adv dets: {len(pred_a)}  |  '
        f'Adv pts: {n_adv}',
        fontsize=13, fontweight='bold'
    )

    for col, (ax, title, preds, extra) in enumerate([
        (axes[0], 'Clean Point Cloud', pred_c, None),
        (axes[1], 'Adversarial Point Cloud', pred_a, adv_world),
    ]):
        ax.set_title(title, fontsize=12)
        ax.set_facecolor('#1a1a2e')

        mask = ((np.abs(pc_np[:, 0] - cx) < r) &
                (np.abs(pc_np[:, 1] - cy) < r))
        z = pc_np[mask, 2]
        ax.scatter(pc_np[mask, 0], pc_np[mask, 1],
                   s=0.08, c=z, cmap='gray', alpha=0.5, vmin=-2, vmax=1)

        if extra is not None and len(extra) > 0:
            ax.scatter(extra[:, 0], extra[:, 1],
                       s=3, c='red', alpha=0.9, zorder=4,
                       label=f'Adv pts ({len(extra)})')

        for i, bb in enumerate(gt_bboxes):
            corners = bbox_corners_3d(bb)[:4, :2]
            poly = plt.Polygon(corners, fill=False, edgecolor='lime',
                               lw=2, label='GT' if i == 0 else None)
            ax.add_patch(poly)

        for i, bb in enumerate(preds):
            corners = bbox_corners_3d(bb)[:4, :2]
            poly = plt.Polygon(corners, fill=False, edgecolor='orange',
                               lw=1.5, label='Detection' if i == 0 else None)
            ax.add_patch(poly)

        ax.scatter([inj_pos[0]], [inj_pos[1]], s=60, c='yellow',
                   edgecolors='black', marker='*', zorder=10,
                   label='Inject pos')

        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#2a2a4a', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    return True


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Visualize point-opt adversarial attack (3D + BEV)')
    p.add_argument('--ckpt', default='results/adv_points_pointopt_final.pth',
                   help='Point-opt checkpoint')
    p.add_argument('--config', default='configs/attack_config.yaml')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n-samples', type=int, default=5)
    p.add_argument('--out-dir', default='results')
    p.add_argument('--detect', action='store_true',
                   help='Run PointRCNN detection and overlay results')
    p.add_argument('--mesh-method', default='alpha',
                   choices=['alpha', 'ball_pivoting'],
                   help='Mesh reconstruction method')
    p.add_argument('--bev', action='store_true',
                   help='Also generate BEV scene comparison')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load adversarial points
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    adv_pts = ckpt['adv_points'].numpy()
    print(f'Loaded {args.ckpt}: {len(adv_pts)} points')
    print(f'  Extent: {adv_pts.max(0) - adv_pts.min(0)}')

    # Optionally load PointRCNN
    wrapper = None
    if args.detect or args.bev:
        from model.pointrcnn_wrapper import PointRCNNWrapper
        wrapper = PointRCNNWrapper(
            config_path=cfg['model']['pointrcnn_config'],
            ckpt_path=cfg['model']['pointrcnn_ckpt'],
            device=device,
        )
        print(f'Loaded PointRCNN on {device}')

    # Dataset
    val_ds = KITTIDataset(cfg['data']['kitti_root'], split=cfg['data']['split'],
                          pc_range=cfg['data']['pc_range'], filter_objs=True)
    inj_cfg = cfg['attack']['injection']

    # Use same seed=42 as training to get consistent injection positions
    print("Pre-computing injection positions (seed=42)...")
    injection_cache, valid_indices = precompute_injections(val_ds, inj_cfg)
    print(f"  {len(valid_indices)}/{len(val_ds)} valid positions")

    count = 0
    for idx in valid_indices:
        if count >= args.n_samples:
            break

        sample = val_ds[idx]
        inj_pos = injection_cache[idx]['pos']

        sid = sample['sample_id']
        print(f'\n[{count + 1}/{args.n_samples}] Sample {sid}: '
              f'{len(sample["gt_bboxes"])} cars, '
              f'{len(sample["pointcloud"])} pts, '
              f'inject @ [{inj_pos[0]:.1f}, {inj_pos[1]:.1f}, {inj_pos[2]:.1f}]')

        # 3D close-up
        p1 = os.path.join(args.out_dir, f'attack_3d_{sid}.png')
        visualize_3d_closeup(
            sample, adv_pts, inj_pos, cfg,
            wrapper=wrapper, device=device,
            save_path=p1, mesh_method=args.mesh_method,
        )
        print(f'  → {p1}')

        # BEV scene
        if args.bev and wrapper is not None:
            p2 = os.path.join(args.out_dir, f'attack_scene_{sid}.png')
            visualize_bev_scene(sample, adv_pts, inj_pos,
                                wrapper, device, cfg, p2)
            print(f'  → {p2}')

        count += 1

    print(f'\nDone. {count} samples → {args.out_dir}/')


if __name__ == '__main__':
    main()
