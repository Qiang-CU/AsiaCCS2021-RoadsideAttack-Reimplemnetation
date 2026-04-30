"""
LiDAR pose sweep evaluation (AsiaCCS 2021, Table 3).

Paper protocol:
  - Object fixed at (4, -2, 0)
  - LiDAR placed at (x, y, z) where x∈[-3,3], y∈[-1,1], z∈[0.7,0.8]
  - 900 total poses (30 x-steps × 30 y-steps, z fixed at midpoint)
  - Detection threshold: 0.3
  - Report: mis-classification rate = detected_as_car / total

Implementation: LiDAR at origin, object at relative position (4-x, -2-y, 0-z).
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from attack.renderer import render_adversarial_points
from attack.inject import inject_points
from attack.ground import generate_ground_raytrace


def generate_sweep_poses(config):
    """
    Generate the 900-pose grid from paper's LiDAR position ranges.

    Returns:
        object_positions: (N, 3) relative object positions (LiDAR at origin)
        lidar_positions:  (N, 3) actual LiDAR positions in world frame
    """
    ps = config['pose_sweep']
    obj_pos = np.array(ps['object_pos'], dtype=np.float32)
    n_x = ps['n_x']
    n_y = ps['n_y']
    x_range = ps['lidar_x_range']
    y_range = ps['lidar_y_range']
    z_val = ps['lidar_z']

    xs = np.linspace(x_range[0], x_range[1], n_x)
    ys = np.linspace(y_range[0], y_range[1], n_y)

    object_positions = []
    lidar_positions = []
    for x in xs:
        for y in ys:
            lidar_pos = np.array([x, y, z_val], dtype=np.float32)
            rel_pos = obj_pos - lidar_pos
            object_positions.append(rel_pos)
            lidar_positions.append(lidar_pos)

    return np.array(object_positions), np.array(lidar_positions)


def generate_ground_plane(ground_z=-0.75, n_points=3000):
    """Deterministic ground plane via LiDAR ray tracing."""
    return generate_ground_raytrace(ground_z=ground_z, h_step_deg=0.5)


def _load_adv_object(ckpt_path, ckpt_type, device):
    """Load adversarial object from checkpoint."""
    if ckpt_type == 'mesh':
        from attack.whitebox import load_mesh_checkpoint
        loaded = load_mesh_checkpoint(ckpt_path, map_location=device)
        v0 = loaded['v0'].to(device)
        mesh_param = loaded['mesh_param'].to(device)
        translation_param = loaded['translation_param'].to(device)
        faces = loaded['faces'].to(device)

        with torch.no_grad():
            local_verts = v0 + mesh_param + translation_param.unsqueeze(0)

        extent = (local_verts.max(0).values - local_verts.min(0).values)
        print(f"  Mesh: {local_verts.shape[0]} vertices, {faces.shape[0]} faces")
        print(f"  Size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")

        return {
            'type': 'mesh',
            'local_verts': local_verts,
            'faces': faces,
        }

    elif ckpt_type == 'pointopt':
        ckpt = torch.load(ckpt_path, map_location=device)
        adv_points = ckpt['adv_points'].to(device)
        extent = adv_points.max(0).values - adv_points.min(0).values
        print(f"  PointOpt: {adv_points.shape[0]} points")
        print(f"  Size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")
        return {
            'type': 'pointopt',
            'adv_points': adv_points,
        }

    raise ValueError(f"Unknown ckpt_type: {ckpt_type}")


def _detect_checkpoint_type(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'adv_points' in ckpt:
        return 'pointopt'
    if 'v0' in ckpt and 'faces' in ckpt:
        return 'mesh'
    raise ValueError(f"Cannot determine type: {ckpt_path}")


def compute_pose_sweep_asr(config, adv_ckpt_path,
                           ckpt_type=None,
                           device='cuda:0',
                           model_type='pointrcnn',
                           **kwargs):
    """
    Compute ASR using paper's Table 3 protocol.

    Object at (4,-2,0), LiDAR at x∈[-3,3], y∈[-1,1], z~0.75.

    Args:
        model_type: 'pointrcnn' or 'pointpillar'
    """
    dev = torch.device(device)

    if ckpt_type is None:
        ckpt_type = _detect_checkpoint_type(adv_ckpt_path)
        print(f"  Auto-detected checkpoint type: {ckpt_type}")

    atk = config['attack']
    lidar_cfg = atk.get('lidar', {})
    score_thresh = config['model']['score_thresh']

    # Generate sweep poses
    obj_positions, lidar_positions = generate_sweep_poses(config)
    n_total = len(obj_positions)
    print(f"  Sweep: {n_total} poses")

    # Load adversarial object
    obj = _load_adv_object(adv_ckpt_path, ckpt_type, dev)

    # Ground plane
    ps = config.get('pose_sweep', {})
    ground_z = ps.get('ground_z', -0.75)
    bg_pc = generate_ground_plane(ground_z=ground_z)
    print(f"  Background: ground ({len(bg_pc)} points)")

    # Model
    if model_type == 'pointpillar':
        from model.pointpillar_wrapper import PointPillarWrapper
        wrapper = PointPillarWrapper(
            config['model']['pointpillar_config'],
            config['model']['pointpillar_ckpt'],
            device=str(dev),
        )
        print(f"  Model: PointPillar")
    else:
        from model.pointrcnn_wrapper import PointRCNNWrapper
        wrapper = PointRCNNWrapper(
            config['model']['pointrcnn_config'],
            config['model']['pointrcnn_ckpt'],
            device=str(dev),
        )
        print(f"  Model: PointRCNN")

    sensor_pos = torch.zeros(3, device=dev)
    results = []

    for idx in tqdm(range(n_total), desc='Pose sweep'):
        rel_pos = obj_positions[idx]
        lidar_pos = lidar_positions[idx]
        pos_t = torch.tensor(rel_pos, dtype=torch.float32, device=dev)

        if obj['type'] == 'mesh':
            with torch.no_grad():
                verts_world = obj['local_verts'] + pos_t.unsqueeze(0)
                adv_pts = render_adversarial_points(
                    verts_world, obj['faces'], sensor_pos,
                    n_elevation=lidar_cfg.get('n_elevation', 16),
                    elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                    elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                    h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                    margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
                )

            if adv_pts.shape[0] == 0:
                results.append({
                    'pose_idx': idx,
                    'lidar_pos': lidar_pos.tolist(),
                    'object_rel_pos': rel_pos.tolist(),
                    'distance': float(np.linalg.norm(rel_pos[:2])),
                    'n_adv_pts': 0, 'detected': False, 'best_score': 0.0,
                    'reason': 'no_hit_points',
                })
                continue

            adv_pts_np = adv_pts.cpu().numpy()
            merged, n_adv = inject_points(
                bg_pc.copy(), adv_pts_np, np.zeros(3), remove_overlap=True,
            )
        else:
            adv_pts_np = obj['adv_points'].cpu().numpy()
            merged, n_adv = inject_points(
                bg_pc.copy(), adv_pts_np, rel_pos, remove_overlap=True,
            )

        merged_t = torch.from_numpy(merged.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_boxes, pred_scores = wrapper.detect(merged_t, score_thresh)

        # Check if any detection exists near the object position
        detected = False
        best_score = 0.0
        if len(pred_scores) > 0:
            dists = np.sqrt(
                (pred_boxes[:, 0] - rel_pos[0])**2 +
                (pred_boxes[:, 1] - rel_pos[1])**2
            )
            near = dists < 3.0
            if near.any():
                detected = True
                best_score = float(pred_scores[near].max())

        results.append({
            'pose_idx': idx,
            'lidar_pos': lidar_pos.tolist(),
            'object_rel_pos': rel_pos.tolist(),
            'distance': float(np.linalg.norm(rel_pos[:2])),
            'n_adv_pts': n_adv,
            'detected': detected,
            'best_score': best_score,
        })

    if hasattr(wrapper, 'remove_hook'):
        wrapper.remove_hook()

    # Statistics
    n_detected = sum(1 for r in results if r['detected'])
    n_no_hit = sum(1 for r in results if r.get('reason') == 'no_hit_points')
    asr = n_detected / max(n_total, 1)

    # Distance-binned ASR
    dist_bins = {}
    for r in results:
        d = r['distance']
        lo = int(d) if d < 10 else int(d // 2) * 2
        key = f"{lo:.0f}-{lo+1:.0f}m"
        if key not in dist_bins:
            dist_bins[key] = {'total': 0, 'detected': 0}
        dist_bins[key]['total'] += 1
        if r['detected']:
            dist_bins[key]['detected'] += 1

    print(f"\n  {'Distance':<12} {'ASR':>10} {'Detail':>15}")
    print(f"  {'-'*40}")
    for k in sorted(dist_bins.keys(), key=lambda s: float(s.split('-')[0])):
        b = dist_bins[k]
        b_asr = b['detected'] / max(b['total'], 1)
        print(f"  {k:<12} {b_asr*100:>8.1f}% {b['detected']}/{b['total']:>10}")

    print(f"\n  Overall: {n_detected}/{n_total} = {asr*100:.1f}%")
    if n_no_hit > 0:
        print(f"  ({n_no_hit} poses had 0 hit points)")

    stats = {
        'n_total': n_total,
        'n_detected': n_detected,
        'n_no_hit': n_no_hit,
        'asr': asr,
        'asr_pct': f'{asr*100:.1f}%',
        'per_pose': results,
        'dist_bins': dist_bins,
    }

    return asr, stats


def plot_pose_sweep_heatmap(stats, config, save_path='results/pose_sweep_heatmap.png'):
    """Plot LiDAR x × y heatmap of detection results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    ps = config['pose_sweep']
    n_x = ps['n_x']
    n_y = ps['n_y']

    grid = np.zeros((n_x, n_y))
    scores = np.zeros((n_x, n_y))
    for r in stats['per_pose']:
        xi = r['pose_idx'] // n_y
        yi = r['pose_idx'] % n_y
        grid[xi, yi] = 1.0 if r['detected'] else 0.0
        scores[xi, yi] = r['best_score']

    x_range = ps['lidar_x_range']
    y_range = ps['lidar_y_range']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(
        grid, aspect='auto', origin='lower',
        extent=[y_range[0], y_range[1], x_range[0], x_range[1]],
        cmap='RdYlGn', vmin=0, vmax=1,
    )
    ax.set_xlabel('LiDAR y')
    ax.set_ylabel('LiDAR x')
    ax.set_title(f"Detection (ASR = {stats['asr']*100:.1f}%)")
    plt.colorbar(im, ax=ax, label='Detected')

    ax = axes[1]
    im2 = ax.imshow(
        scores, aspect='auto', origin='lower',
        extent=[y_range[0], y_range[1], x_range[0], x_range[1]],
        cmap='hot', vmin=0, vmax=1,
    )
    ax.set_xlabel('LiDAR y')
    ax.set_ylabel('LiDAR x')
    ax.set_title('Best detection confidence')
    plt.colorbar(im2, ax=ax, label='Score')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved to {save_path}")
