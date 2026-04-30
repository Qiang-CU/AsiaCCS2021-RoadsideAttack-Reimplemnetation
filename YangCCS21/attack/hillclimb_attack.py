"""
Hill-climbing black-box attack with robust multi-ground evaluation.

Key insight: detection depends heavily on the surrounding ground points.
We evaluate each mesh across multiple ground seeds to find shapes that
produce robust detections regardless of ground configuration.
"""

import os
import numpy as np
import torch

from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from model.pointrcnn_wrapper import PointRCNNWrapper


def _make_ground(seed, ground_z=-0.75, n_pts=3000):
    rng = np.random.RandomState(seed)
    x = rng.exponential(10, n_pts).clip(0, 40).astype(np.float32)
    y = rng.uniform(-20, 20, n_pts).astype(np.float32)
    z = np.full(n_pts, ground_z, dtype=np.float32)
    i = rng.uniform(0, 0.5, n_pts).astype(np.float32)
    return np.stack([x, y, z, i], 1)


def eval_mesh(verts_t, faces_t, grounds, object_pos, views, device,
              wrapper, lidar_cfg, thresh=0.05):
    """Evaluate mesh across multiple views AND multiple ground configurations.

    Returns (fitness, total_detections, max_score).
    fitness = sum of detection scores across all (view, ground) combos
              + 0.1 * number of detections
    """
    total_score = 0.0
    n_det = 0
    max_score = 0.0
    sensor_origin = torch.zeros(3, device=device)

    with torch.no_grad():
        for view in views:
            lidar_pos = np.array(view, dtype=np.float32)
            rel_pos = object_pos - lidar_pos
            inj_t = torch.tensor(rel_pos, dtype=torch.float32, device=device)
            vw = verts_t + inj_t.unsqueeze(0)

            pts = render_adversarial_points(
                vw, faces_t, sensor_origin,
                n_elevation=lidar_cfg.get('n_elevation', 16),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0))

            if pts.shape[0] == 0:
                continue

            a4 = torch.cat([pts, torch.ones(pts.shape[0], 1,
                                             device=device)], dim=1)

            for ground_t in grounds:
                merged = torch.cat([ground_t, a4], dim=0)
                boxes, scores = wrapper.detect(merged, thresh)

                if len(scores) > 0:
                    s = float(scores.max())
                    total_score += s
                    max_score = max(max_score, s)
                    n_det += 1

    fitness = total_score + 0.2 * n_det
    return fitness, n_det, max_score


def run_hillclimb(dataset, config, save_dir='results',
                  warm_start_ckpt=None):
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    device = torch.device(config.get('device', 'cuda:0'))
    lidar_cfg = atk.get('lidar', {})
    object_pos = np.array(atk['object_pos'], dtype=np.float32)
    b = np.array(atk['size_limit'])

    n_iters = atk.get('hillclimb_iters', 10000)
    perturb_scale = atk.get('hillclimb_scale', 0.02)

    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0_np = v0.numpy()
    faces_t = faces.to(device)
    n_verts = v0.shape[0]

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location='cpu',
                          weights_only=False)
        offset = ckpt['offset'].numpy()
        print(f"Warm-started from {warm_start_ckpt}")
    else:
        offset = np.zeros((n_verts, 3), dtype=np.float32)

    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=str(device), enable_ste=False)

    ground_z = config.get('pose_sweep', {}).get('ground_z', -0.75)
    n_ground_seeds = 5
    grounds = []
    for gs in range(n_ground_seeds):
        g = _make_ground(gs * 17, ground_z)
        grounds.append(torch.tensor(g, dtype=torch.float32, device=device))

    views = [
        [0.0, 0.0, 0.75],
        [2.0, 1.0, 0.75],
        [-2.0, -1.0, 0.75],
        [1.0, -0.5, 0.75],
        [-1.0, 0.5, 0.75],
    ]

    total_combos = len(views) * len(grounds)

    def apply_constraints(off):
        verts = v0_np + off
        ctr = verts.mean(axis=0, keepdims=True)
        verts = verts - ctr
        for d in range(3):
            verts[:, d] = np.clip(verts[:, d], -b[d], b[d])
        return verts + ctr - v0_np

    offset = apply_constraints(offset)
    verts_t = torch.tensor(v0_np + offset, dtype=torch.float32, device=device)
    best_score, best_det, best_max = eval_mesh(
        verts_t, faces_t, grounds, object_pos, views, device,
        wrapper, lidar_cfg)

    print(f"\n{'='*60}")
    print(f"  Hill-climbing Attack (robust multi-ground)")
    print(f"  Initial: fitness={best_score:.4f} ({best_det}/{total_combos} det, "
          f"max_s={best_max:.4f})")
    print(f"  {n_iters} iters, scale={perturb_scale}")
    print(f"  {n_verts} verts, constraint={b.tolist()}")
    print(f"  {len(views)} views x {len(grounds)} grounds = "
          f"{total_combos} combos/eval")
    print(f"{'='*60}")

    best_offset = offset.copy()
    improved = 0
    rng = np.random.RandomState(42)

    for step in range(n_iters):
        if step % 3 == 0:
            scale = perturb_scale * 2
        elif step % 3 == 1:
            scale = perturb_scale
        else:
            scale = perturb_scale * 0.5

        n_perturb = max(1, rng.randint(1, min(20, n_verts)))
        vert_ids = rng.choice(n_verts, n_perturb, replace=False)
        perturbation = rng.randn(n_perturb, 3).astype(np.float32) * scale

        trial_offset = best_offset.copy()
        trial_offset[vert_ids] += perturbation
        trial_offset = apply_constraints(trial_offset)

        verts_t = torch.tensor(v0_np + trial_offset,
                               dtype=torch.float32, device=device)
        score, n_det, max_s = eval_mesh(
            verts_t, faces_t, grounds, object_pos, views, device,
            wrapper, lidar_cfg)

        if score > best_score:
            best_score = score
            best_det = n_det
            best_max = max_s
            best_offset = trial_offset.copy()
            improved += 1

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{n_iters}: fitness={best_score:.4f} "
                  f"({best_det}/{total_combos} det, max_s={best_max:.4f}) "
                  f"improved={improved}")

        if (step + 1) % 1000 == 0:
            spath = os.path.join(save_dir, 'adv_mesh_hillclimb_latest.pth')
            torch.save({
                'v0': v0, 'faces': faces,
                'offset': torch.tensor(best_offset, dtype=torch.float32),
                't': torch.zeros(3),
                'param_mode': 'mesh_offset',
                'b': torch.tensor(b),
                'history': {'best_score': best_score, 'best_det': best_det,
                            'best_max': best_max, 'improved': improved},
            }, spath)

    fpath = os.path.join(save_dir, 'adv_mesh_hillclimb_final.pth')
    torch.save({
        'v0': v0, 'faces': faces,
        'offset': torch.tensor(best_offset, dtype=torch.float32),
        't': torch.zeros(3),
        'param_mode': 'mesh_offset',
        'b': torch.tensor(b),
        'history': {'best_score': best_score, 'best_det': best_det,
                    'best_max': best_max, 'improved': improved},
    }, fpath)

    wrapper.remove_hook()
    print(f"\nDone. Best: fitness={best_score:.4f} "
          f"({best_det}/{total_combos} det, max_s={best_max:.4f})")
    print(f"Saved -> {fpath}")
    return best_offset, best_score
