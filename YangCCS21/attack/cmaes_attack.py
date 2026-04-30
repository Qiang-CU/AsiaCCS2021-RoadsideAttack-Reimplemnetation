"""
CMA-ES black-box attack for AsiaCCS 2021 appearing attack.

Uses PCA-reduced parameterization to make CMA-ES tractable on high-dim mesh.
Directly optimizes the full detection pipeline output (score × detection_count),
bypassing gradient issues entirely.
"""

import os
import numpy as np
import torch
import cma

from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from model.pointrcnn_wrapper import PointRCNNWrapper


def _make_ground(ground_z=-0.75):
    rng = np.random.RandomState(123)
    x = rng.exponential(10, 3000).clip(0, 40).astype(np.float32)
    y = rng.uniform(-20, 20, 3000).astype(np.float32)
    z = np.full(3000, ground_z, dtype=np.float32)
    i = rng.uniform(0, 0.5, 3000).astype(np.float32)
    return np.stack([x, y, z, i], 1)


def run_cmaes_attack(dataset, config, save_dir='results',
                     warm_start_ckpt=None):
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    device = torch.device(config.get('device', 'cuda:0'))
    lidar_cfg = atk.get('lidar', {})
    object_pos = np.array(atk['object_pos'], dtype=np.float32)
    b = np.array(atk['size_limit'])

    n_evals = atk.get('cmaes_evals', 5000)
    pca_dim = atk.get('cmaes_pca_dim', 50)
    n_eval_views = atk.get('cmaes_eval_views', 5)
    sigma0 = atk.get('cmaes_sigma0', 0.3)

    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0_np = v0.numpy()
    faces_t = faces.to(device)
    n_verts = v0.shape[0]
    sensor_origin = torch.zeros(3, device=device)

    # PCA basis: random orthonormal directions in vertex space
    rng = np.random.RandomState(42)
    full_dim = n_verts * 3
    A = rng.randn(full_dim, pca_dim).astype(np.float32)
    A, _ = np.linalg.qr(A)
    A = A[:, :pca_dim]

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location='cpu', weights_only=False)
        offset0 = ckpt['offset'].numpy().flatten()
        z0 = np.linalg.lstsq(A, offset0, rcond=None)[0]
        print(f"Warm-started from {warm_start_ckpt}")
        print(f"  Reconstruction error: "
              f"{np.linalg.norm(A @ z0 - offset0):.4f} / {np.linalg.norm(offset0):.4f}")
    else:
        z0 = np.zeros(pca_dim, dtype=np.float64)

    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=str(device), enable_ste=False)

    ground_z = config.get('pose_sweep', {}).get('ground_z', -0.75)
    ground = _make_ground(ground_z)
    ground_t = torch.tensor(ground, dtype=torch.float32, device=device)

    eval_views = [
        np.array([0.0, 0.0, 0.75]),
        np.array([2.0, 1.0, 0.75]),
        np.array([-2.0, -1.0, 0.75]),
        np.array([1.0, -0.5, 0.75]),
        np.array([-1.0, 0.5, 0.75]),
    ]

    best_score = -np.inf
    best_offset = None
    eval_count = [0]

    def objective(z):
        z = np.array(z, dtype=np.float32)
        offset = (A @ z).reshape(n_verts, 3)

        # Apply physical constraints
        verts = v0_np + offset
        ctr = verts.mean(axis=0, keepdims=True)
        verts = verts - ctr
        for d in range(3):
            verts[:, d] = np.clip(verts[:, d], -b[d], b[d])
        offset = verts + ctr - v0_np

        verts_t = torch.tensor(v0_np + offset, dtype=torch.float32,
                               device=device)

        total_score = 0.0
        n_det_total = 0

        with torch.no_grad():
            for vi in range(n_eval_views):
                lidar_pos = eval_views[vi].astype(np.float32)
                rel_pos = object_pos - lidar_pos
                inj_t = torch.tensor(rel_pos, dtype=torch.float32,
                                     device=device)
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
                merged = torch.cat([ground_t, a4], dim=0)
                boxes, scores = wrapper.detect(merged, 0.1)

                if len(scores) > 0:
                    ms = float(scores.max())
                    total_score += ms
                    n_det_total += 1

        eval_count[0] += 1
        # CMA-ES minimizes, so negate
        fitness = -(total_score + 0.1 * n_det_total)

        nonlocal best_score, best_offset
        if -fitness > best_score:
            best_score = -fitness
            best_offset = offset.copy()

        if eval_count[0] % 50 == 0:
            print(f"  Eval {eval_count[0]}: score={-fitness:.4f} "
                  f"det={n_det_total}/{n_eval_views} "
                  f"best={best_score:.4f}")

        return fitness

    print(f"\n{'='*60}")
    print(f"  CMA-ES Black-box Attack")
    print(f"  PCA dim: {pca_dim}, sigma0: {sigma0}")
    print(f"  Max evaluations: {n_evals}")
    print(f"  Eval views: {n_eval_views}")
    print(f"  Object at {object_pos.tolist()}")
    print(f"{'='*60}")

    opts = {
        'maxfevals': n_evals,
        'popsize': 20,
        'tolfun': 1e-8,
        'verb_disp': 100,
        'seed': 42,
    }

    es = cma.CMAEvolutionStrategy(z0.tolist(), sigma0, opts)
    es.optimize(objective)

    print(f"\nCMA-ES finished. Best score: {best_score:.4f}")

    if best_offset is not None:
        save_path = os.path.join(save_dir, 'adv_mesh_cmaes_final.pth')
        torch.save({
            'v0': v0, 'faces': faces,
            'offset': torch.tensor(best_offset, dtype=torch.float32),
            't': torch.zeros(3),
            'param_mode': 'mesh_offset',
            'b': torch.tensor(b),
            'history': {'best_score': best_score},
        }, save_path)
        print(f"Saved best mesh → {save_path}")

    wrapper.remove_hook()
    return best_offset, best_score
