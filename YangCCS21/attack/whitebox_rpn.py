"""
White-box appearing attack — RPN-only gradient optimization.

Both original PointRCNN and OpenPCDet have non-differentiable ROI pooling
(no backward pass in CUDA kernel). Gradient flows through:

  mesh → renderer → adv_pts → [ground | adv_pts]
    → PointNet++ backbone → point_features
    → RPN cls_layers → foreground logits
    → L_rpn_cls + L_feat_backbone → backward → mesh
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from attack.ground import generate_ground_raytrace
from attack.loss import (L_rpn_cls, L_feat_backbone, L_area)
from model.pointrcnn_wrapper import PointRCNNWrapper


def _sample_lidar(config, rng):
    atk = config['attack']
    return np.array([
        rng.uniform(*atk['lidar_x_range']),
        rng.uniform(*atk['lidar_y_range']),
        rng.uniform(*atk['lidar_z_range']),
    ], dtype=np.float32)


def run_whitebox_rpn(dataset, config, save_dir='results',
                     warm_start_ckpt=None):
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    lw = atk['loss_weights']
    device = torch.device(config.get('device', 'cuda:0'))
    rng = np.random.RandomState(42)

    n_iters = atk['n_iters']
    lr = atk['lr']
    grad_scale = atk.get('grad_scale', 10.0)
    n_views = atk.get('n_views_per_step', 4)
    lidar_cfg = atk.get('lidar', {})
    object_pos = np.array(atk['object_pos'], dtype=np.float32)

    alpha_feat = lw.get('alpha_feat', 0.1)
    gamma_area = lw['gamma_area']
    kappa = lw.get('kappa_rpn', 0.0)

    mesh_radius = atk.get('mesh_radius', 0.4)
    v0, faces, _ = create_icosphere(
        subdivisions=atk['mesh_subdivisions'], radius=mesh_radius)
    v0, faces = v0.to(device), faces.to(device)
    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location=device,
                          weights_only=False)
        mesh_param = ckpt['offset'].to(device).requires_grad_(True)
        print(f"Warm-started from {warm_start_ckpt}")
    else:
        mesh_param = torch.zeros_like(v0, requires_grad=True)

    ref_feat = None
    feat_path = config.get('ref_feature', {}).get(
        'backbone_path', 'results/ref_car_feature_backbone.pt')
    if os.path.exists(feat_path):
        ref_feat = torch.load(feat_path, map_location=device,
                              weights_only=True).squeeze()
        print(f"  Backbone ref: {ref_feat.shape}")

    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=str(device), enable_ste=False)

    optimizer = torch.optim.Adam([mesh_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

    ground_z = config.get('pose_sweep', {}).get('ground_z', -0.75)
    ground_np = generate_ground_raytrace(ground_z=ground_z, h_step_deg=0.5)
    ground_t = torch.tensor(ground_np, dtype=torch.float32, device=device)
    n_ground = ground_t.shape[0]
    sensor_origin = torch.zeros(3, device=device)

    history = {'L_total': [], 'L_cls': [], 'L_feat': [],
               'L_area': []}

    print(f"\n{'='*60}")
    print(f"  RPN-only White-box Attack")
    print(f"  {n_iters} iters, lr={lr}, views/step={n_views}")
    print(f"  Ground: {n_ground} raytrace pts (deterministic)")
    print(f"  Mesh: {v0.shape[0]} verts, radius={mesh_radius}")
    print(f"  Size limit: {atk['size_limit']}")
    print(f"  Loss = L_rpn_cls + {alpha_feat}*L_feat + "
          f"{gamma_area}*L_area")
    print(f"{'='*60}")

    monitor_every = max(n_iters // 20, 100)

    pbar = tqdm(range(n_iters), desc='RPN attack')
    for step in pbar:
        optimizer.zero_grad()
        step_ld = {}
        valid_views = 0

        progress = step / max(n_iters - 1, 1)
        if progress < 0.3:
            cur_n_views = 1
        elif progress < 0.6:
            cur_n_views = min(2, n_views)
        else:
            cur_n_views = n_views

        for v_idx in range(cur_n_views):
            if progress < 0.3:
                lidar_pos = np.array([0.0, 0.0, 0.75], dtype=np.float32)
            elif progress < 0.6:
                fixed_views = [
                    np.array([0.0, 0.0, 0.75], dtype=np.float32),
                    np.array([2.0, 1.0, 0.75], dtype=np.float32),
                    np.array([-2.0, -1.0, 0.75], dtype=np.float32),
                ]
                lidar_pos = fixed_views[v_idx % len(fixed_views)]
            else:
                lidar_pos = _sample_lidar(config, rng)
            rel_pos = object_pos - lidar_pos

            vertices = v0 + mesh_param
            inj_t = torch.tensor(rel_pos, dtype=torch.float32, device=device)
            vw = vertices + inj_t.unsqueeze(0)

            adv_pts = render_adversarial_points(
                vw, faces, sensor_origin,
                n_elevation=lidar_cfg.get('n_elevation', 16),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0))
            if adv_pts.shape[0] == 0:
                continue

            n_adv = adv_pts.shape[0]

            if grad_scale != 1.0:
                adv_pts = (adv_pts * grad_scale
                           + adv_pts.detach() * (1 - grad_scale))

            adv_4 = torch.cat([adv_pts,
                               torch.ones(n_adv, 1, device=device)], dim=1)
            merged = torch.cat([ground_t, adv_4], dim=0)

            result = wrapper.forward_with_grad(merged, rpn_only=True)

            l_cls = L_rpn_cls(result.get('point_cls_logits'),
                              n_ground, n_adv, kappa=kappa)
            l_feat = L_feat_backbone(result.get('point_features'),
                                     ref_feat, n_ground, n_adv)
            l_area = L_area(vertices, faces)

            loss = l_cls + alpha_feat * l_feat + gamma_area * l_area

            (loss / cur_n_views).backward()
            valid_views += 1

            ld = {'L_total': loss.item(), 'L_cls': l_cls.item(),
                  'L_feat': l_feat.item(), 'L_area': l_area.item(),
                  }
            for k, v in ld.items():
                step_ld[k] = step_ld.get(k, 0) + v / cur_n_views

        if valid_views > 0:
            torch.nn.utils.clip_grad_norm_([mesh_param], max_norm=5.0)
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            vs = v0 + mesh_param
            ctr = vs.mean(dim=0, keepdim=True)
            vs = vs - ctr
            for d in range(3):
                vs[:, d].data.clamp_(-b[d].item(), b[d].item())
            mesh_param.data.copy_(vs + ctr - v0)

        for k in history:
            history[k].append(step_ld.get(k, 0.0))

        pbar.set_postfix({
            'cls': f'{step_ld.get("L_cls", 0):.3f}',
            'feat': f'{step_ld.get("L_feat", 0):.3f}',
            'v': valid_views,
        })

        if (step + 1) % monitor_every == 0:
            with torch.no_grad():
                lp = np.array([0.0, 0.0, 0.75], dtype=np.float32)
                rp = object_pos - lp
                vv = v0 + mesh_param.detach()
                it = torch.tensor(rp, dtype=torch.float32, device=device)
                pts = render_adversarial_points(
                    vv + it.unsqueeze(0), faces, sensor_origin,
                    n_elevation=16, elev_min_deg=-15.0,
                    elev_max_deg=15.0, h_step_deg=0.2, margin_deg=2.0)
                if pts.shape[0] > 0:
                    a4 = torch.cat([pts, torch.ones(pts.shape[0], 1,
                                                     device=device)], dim=1)
                    m = torch.cat([ground_t, a4], dim=0)
                    pred_boxes, pred_scores = wrapper.detect(m, 0.3)
                    nd = len(pred_scores)
                    bs = float(pred_scores.max()) if nd > 0 else 0.0
                    print(f"\n  [Step {step+1}] hits={pts.shape[0]} "
                          f"det={nd} best={bs:.3f}")

        if (step + 1) % 1000 == 0:
            spath = os.path.join(save_dir, 'adv_mesh_whitebox_latest.pth')
            torch.save({
                'v0': v0.cpu(), 'faces': faces.cpu(),
                'offset': mesh_param.detach().cpu(),
                't': torch.zeros(3),
                'history': history, 'param_mode': 'mesh_offset',
                'b': b.cpu(),
                'mesh_radius': mesh_radius,
            }, spath)

    fpath = os.path.join(save_dir, 'adv_mesh_whitebox_final.pth')
    torch.save({
        'v0': v0.cpu(), 'faces': faces.cpu(),
        'offset': mesh_param.detach().cpu(),
        't': torch.zeros(3),
        'history': history, 'param_mode': 'mesh_offset',
        'b': b.cpu(),
        'mesh_radius': mesh_radius,
    }, fpath)

    wrapper.remove_hook()
    print(f"\nDone. Saved to {fpath}")
    return mesh_param.detach(), history
