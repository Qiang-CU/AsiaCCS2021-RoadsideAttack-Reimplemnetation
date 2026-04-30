"""
White-box appearing attack on PointRCNN (AsiaCCS 2021, Eq. 10).

Gradient chain:
  mesh vertices → LiDAR renderer → adv_pts
    → PointNet++ backbone → RPN → NMS (detach) → ROI pooling (STE) → RCNN head
    → L_cls + L_feat + L_box + L_area
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from attack.loss import appearing_loss
from attack.ground import generate_ground_raytrace


def _sample_lidar_position(config, rng):
    atk = config['attack']
    x = rng.uniform(*atk['lidar_x_range'])
    y = rng.uniform(*atk['lidar_y_range'])
    z = rng.uniform(*atk['lidar_z_range'])
    return np.array([x, y, z], dtype=np.float32)


def _object_relative_pos(object_pos, lidar_pos):
    return np.array(object_pos, dtype=np.float32) - lidar_pos


def load_mesh_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    return {
        'ckpt': ckpt,
        'mesh_param': ckpt['offset'],
        'translation_param': ckpt.get('t', torch.zeros(3)),
        'v0': ckpt['v0'],
        'faces': ckpt['faces'],
        'history': ckpt.get('history', {}),
        'b': ckpt.get('b'),
        'c': ckpt.get('c'),
        'param_mode': ckpt.get('param_mode', 'mesh_offset'),
    }


def _save_checkpoint(mesh_param, v0, faces, history,
                     save_dir, tag='latest', b=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'adv_mesh_whitebox_{tag}.pth')
    data = {
        'v0': v0.detach().cpu(),
        'faces': faces.detach().cpu(),
        'offset': mesh_param.detach().cpu(),
        't': torch.zeros(3),
        'translation_param': torch.zeros(3),
        'history': history,
        'param_mode': 'mesh_offset',
    }
    if b is not None:
        data['b'] = b.detach().cpu()
    torch.save(data, path)
    return path


def run_whitebox_attack(dataset, config, save_dir='results',
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

    alpha_rpn = lw.get('alpha_rpn', 0.1)
    alpha_feat = lw.get('alpha_feat', 0.1)
    beta_box = lw.get('beta_box', 0.1)
    gamma_area = lw['gamma_area']
    kappa_rcnn = lw.get('kappa_rcnn', 0.0)
    kappa_rpn = lw.get('kappa_rpn', 0.0)

    # Mesh
    mesh_radius = atk.get('mesh_radius', 0.4)
    v0, faces, _ = create_icosphere(subdivisions=atk['mesh_subdivisions'],
                                     radius=mesh_radius)
    v0, faces = v0.to(device), faces.to(device)
    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        if 'offset' in ckpt:
            mesh_param = ckpt['offset'].to(device).requires_grad_(True)
        elif 'mesh_param' in ckpt:
            mesh_param = ckpt['mesh_param'].to(device).float().requires_grad_(True)
        else:
            mesh_param = torch.zeros_like(v0, requires_grad=True)
        print(f"Warm-started from {warm_start_ckpt}")
    else:
        mesh_param = torch.zeros_like(v0, requires_grad=True)

    from model.pointrcnn_wrapper import PointRCNNWrapper
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=str(device), enable_ste=True)

    # Load precomputed reference features for feature/box matching losses
    ref_cfg = config.get('ref_feature', {})
    ref_backbone_feature = None
    ref_rcnn_feature = None
    ref_orientation = 0.0
    ref_box_size = torch.tensor([3.9, 1.6, 1.56], dtype=torch.float32)

    _ref_paths = {
        'backbone': ref_cfg.get('backbone_path', 'results/ref_car_feature_backbone.pt'),
        'rcnn': ref_cfg.get('rcnn_path', 'results/ref_car_feature_rcnn.pt'),
        'orientation': ref_cfg.get('orientations_path', 'results/ref_car_orientations.pt'),
        'box_size': ref_cfg.get('box_size_path', 'results/ref_car_box_size.pt'),
    }
    _missing = [k for k, p in _ref_paths.items() if not os.path.exists(p)]

    if _missing:
        print(f"\n  WARNING: reference feature files missing: {_missing}")
        print(f"  L_feat / L_box losses will be disabled (= 0).")
        print(f"  Run first:  python run_attack.py --mode precompute")
        print()

    if os.path.exists(_ref_paths['backbone']):
        ref_backbone_feature = torch.load(_ref_paths['backbone'], map_location=device, weights_only=False)
        print(f"  Loaded backbone ref: {_ref_paths['backbone']}")

    if os.path.exists(_ref_paths['rcnn']):
        ref_rcnn_feature = torch.load(_ref_paths['rcnn'], map_location=device, weights_only=False)
        print(f"  Loaded RCNN ref: {_ref_paths['rcnn']}")

    if os.path.exists(_ref_paths['orientation']):
        orient_data = torch.load(_ref_paths['orientation'], map_location=device, weights_only=False)
        ref_orientation = orient_data.get('rpn_orientation', 0.0)
        print(f"  Loaded ref orientation: {ref_orientation:.4f}")

    if os.path.exists(_ref_paths['box_size']):
        ref_box_size = torch.load(_ref_paths['box_size'], map_location=device, weights_only=False)
        print(f"  Loaded ref box size: {ref_box_size.tolist()}")

    optimizer = torch.optim.Adam([mesh_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

    ps_cfg = config.get('pose_sweep', {})
    ground_z = ps_cfg.get('ground_z', -0.75)
    ground_np = generate_ground_raytrace(ground_z=ground_z, h_step_deg=0.5)
    ground_t = torch.tensor(ground_np, dtype=torch.float32, device=device)

    history = {
        'L_total': [], 'L_rcnn_cls': [], 'L_rpn_cls': [],
        'L_rcnn_feat': [], 'L_backbone_feat': [], 'L_box': [],
        'L_area': [],
    }

    print(f"\n{'='*70}")
    print(f"  White-box Attack — PointRCNN (NMS → STE ROI pooling)")
    print(f"  {n_iters} iters, lr={lr}, grad_scale={grad_scale}, "
          f"views/step={n_views}")
    print(f"  Mesh: {v0.shape[0]} verts, radius={mesh_radius}")
    print(f"  Size limit: {atk['size_limit']}")
    print(f"  Ground: {ground_np.shape[0]} deterministic points")
    print(f"{'='*70}")

    monitor_every = max(n_iters // 20, 50)
    sensor_origin = torch.zeros(3, device=device)

    n_ground = ground_t.shape[0]

    pbar = tqdm(range(n_iters), desc='Whitebox (PointRCNN)')
    for step in pbar:
        optimizer.zero_grad()

        view_losses = []
        view_loss_dicts = []
        valid_views = 0

        for v_idx in range(n_views):
            lidar_pos = _sample_lidar_position(config, rng)
            rel_pos = _object_relative_pos(object_pos, lidar_pos)

            vertices = v0 + mesh_param
            inj_t = torch.tensor(rel_pos, dtype=torch.float32, device=device)
            vertices_world = vertices + inj_t.unsqueeze(0)

            adv_pts = render_adversarial_points(
                vertices_world, faces, sensor_origin,
                n_elevation=lidar_cfg.get('n_elevation', 16),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
            )

            if adv_pts.shape[0] == 0:
                continue

            n_adv = adv_pts.shape[0]

            if grad_scale != 1.0:
                adv_pts = adv_pts * grad_scale + adv_pts.detach() * (1 - grad_scale)

            adv_4 = torch.cat([adv_pts,
                               torch.ones(n_adv, 1, device=device)], dim=1)
            merged = torch.cat([ground_t, adv_4], dim=0)

            batch_dict = wrapper.build_batch_dict(merged)
            result = wrapper.forward_attack(batch_dict)

            verts_for_reg = v0 + mesh_param
            view_loss, ld = appearing_loss(
                rcnn_cls_preds=result.get('rcnn_cls_preds'),
                rcnn_features=result.get('rcnn_features'),
                ref_rcnn_feature=ref_rcnn_feature,
                rcnn_box_preds=result.get('rcnn_box_preds'),
                ref_orientation=ref_orientation,
                ref_box_size=ref_box_size,
                point_cls_logits=result.get('point_cls_logits'),
                point_features=result.get('point_features'),
                ref_backbone_feature=ref_backbone_feature,
                vertices=verts_for_reg, faces=faces,
                n_scene=n_ground, n_adv=n_adv,
                alpha_rpn=alpha_rpn, alpha_feat=alpha_feat,
                beta_box=beta_box,
                gamma_area=gamma_area,
                kappa_rcnn=kappa_rcnn, kappa_rpn=kappa_rpn,
            )

            view_losses.append(view_loss)
            view_loss_dicts.append(ld)
            valid_views += 1

        if valid_views > 0:
            total_loss = sum(view_losses) / valid_views
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([mesh_param], 1.0)
            optimizer.step()

            avg_ld = {}
            for k in view_loss_dicts[0]:
                avg_ld[k] = sum(d[k] for d in view_loss_dicts) / valid_views
        else:
            avg_ld = {k: 0.0 for k in history}

        scheduler.step()

        # Physical constraints: clamp mesh within size_limit
        with torch.no_grad():
            verts_shape = v0 + mesh_param
            centre = verts_shape.mean(dim=0, keepdim=True)
            verts_shape = verts_shape - centre
            for d in range(3):
                verts_shape[:, d].data.clamp_(-b[d].item(), b[d].item())
            mesh_param.data.copy_(verts_shape + centre - v0)

        for k in history:
            history[k].append(avg_ld.get(k, 0.0))

        pbar.set_postfix({
            'tot': f'{avg_ld.get("L_total", 0):.4f}',
            'rcnn': f'{avg_ld.get("L_rcnn_cls", 0):.3f}',
            'rpn': f'{avg_ld.get("L_rpn_cls", 0):.3f}',
            'box': f'{avg_ld.get("L_box", 0):.3f}',
            'v': valid_views,
        })

        if (step + 1) % monitor_every == 0:
            with torch.no_grad():
                lidar_pos = np.array([0.0, 0.0, 0.75], dtype=np.float32)
                rel_pos = _object_relative_pos(object_pos, lidar_pos)
                verts_check = v0 + mesh_param.detach()
                vw = verts_check + torch.tensor(rel_pos, dtype=torch.float32, device=device).unsqueeze(0)
                pts = render_adversarial_points(
                    vw, faces, sensor_origin,
                    n_elevation=lidar_cfg.get('n_elevation', 16),
                    elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                    elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                    h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                    margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
                )
                if pts.shape[0] > 0:
                    from attack.inject import inject_points
                    merged_np, _ = inject_points(ground_np.copy(),
                        pts.cpu().numpy(), np.zeros(3), remove_overlap=True)
                    pred_boxes, pred_scores = wrapper.detect(
                        torch.from_numpy(merged_np.astype(np.float32)).to(device), 0.3)
                    n_det = len(pred_scores)
                    best_s = float(pred_scores.max()) if n_det > 0 else 0.0
                    print(f"\n  [Step {step+1}] hits={pts.shape[0]}, "
                          f"det={n_det}, best={best_s:.3f}")

        if (step + 1) % 500 == 0:
            _save_checkpoint(mesh_param, v0, faces, history,
                             save_dir, tag='latest', b=b)

    _save_checkpoint(mesh_param, v0, faces, history,
                     save_dir, tag='final', b=b)
    if hasattr(wrapper, 'remove_hook'):
        wrapper.remove_hook()
    print(f"\nAttack complete. Results saved to {save_dir}/")
    return mesh_param.detach(), history


def apply_attack_to_sample(sample, mesh_param, translation_param, v0, faces,
                           config, device, injection_pos=None,
                           param_mode='mesh_offset', save_dir='results',
                           save_stage='apply'):
    """Apply optimized mesh to a sample for evaluation."""
    atk = config['attack']
    lidar_cfg = atk.get('lidar', {})
    sensor_pos = torch.zeros(3, device=device)

    delta_v_d = mesh_param.to(device)
    v0_d = v0.to(device)
    faces_d = faces.to(device)

    if injection_pos is None:
        injection_pos = np.array(atk.get('object_pos', [4.0, -2.0, 0.0]),
                                 dtype=np.float32)

    with torch.no_grad():
        vertices = v0_d + delta_v_d
        inj_t = torch.tensor(injection_pos, dtype=torch.float32, device=device)
        vertices_world = vertices + inj_t.unsqueeze(0)

        adv_pts = render_adversarial_points(
            vertices_world, faces_d, sensor_pos,
            n_elevation=lidar_cfg.get('n_elevation', 16),
            elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
            elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
            h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
            margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
        )

    adv_pts_np = adv_pts.cpu().numpy()
    from attack.inject import inject_points
    merged, n_adv = inject_points(
        sample['pointcloud'], adv_pts_np,
        np.zeros(3), remove_overlap=True,
    )
    return merged, n_adv
