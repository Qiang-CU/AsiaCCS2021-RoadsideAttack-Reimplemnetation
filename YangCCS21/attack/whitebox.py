"""
White-box appearing attack loop for PointRCNN.

Single-stage RPN-level optimization, inspired by Tu et al. (CVPR 2020).
All losses operate on per-point RPN outputs which have full gradient support:
  - point_cls_logits  (N,)    foreground score
  - rpn_box_preds     (N, 7)  predicted box [cx,cy,cz,l,w,h,yaw]
  - point_features    (N, 128) backbone features

No STE or RCNN-level loss is needed: if RPN outputs are strong enough
(high foreground logits + box aligned to injection position), proposals
naturally appear and RCNN classifies them with high confidence.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from attack.reparameterize import reparameterize
from attack.inject import (
    build_bev_occupancy, sample_injection_position,
    inject_points, get_injection_metadata,
)
from attack.loss import appearing_loss, apply_physical_constraints
from model.pointrcnn_wrapper import PointRCNNWrapper


def _infer_mesh_param_mode(atk=None, ckpt=None):
    """Resolve mesh parameterisation mode from config or checkpoint."""
    if ckpt is not None:
        mode = ckpt.get('param_mode')
        if mode:
            return mode
        if 'offset' in ckpt or 't' in ckpt:
            return 'mesh_offset'
        return 'reparameterize'

    atk = atk or {}
    mesh_cfg = atk.get('mesh', {})
    return mesh_cfg.get('param_mode', atk.get('mesh_param_mode', 'mesh_offset'))


def _build_mesh_vertices(v0, mesh_param, translation_param, param_mode, b, c,
                         device):
    """Map optimisable variables to local-frame mesh vertices."""
    if param_mode == 'reparameterize':
        return reparameterize(
            v0, mesh_param, translation_param, torch.eye(3, device=device), b, c
        )
    if param_mode == 'mesh_offset':
        return v0 + mesh_param + translation_param.unsqueeze(0)
    raise ValueError(f'Unknown mesh param_mode: {param_mode}')


def _apply_mesh_offset_constraints(offset, translation, v0, size_limit,
                                   translation_limit):
    """Explicitly clamp mesh size and local translation for offset+t mode."""
    for d in range(3):
        translation.data[d].clamp_(
            -translation_limit[d].item(), translation_limit[d].item()
        )

    verts_shape = v0 + offset
    centre = verts_shape.mean(dim=0, keepdim=True)
    verts_shape = verts_shape - centre
    for d in range(3):
        verts_shape[:, d].data.clamp_(
            -size_limit[d].item(), size_limit[d].item()
        )
    offset.data.copy_(verts_shape + centre - v0)


def _maybe_save_hit_points(adv_pts, sample, injection_pos, save_cfg, save_dir,
                           stage, step=None):
    """Optionally persist LiDAR-hit adversarial points for debugging."""
    if not save_cfg or not save_cfg.get('enabled', False):
        return

    when = save_cfg.get('when', ['apply'])
    if isinstance(when, str):
        when = [when]
    if 'all' not in when and stage not in when:
        return

    out_dir = os.path.join(save_dir, save_cfg.get('subdir', 'debug_mesh_hits'))
    os.makedirs(out_dir, exist_ok=True)

    max_files = save_cfg.get('max_files')
    if max_files is not None and max_files >= 0:
        n_existing = len([n for n in os.listdir(out_dir) if n.endswith('.pth')])
        if n_existing >= max_files:
            return

    name_parts = [stage, str(sample.get('sample_id', 'unknown'))]
    if step is not None:
        name_parts.append(f'step{int(step):04d}')
    out_path = os.path.join(out_dir, '_'.join(name_parts) + '.pth')

    payload = {
        'adv_points': adv_pts.detach().cpu(),
        'injection_pos': np.asarray(injection_pos, dtype=np.float32),
        'sample_id': sample.get('sample_id'),
        'stage': stage,
    }
    if step is not None:
        payload['step'] = int(step)
    torch.save(payload, out_path)


def load_mesh_checkpoint(ckpt_path, map_location='cpu'):
    """Load mesh checkpoint and normalize old/new parameterization fields."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    param_mode = _infer_mesh_param_mode(ckpt=ckpt)
    if param_mode == 'mesh_offset':
        mesh_param = ckpt['offset']
        translation_param = ckpt['t']
    else:
        mesh_param = ckpt['delta_v']
        translation_param = ckpt['t_tilde']

    return {
        'ckpt': ckpt,
        'param_mode': param_mode,
        'mesh_param': mesh_param,
        'translation_param': translation_param,
        'v0': ckpt['v0'],
        'faces': ckpt['faces'],
        'b': ckpt.get('b'),
        'c': ckpt.get('c'),
        'history': ckpt.get('history', {}),
        'method': ckpt.get('method', 'single_stage_whitebox'),
    }


def count_proposals_near(rois, injection_pos, proximity=3.0):
    """Count ROI proposals within proximity of injection position."""
    if rois is None:
        return 0
    rois_np = rois.detach().cpu().numpy().squeeze(0)  # (K, 7)
    if len(rois_np) == 0:
        return 0
    dists = np.sqrt((rois_np[:, 0] - injection_pos[0])**2 +
                    (rois_np[:, 1] - injection_pos[1])**2)
    return int((dists < proximity).sum())


def best_rcnn_confidence_near(pred_dicts, pos, radius=3.0):
    """Best RCNN post-NMS detection confidence near injection position."""
    if pred_dicts is None or len(pred_dicts) == 0:
        return 0.0
    pred = pred_dicts[0]
    boxes = pred['pred_boxes'].detach().cpu().numpy()
    scores = pred['pred_scores'].detach().cpu().numpy()
    if len(boxes) == 0:
        return 0.0
    dists = np.sqrt((boxes[:, 0] - pos[0])**2 + (boxes[:, 1] - pos[1])**2)
    near = dists < radius
    if not near.any():
        return 0.0
    return float(scores[near].max())


def precompute_injections(dataset, inj_cfg, rng=None):
    """Pre-compute injection positions for all frames."""
    if rng is None:
        rng = np.random.RandomState(42)
    cache = {}
    valid = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        occ, gi = build_bev_occupancy(
            sample['pointcloud'], sample['gt_bboxes'],
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )
        pos, is_valid = sample_injection_position(
            occ, gi,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
            rng=rng,
        )
        cache[idx] = {
            'pos': pos, 'valid': is_valid,
            'metadata': get_injection_metadata(
                sample['sample_id'], pos, is_valid, len(sample['gt_bboxes'])
            ),
        }
        if is_valid:
            valid.append(idx)
    if not valid:
        valid = list(range(len(dataset)))
    return cache, valid


def _run_one_frame(wrapper, dataset, frame_idx, injection_cache,
                   v0, mesh_param, translation_param, faces, b, c,
                   lidar_cfg, sensor_pos, device,
                   rpn_only=False, run_post=True,
                   param_mode='mesh_offset', hit_save_cfg=None,
                   save_dir='results', save_stage='train', save_step=None):
    """
    Forward pass for one frame: mesh params -> render -> inject -> PointRCNN.

    Args:
        rpn_only:  skip RCNN head (faster for training)
        run_post:  run post_processing for monitoring metrics

    Returns result dict + metadata, or None if frame produces no adversarial points.
    """
    sample = dataset[int(frame_idx)]
    inj_pos = injection_cache[int(frame_idx)]['pos']

    vertices = _build_mesh_vertices(
        v0, mesh_param, translation_param, param_mode, b, c, device
    )
    inj_t = torch.tensor(inj_pos, dtype=torch.float32, device=device)
    vertices_world = vertices + inj_t.unsqueeze(0)

    adv_pts = render_adversarial_points(
        vertices_world, faces, sensor_pos,
        n_elevation=lidar_cfg.get('n_elevation', 64),
        elev_min_deg=lidar_cfg.get('elev_min_deg', -24.9),
        elev_max_deg=lidar_cfg.get('elev_max_deg', 2.0),
        h_step_deg=lidar_cfg.get('h_step_deg', 0.08),
        margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
    )
    if adv_pts.shape[0] == 0:
        return None

    _maybe_save_hit_points(
        adv_pts, sample, inj_pos, hit_save_cfg, save_dir, save_stage,
        step=save_step,
    )

    pc_tensor = torch.tensor(
        sample['pointcloud'], dtype=torch.float32, device=device
    )
    merged_pc, n_adv = inject_points(
        pc_tensor, adv_pts, torch.zeros(3, device=device),
        remove_overlap=True,
    )
    n_scene = merged_pc.shape[0] - n_adv

    result = wrapper.forward_with_grad(merged_pc, rpn_only=rpn_only,
                                       run_post=run_post)
    return {
        'result': result,
        'n_scene': n_scene,
        'n_adv': n_adv,
        'inj_pos': inj_pos,
        'vertices': vertices,
    }


def run_whitebox_attack(dataset, config, save_dir='results',
                        warm_start_ckpt=None):
    """
    Single-stage white-box appearing attack.

    Optimizes mesh so that RPN per-point outputs (foreground logits, box
    predictions, backbone features) produce strong car-like proposals at
    the injection position.  Modeled after Tu et al. (CVPR 2020).

    Args:
        dataset:          KITTIDataset
        config:           dict from attack_config.yaml
        save_dir:         output directory
        warm_start_ckpt:  path to checkpoint for warm-start

    Returns:
        mesh_param, translation_param, history
    """
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    inj_cfg = atk['injection']
    lidar_cfg = atk.get('lidar', {})
    lw = atk['loss_weights']
    device_str = config.get('device', 'cuda:0')
    device = torch.device(device_str)

    n_iters = atk['n_iters']
    lr = atk['lr']
    grad_scale = atk.get('grad_scale', 10.0)
    batch_size = atk.get('multi_frame_batch', 8)
    target_car_size = tuple(atk.get('phantom_box_size', [4.0, 1.6, 1.5]))
    mesh_param_mode = _infer_mesh_param_mode(atk=atk)
    mesh_cfg = atk.get('mesh', {})
    hit_save_cfg = mesh_cfg.get('save_hit_points', atk.get('save_hit_points', {}))

    # ── Mesh setup ──
    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0, faces = v0.to(device), faces.to(device)
    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)
    c = torch.tensor(atk['translation_limit'], dtype=torch.float32, device=device)
    sensor_pos = torch.zeros(3, device=device)

    # ── Initialize parameters ──
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        loaded = load_mesh_checkpoint(warm_start_ckpt, map_location=device)
        if loaded['param_mode'] != mesh_param_mode:
            raise ValueError(
                f"Warm-start checkpoint mode {loaded['param_mode']} does not "
                f"match config mode {mesh_param_mode}"
            )
        mesh_param = loaded['mesh_param'].to(device).requires_grad_(True)
        translation_param = (
            loaded['translation_param'].to(device).requires_grad_(True)
        )
        print(f"Warm-started from {warm_start_ckpt}")
    else:
        mesh_param = torch.zeros_like(v0, requires_grad=True)
        translation_param = torch.zeros(3, device=device, requires_grad=True)

    # ── Load reference car feature ──
    ref_path = config.get('ref_feature', {}).get('output_path',
                                                  'results/ref_car_feature.pt')
    if os.path.exists(ref_path):
        ref_feature = torch.load(ref_path, map_location=device,
                                 weights_only=True).squeeze()
        print(f"  Loaded ref feature: {ref_feature.shape}")
    else:
        print(f"  WARNING: ref feature not found at {ref_path}, using zeros")
        ref_feature = torch.zeros(128, device=device)

    # ── Pre-compute injection positions ──
    print("Pre-computing injection positions...")
    injection_cache, valid_indices = precompute_injections(dataset, inj_cfg)
    print(f"  {len(valid_indices)}/{len(dataset)} valid positions")

    # Select training subset: prefer frames at 5-20m range
    train_indices = []
    for vi in valid_indices:
        pos = injection_cache[vi]['pos']
        d = np.sqrt(pos[0]**2 + pos[1]**2)
        if 5.0 <= d <= 20.0:
            train_indices.append(vi)
    if len(train_indices) < 32:
        train_indices = valid_indices[:min(100, len(valid_indices))]
    else:
        rng_sel = np.random.RandomState(42)
        rng_sel.shuffle(train_indices)
        train_indices = train_indices[:min(64, len(train_indices))]
    print(f"  Training subset: {len(train_indices)} frames")

    meta_list = [injection_cache[i]['metadata'] for i in range(len(dataset))
                 if i in injection_cache]
    with open(os.path.join(save_dir, 'injection_metadata.json'), 'w') as f:
        json.dump(meta_list, f, indent=2)

    # ── Model ──
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=device_str,
        enable_ste=False,
    )

    # ── Optimizer ──
    optimizer = torch.optim.Adam([mesh_param, translation_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

    history = {
        'L_total': [], 'L_cls': [], 'L_loc': [], 'L_size': [],
        'L_feat': [], 'L_lap': [],
        'n_proposals_near': [], 'rcnn_best_conf': [],
    }

    # ── Main optimization loop ──
    print(f"\n{'='*70}")
    print(f"  Single-stage RPN optimization ({n_iters} iters, lr={lr})")
    print(f"  Mesh param mode: {mesh_param_mode}")
    print(f"  Loss = L_cls + {lw['beta_loc']}*L_loc + {lw['beta_size']}*L_size"
          f" + {lw['alpha_feat']}*L_feat + {lw['lambda_lap']}*L_lap")
    print(f"{'='*70}")

    effective_batch = min(batch_size, len(train_indices))
    monitor_every = max(n_iters // 20, 50)

    pbar = tqdm(range(n_iters), desc='Whitebox (RPN)')
    for step in pbar:
        optimizer.zero_grad()

        batch_idx = np.random.choice(
            train_indices, size=min(effective_batch, len(train_indices)),
            replace=False,
        )

        step_loss = {k: 0.0 for k in history}
        n_valid = 0

        for fi in batch_idx:
            frame = _run_one_frame(
                wrapper, dataset, fi, injection_cache,
                v0, mesh_param, translation_param, faces, b, c,
                lidar_cfg, sensor_pos, device,
                rpn_only=True, run_post=False,
                param_mode=mesh_param_mode,
                hit_save_cfg=hit_save_cfg,
                save_dir=save_dir,
                save_stage='train',
                save_step=step + 1,
            )
            if frame is None:
                continue

            r = frame['result']
            loss, ld = appearing_loss(
                point_cls_logits=r['point_cls_logits'],
                rpn_box_preds=r['rpn_box_preds'],
                point_features=r['point_features'],
                ref_feature=ref_feature,
                vertices=frame['vertices'], faces=faces, adj=adj,
                n_scene=frame['n_scene'], n_adv=frame['n_adv'],
                injection_pos=frame['inj_pos'], device=device,
                kappa=lw.get('kappa', 5.0),
                alpha_feat=lw['alpha_feat'],
                beta_loc=lw['beta_loc'],
                beta_size=lw['beta_size'],
                lambda_lap=lw['lambda_lap'],
                target_size=target_car_size,
            )

            (loss / len(batch_idx)).backward()

            for k, v in ld.items():
                if k in step_loss:
                    step_loss[k] += v
            n_valid += 1

        if n_valid > 0:
            for k in step_loss:
                step_loss[k] /= n_valid

            # Amplify gradients through the renderer → backbone chain
            if grad_scale != 1.0:
                if mesh_param.grad is not None:
                    mesh_param.grad.data.mul_(grad_scale)
                if translation_param.grad is not None:
                    translation_param.grad.data.mul_(grad_scale)

            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            if mesh_param_mode == 'reparameterize':
                apply_physical_constraints(mesh_param, v0, tuple(b.tolist()))
            else:
                _apply_mesh_offset_constraints(
                    mesh_param, translation_param, v0, b, c
                )

        for k in history:
            history[k].append(step_loss.get(k, 0.0))

        pbar.set_postfix({
            'loss': f'{step_loss.get("L_total", 0):.4f}',
            'cls': f'{step_loss.get("L_cls", 0):.4f}',
            'loc': f'{step_loss.get("L_loc", 0):.4f}',
        })

        # Periodic monitoring: check proposals + RCNN confidence
        if (step + 1) % monitor_every == 0:
            with torch.no_grad():
                check_idx = np.random.choice(
                    train_indices, size=min(4, len(train_indices)),
                    replace=False,
                )
                total_prop, total_conf, n_checked = 0, 0.0, 0
                for ci in check_idx:
                    cf = _run_one_frame(
                        wrapper, dataset, ci, injection_cache,
                        v0, mesh_param.detach(), translation_param.detach(),
                        faces, b, c,
                        lidar_cfg, sensor_pos, device,
                        rpn_only=False, run_post=True,
                        param_mode=mesh_param_mode,
                        hit_save_cfg=hit_save_cfg,
                        save_dir=save_dir,
                        save_stage='monitor',
                        save_step=step + 1,
                    )
                    if cf is None:
                        continue
                    cr = cf['result']
                    total_prop += count_proposals_near(
                        cr.get('rois'), cf['inj_pos'], 5.0)
                    total_conf += best_rcnn_confidence_near(
                        cr.get('pred_dicts'), cf['inj_pos'], 5.0)
                    n_checked += 1
                avg_prop = total_prop / max(n_checked, 1)
                avg_conf = total_conf / max(n_checked, 1)
            history['n_proposals_near'][-1] = avg_prop
            history['rcnn_best_conf'][-1] = avg_conf
            print(f"\n  [Step {step+1}] proposals_near={avg_prop:.1f}, "
                  f"rcnn_conf={avg_conf:.3f}")

        if (step + 1) % 200 == 0:
            _save_checkpoint(
                mesh_param, translation_param, v0, faces, history,
                save_dir, tag='latest', param_mode=mesh_param_mode, b=b, c=c,
            )

    _save_checkpoint(
        mesh_param, translation_param, v0, faces, history,
        save_dir, tag='final', param_mode=mesh_param_mode, b=b, c=c,
    )

    wrapper.remove_hook()
    print(f"\nAttack complete. Results saved to {save_dir}/")
    return mesh_param.detach(), translation_param.detach(), history


def apply_attack_to_sample(sample, mesh_param, translation_param, v0, faces,
                           config, device, injection_pos=None,
                           param_mode='reparameterize', save_dir='results',
                           save_stage='apply'):
    """
    Apply the optimized adversarial mesh to a single sample for evaluation.
    """
    atk = config['attack']
    inj_cfg = atk['injection']

    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)
    c = torch.tensor(atk['translation_limit'], dtype=torch.float32, device=device)
    sensor_pos = torch.zeros(3, device=device)
    lidar_cfg = atk.get('lidar', {})
    mesh_cfg = atk.get('mesh', {})
    hit_save_cfg = mesh_cfg.get('save_hit_points', atk.get('save_hit_points', {}))

    delta_v_d = mesh_param.to(device)
    t_tilde_d = translation_param.to(device)
    v0_d = v0.to(device)
    faces_d = faces.to(device)

    if injection_pos is None:
        occ, gi = build_bev_occupancy(
            sample['pointcloud'], sample['gt_bboxes'],
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )
        injection_pos, _ = sample_injection_position(
            occ, gi,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
        )

    with torch.no_grad():
        vertices = _build_mesh_vertices(
            v0_d, delta_v_d, t_tilde_d, param_mode, b, c, device
        )
        inj_t = torch.tensor(injection_pos, dtype=torch.float32, device=device)
        vertices_world = vertices + inj_t.unsqueeze(0)

        adv_pts = render_adversarial_points(
            vertices_world, faces_d, sensor_pos,
            n_elevation=lidar_cfg.get('n_elevation', 64),
            elev_min_deg=lidar_cfg.get('elev_min_deg', -24.9),
            elev_max_deg=lidar_cfg.get('elev_max_deg', 2.0),
            h_step_deg=lidar_cfg.get('h_step_deg', 0.08),
            margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
        )

    mesh_cfg = atk.get('mesh', {})
    hit_save_cfg = mesh_cfg.get('save_hit_points', atk.get('save_hit_points', {}))
    _maybe_save_hit_points(
        adv_pts, sample, injection_pos, hit_save_cfg, save_dir, save_stage
    )

    adv_pts_np = adv_pts.cpu().numpy()
    merged, n_adv = inject_points(
        sample['pointcloud'], adv_pts_np,
        np.zeros(3),
        remove_overlap=True,
    )
    return merged, n_adv


def _save_checkpoint(mesh_param, translation_param, v0, faces, history, save_dir,
                     tag='latest', param_mode='mesh_offset', b=None, c=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'adv_mesh_whitebox_{tag}.pth')
    data = {
        'v0': v0.detach().cpu(),
        'faces': faces.detach().cpu(),
        'history': history,
        'method': 'single_stage_whitebox',
        'param_mode': param_mode,
    }
    if param_mode == 'mesh_offset':
        data['offset'] = mesh_param.detach().cpu()
        data['t'] = translation_param.detach().cpu()
    else:
        data['delta_v'] = mesh_param.detach().cpu()
        data['t_tilde'] = translation_param.detach().cpu()
    if b is not None:
        data['b'] = b.detach().cpu()
    if c is not None:
        data['c'] = c.detach().cpu()
    torch.save(data, path)
    return path
