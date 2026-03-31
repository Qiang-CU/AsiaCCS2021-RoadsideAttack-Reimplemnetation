"""
Direct point-cloud optimisation attack for PointRCNN (appearing).

Instead of mesh → differentiable-render → points, we directly optimise the
(x, y, z) coordinates of N adversarial points.  This removes two major
sources of gradient noise (mesh reparameterisation and ray–triangle
intersection) while keeping the same RPN-level loss terms.

Gradient path:
    L  →  cls_layers / box_layers  →  PointNet2 backbone  →  input points
       →  inject_points (cat, differentiable)  →  adv_points  (optimised)

Usage:
    from attack.whitebox_pointopt import run_pointopt_attack
    adv_pts, history = run_pointopt_attack(dataset, config)
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from attack.inject import (
    build_bev_occupancy, sample_injection_position,
    inject_points, get_injection_metadata,
)
from attack.loss import L_cls, L_feat, L_rpn_box_loc, L_rpn_box_size
from attack.physical_constraints import (
    chamfer_distance, knn_smoothing_loss,
    estimate_normals_knn, normal_projection_loss,
)
from model.pointrcnn_wrapper import PointRCNNWrapper


# ---------------------------------------------------------------------------
# Point-cloud initialisers
# ---------------------------------------------------------------------------

def init_car_surface_points(n_points=400, size=(3.9, 1.6, 1.56),
                            device='cpu'):
    """
    Initialise points on the surface of a car-sized box.

    Points are uniformly sampled on the six faces of a box centred at
    the origin with half-extents ``size / 2``.  This gives the detector
    a realistic LiDAR-scan-like geometry to start from.

    Args:
        n_points: total number of points to generate
        size:     (length, width, height) in metres
        device:   torch device

    Returns:
        pts: (n_points, 3) tensor
    """
    l, w, h = [s / 2.0 for s in size]
    faces_area = [
        l * h,  # front / back  (x = ±l)
        w * h,  # left / right  (y = ±w)
        l * w,  # top / bottom  (z = ±h)
    ]
    total = 2.0 * sum(faces_area)
    n_per_face = [max(1, int(n_points * 2 * a / total)) for a in faces_area]
    # Double because two faces per pair
    pts = []

    def _sample_face(fixed_axis, fixed_val, ext1, ext2, n):
        u = torch.rand(n, device=device) * 2 * ext1 - ext1
        v = torch.rand(n, device=device) * 2 * ext2 - ext2
        p = torch.zeros(n, 3, device=device)
        axes = [0, 1, 2]
        axes.remove(fixed_axis)
        p[:, fixed_axis] = fixed_val
        p[:, axes[0]] = u
        p[:, axes[1]] = v
        return p

    # x = ±l  (front / back)
    pts.append(_sample_face(0, l, w, h, n_per_face[0]))
    pts.append(_sample_face(0, -l, w, h, n_per_face[0]))
    # y = ±w  (left / right)
    pts.append(_sample_face(1, w, l, h, n_per_face[1]))
    pts.append(_sample_face(1, -w, l, h, n_per_face[1]))
    # z = ±h  (top / bottom)
    pts.append(_sample_face(2, h, l, w, n_per_face[2]))
    pts.append(_sample_face(2, -h, l, w, n_per_face[2]))

    pts = torch.cat(pts, dim=0)
    # Trim or pad to exact n_points
    if pts.shape[0] > n_points:
        idx = torch.randperm(pts.shape[0], device=device)[:n_points]
        pts = pts[idx]
    elif pts.shape[0] < n_points:
        extra = n_points - pts.shape[0]
        idx = torch.randint(0, pts.shape[0], (extra,), device=device)
        pts = torch.cat([pts, pts[idx]], dim=0)

    return pts


def init_from_gt_cars(dataset, n_points=400, n_instances=50, device='cpu'):
    """
    Initialise adversarial points from real GT car LiDAR scans.

    Collects points inside GT car boxes, centres them, and computes an
    average point distribution.

    Returns:
        pts: (n_points, 3) tensor in local frame (centred at origin)
    """
    all_pts = []
    count = 0
    for idx in range(min(len(dataset), 500)):
        if count >= n_instances:
            break
        sample = dataset[idx]
        pc = sample['pointcloud']  # (N, 4)
        gt_boxes = sample['gt_bboxes']
        for box in gt_boxes:
            cx, cy, cz, dl, dw, dh, yaw = box[:7]
            dx = pc[:, 0] - cx
            dy = pc[:, 1] - cy
            dz = pc[:, 2] - cz
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            lx = dx * cos_y - dy * sin_y
            ly = dx * sin_y + dy * cos_y
            inside = (np.abs(lx) < dl / 2) & (np.abs(ly) < dw / 2) & (np.abs(dz) < dh / 2)
            if inside.sum() < 10:
                continue
            car_pts = np.stack([lx[inside], ly[inside], dz[inside]], axis=1)
            all_pts.append(car_pts)
            count += 1
            if count >= n_instances:
                break

    if not all_pts:
        print("  No GT car points found, falling back to box surface init")
        return init_car_surface_points(n_points, device=device)

    all_pts = np.concatenate(all_pts, axis=0)
    # Sub-sample to n_points using FPS-like random selection
    if len(all_pts) > n_points:
        idx = np.random.choice(len(all_pts), n_points, replace=False)
        all_pts = all_pts[idx]
    elif len(all_pts) < n_points:
        idx = np.random.choice(len(all_pts), n_points, replace=True)
        all_pts = all_pts[idx]

    pts = torch.tensor(all_pts, dtype=torch.float32, device=device)
    # Centre
    pts = pts - pts.mean(dim=0, keepdim=True)
    return pts


# ---------------------------------------------------------------------------
# Regularisation losses for point clouds (replace Laplacian mesh loss)
# ---------------------------------------------------------------------------

def uniformity_loss(pts):
    """
    Encourage even spacing among adversarial points.

    Uses the mean of reciprocal nearest-neighbour distances to penalise
    clusters.  Much cheaper than full Chamfer and sufficient for
    regularisation.

    Args:
        pts: (N, 3) tensor

    Returns:
        loss: scalar tensor
    """
    if pts.shape[0] < 2:
        return pts.new_tensor(0.0)

    # Pairwise squared distances  (N, N)
    diff = pts.unsqueeze(0) - pts.unsqueeze(1)  # (N, N, 3)
    d2 = (diff * diff).sum(dim=-1)              # (N, N)
    # Mask self-distance
    eye = torch.eye(pts.shape[0], device=pts.device, dtype=torch.bool)
    d2 = d2 + eye.float() * 1e6
    nn_d2 = d2.min(dim=1).values               # (N,)
    # Penalise points that are too close (reciprocal)
    loss = (1.0 / (nn_d2 + 1e-4)).mean()
    return loss


def bbox_projection(pts, half_extents):
    """
    Project points back into a bounding box (in-place on .data).

    Args:
        pts:          (N, 3) tensor (modified in-place)
        half_extents: (3,) tensor [hx, hy, hz]
    """
    for d in range(3):
        pts.data[:, d].clamp_(-half_extents[d].item(), half_extents[d].item())


# ---------------------------------------------------------------------------
# Precompute injection positions (shared with mesh approach)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-frame forward pass
# ---------------------------------------------------------------------------

def _forward_one_frame(wrapper, dataset, frame_idx, injection_cache,
                       adv_points, device, rpn_only=True, run_post=False,
                       noise_sigma=0.0):
    """
    Forward pass: inject adv_points into scene → PointRCNN → RPN outputs.

    Args:
        wrapper:         PointRCNNWrapper
        dataset:         KITTIDataset
        frame_idx:       int
        injection_cache: dict from precompute_injections
        adv_points:      (N_adv, 3) tensor with requires_grad
        device:          torch device
        rpn_only:        skip RCNN head
        run_post:        run post_processing
        noise_sigma:     point-level EoT noise std (meters), 0 = disabled

    Returns:
        dict with 'result', 'n_scene', 'n_adv', 'inj_pos', or None
    """
    sample = dataset[int(frame_idx)]
    inj_pos = injection_cache[int(frame_idx)]['pos']

    pc_tensor = torch.tensor(
        sample['pointcloud'], dtype=torch.float32, device=device
    )

    pts_for_inject = adv_points
    if noise_sigma > 0 and adv_points.requires_grad:
        noise = torch.randn_like(adv_points) * noise_sigma
        pts_for_inject = adv_points + noise

    merged_pc, n_adv = inject_points(
        pc_tensor, pts_for_inject, inj_pos, remove_overlap=True,
    )

    if n_adv == 0:
        return None

    n_scene = merged_pc.shape[0] - n_adv

    result = wrapper.forward_with_grad(merged_pc, rpn_only=rpn_only,
                                       run_post=run_post)
    return {
        'result': result,
        'n_scene': n_scene,
        'n_adv': n_adv,
        'inj_pos': inj_pos,
    }


# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------

def _count_proposals_near(rois, injection_pos, proximity=3.0):
    if rois is None:
        return 0
    rois_np = rois.detach().cpu().numpy().squeeze(0)
    if len(rois_np) == 0:
        return 0
    dists = np.sqrt((rois_np[:, 0] - injection_pos[0])**2 +
                    (rois_np[:, 1] - injection_pos[1])**2)
    return int((dists < proximity).sum())


def _best_rcnn_conf(pred_dicts, pos, radius=3.0):
    if pred_dicts is None or len(pred_dicts) == 0:
        return 0.0
    pred = pred_dicts[0]
    if pred['pred_boxes'].shape[0] == 0:
        return 0.0
    boxes = pred['pred_boxes'].detach().cpu().numpy()
    scores = pred['pred_scores'].detach().cpu().numpy()
    dists = np.sqrt((boxes[:, 0] - pos[0])**2 + (boxes[:, 1] - pos[1])**2)
    near = dists < radius
    if not near.any():
        return 0.0
    return float(scores[near].max())


# ---------------------------------------------------------------------------
# Point-opt loss (reuses existing loss components, no mesh terms)
# ---------------------------------------------------------------------------

def pointopt_loss(point_cls_logits, rpn_box_preds, point_features,
                  ref_feature, adv_points,
                  n_scene, n_adv, injection_pos, device,
                  kappa=5.0, alpha_feat=0.1, beta_loc=0.5,
                  beta_size=0.2, lambda_uni=0.001,
                  target_size=(3.9, 1.6, 1.56),
                  pts_init=None, normals_init=None,
                  lambda_cd=0.0, lambda_knn=0.0, lambda_nproj=0.0,
                  knn_k=10):
    """
    Direct point optimisation loss.

    Same RPN-level terms as appearing_loss, but replaces Laplacian mesh
    smoothing with a point uniformity regulariser.  When physical
    constraint weights are > 0, additionally computes:
      - L_cd:    Chamfer distance to initial points
      - L_knn:   kNN smoothing (uniform spacing)
      - L_nproj: normal projection (prevent inward movement)

    Returns:
        total_loss, loss_dict
    """
    l_cls = L_cls(point_cls_logits, n_scene, n_adv, kappa=kappa)
    l_loc = L_rpn_box_loc(rpn_box_preds, n_scene, n_adv, injection_pos, device)
    l_size = L_rpn_box_size(rpn_box_preds, n_scene, n_adv,
                            target_size=target_size, device=device)
    l_feat = L_feat(point_features, ref_feature, n_scene, n_adv)
    l_uni = uniformity_loss(adv_points)

    total = (l_cls
             + beta_loc * l_loc
             + beta_size * l_size
             + alpha_feat * l_feat
             + lambda_uni * l_uni)

    loss_dict = {
        'L_total': 0.0,
        'L_cls': l_cls.item(),
        'L_loc': l_loc.item(),
        'L_size': l_size.item(),
        'L_feat': l_feat.item(),
        'L_uni': l_uni.item(),
    }

    if lambda_cd > 0 and pts_init is not None:
        l_cd = chamfer_distance(adv_points, pts_init)
        total = total + lambda_cd * l_cd
        loss_dict['L_cd'] = l_cd.item()

    if lambda_knn > 0:
        l_knn = knn_smoothing_loss(adv_points, k=knn_k)
        total = total + lambda_knn * l_knn
        loss_dict['L_knn'] = l_knn.item()

    if lambda_nproj > 0 and pts_init is not None and normals_init is not None:
        l_nproj = normal_projection_loss(adv_points, pts_init, normals_init)
        total = total + lambda_nproj * l_nproj
        loss_dict['L_nproj'] = l_nproj.item()

    loss_dict['L_total'] = total.item()
    return total, loss_dict


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(adv_points, history, save_dir, tag='latest',
                     pts_init=None, method='pointopt'):
    path = os.path.join(save_dir, f'whitebox_pointopt_{tag}.pth')
    data = {
        'adv_points': adv_points.detach().cpu(),
        'history': history,
        'method': method,
    }
    if pts_init is not None:
        data['pts_init'] = pts_init.detach().cpu()
    torch.save(data, path)


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def _process_frames_on_gpu(wrapper, dataset, frame_indices, injection_cache,
                           adv_points_main, gpu_device, ref_feature,
                           lw, lambda_uni, target_car_size, total_batch_size,
                           pts_init=None, normals_init=None,
                           phys_cfg=None):
    """Run forward + backward for a subset of frames on one GPU.

    Returns (grad, loss_dicts, n_valid).  ``grad`` is on ``gpu_device``.
    """
    torch.cuda.set_device(gpu_device)

    adv_local = adv_points_main.detach().to(gpu_device).requires_grad_(True)
    ref_local = ref_feature.to(gpu_device)
    pts_init_local = pts_init.to(gpu_device) if pts_init is not None else None
    normals_local = normals_init.to(gpu_device) if normals_init is not None else None

    pc = phys_cfg or {}
    lambda_cd = pc.get('lambda_cd', 0.0)
    lambda_knn = pc.get('lambda_knn', 0.0)
    lambda_nproj = pc.get('lambda_nproj', 0.0)
    knn_k = pc.get('knn_k', 10)
    noise_sigma = pc.get('noise_sigma', 0.0)

    loss_dicts = []
    n_valid = 0

    for fi in frame_indices:
        frame = _forward_one_frame(
            wrapper, dataset, fi, injection_cache,
            adv_local, gpu_device,
            rpn_only=True, run_post=False,
            noise_sigma=noise_sigma,
        )
        if frame is None:
            continue

        r = frame['result']
        loss, ld = pointopt_loss(
            point_cls_logits=r['point_cls_logits'],
            rpn_box_preds=r['rpn_box_preds'],
            point_features=r['point_features'],
            ref_feature=ref_local,
            adv_points=adv_local,
            n_scene=frame['n_scene'], n_adv=frame['n_adv'],
            injection_pos=frame['inj_pos'], device=gpu_device,
            kappa=lw.get('kappa', 5.0),
            alpha_feat=lw['alpha_feat'],
            beta_loc=lw['beta_loc'],
            beta_size=lw['beta_size'],
            lambda_uni=lambda_uni,
            target_size=target_car_size,
            pts_init=pts_init_local,
            normals_init=normals_local,
            lambda_cd=lambda_cd,
            lambda_knn=lambda_knn,
            lambda_nproj=lambda_nproj,
            knn_k=knn_k,
        )
        (loss / total_batch_size).backward()
        loss_dicts.append(ld)
        n_valid += 1

    if adv_local.grad is not None:
        grad = adv_local.grad.clone()
    else:
        grad = torch.zeros_like(adv_local)
    return grad, loss_dicts, n_valid


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def run_pointopt_attack(dataset, config, save_dir='results',
                        warm_start_ckpt=None, devices=None, wrappers=None):
    """
    Direct point-cloud optimisation attack (appearing).

    Optimises the (x, y, z) coordinates of N adversarial points so that
    RPN per-point outputs produce strong car-like proposals at the
    injection position.

    Args:
        dataset:          KITTIDataset
        config:           dict from attack_config.yaml
        save_dir:         output directory
        warm_start_ckpt:  optional checkpoint path
        devices:          list of torch.device for multi-GPU (None → single GPU)
        wrappers:         dict {device: PointRCNNWrapper} (None → auto-create)

    Returns:
        adv_points (detached), history
    """
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    inj_cfg = atk['injection']
    device_str = config.get('device', 'cuda:0')
    device = torch.device(device_str)

    # Multi-GPU setup
    if devices is None:
        devices = [device]
    n_gpus = len(devices)
    multi_gpu = n_gpus > 1

    # Point-opt specific config (with sensible defaults)
    po_cfg = atk.get('pointopt', {})
    n_points = po_cfg.get('n_points', 400)
    n_iters = po_cfg.get('n_iters', atk.get('n_iters', 1000))
    lr = po_cfg.get('lr', atk.get('lr', 0.02))
    batch_size = po_cfg.get('multi_frame_batch', atk.get('multi_frame_batch', 8))
    init_mode = po_cfg.get('init', 'gt')  # 'gt' or 'box'
    target_car_size = tuple(atk.get('phantom_box_size', [3.9, 1.6, 1.56]))

    lw = atk['loss_weights']
    lambda_uni = po_cfg.get('lambda_uni', 0.001)

    # Physical constraints config
    phys_cfg = po_cfg.get('physical', {})
    phys_enabled = phys_cfg.get('enabled', False)
    lambda_cd = phys_cfg.get('lambda_cd', 0.0) if phys_enabled else 0.0
    lambda_knn = phys_cfg.get('lambda_knn', 0.0) if phys_enabled else 0.0
    lambda_nproj = phys_cfg.get('lambda_nproj', 0.0) if phys_enabled else 0.0
    knn_k = phys_cfg.get('knn_k', 10)
    noise_sigma = phys_cfg.get('noise_sigma', 0.0) if phys_enabled else 0.0

    # Half-extents for bounding box projection
    half_ext = torch.tensor(
        [s / 2.0 for s in target_car_size], dtype=torch.float32, device=device
    )

    # ── Initialise adversarial points ──
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        adv_points = ckpt['adv_points'].to(device).requires_grad_(True)
        print(f"Warm-started from {warm_start_ckpt}: {adv_points.shape}")
    else:
        if init_mode == 'gt':
            print(f"Initialising {n_points} points from GT car scans...")
            adv_points = init_from_gt_cars(
                dataset, n_points=n_points, device=device
            )
        else:
            print(f"Initialising {n_points} points on car-surface box...")
            adv_points = init_car_surface_points(
                n_points, size=target_car_size, device=device
            )
        adv_points = adv_points.requires_grad_(True)

    # Store initial points for physical constraints (Chamfer + normal proj)
    pts_init = adv_points.detach().clone()
    normals_init = None
    if phys_enabled:
        normals_init = estimate_normals_knn(pts_init, k=knn_k)
        print(f"  Physical constraints: CD={lambda_cd}, kNN={lambda_knn}, "
              f"NProj={lambda_nproj}, k={knn_k}, noise_σ={noise_sigma}")

    print(f"  adv_points shape: {adv_points.shape}")

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

    # Select training subset: prefer frames at 5–20 m range
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

    # ── Model(s) ──
    if wrappers is not None:
        gpu_wrappers = wrappers
    elif multi_gpu:
        gpu_wrappers = {}
        for dev in devices:
            torch.cuda.set_device(dev)
            gpu_wrappers[dev] = PointRCNNWrapper(
                config['model']['pointrcnn_config'],
                config['model']['pointrcnn_ckpt'],
                device=str(dev),
                enable_ste=False,
            )
        print(f"  Model replicas on {n_gpus} GPUs: "
              f"{[str(d) for d in devices]}")
    else:
        single_wrapper = PointRCNNWrapper(
            config['model']['pointrcnn_config'],
            config['model']['pointrcnn_ckpt'],
            device=device_str,
            enable_ste=False,
        )
        gpu_wrappers = {device: single_wrapper}

    # ── Optimiser ──
    optimizer = torch.optim.Adam([adv_points], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

    history = {
        'L_total': [], 'L_cls': [], 'L_loc': [], 'L_size': [],
        'L_feat': [], 'L_uni': [],
        'n_proposals_near': [], 'rcnn_best_conf': [],
    }
    if phys_enabled:
        for k in ('L_cd', 'L_knn', 'L_nproj'):
            history[k] = []

    # ── Main loop ──
    print(f"\n{'='*70}")
    print(f"  Direct Point Optimisation ({n_iters} iters, lr={lr}, "
          f"{n_points} points, {n_gpus} GPU(s))")
    loss_str = (f"  Loss = L_cls + {lw['beta_loc']}*L_loc + {lw['beta_size']}*L_size"
                f" + {lw['alpha_feat']}*L_feat + {lambda_uni}*L_uni")
    if phys_enabled:
        loss_str += (f"\n         + {lambda_cd}*L_cd + {lambda_knn}*L_knn"
                     f" + {lambda_nproj}*L_nproj")
    print(loss_str)
    print(f"{'='*70}")

    effective_batch = min(batch_size, len(train_indices))
    monitor_every = max(n_iters // 20, 50)

    pbar = tqdm(range(n_iters), desc='PointOpt')
    for step in pbar:
        optimizer.zero_grad()

        batch_idx = np.random.choice(
            train_indices, size=min(effective_batch, len(train_indices)),
            replace=False,
        )

        step_loss = {k: 0.0 for k in history}
        n_valid = 0
        total_batch = len(batch_idx)

        if multi_gpu:
            # Split batch across GPUs
            chunks = [[] for _ in range(n_gpus)]
            for i, fi in enumerate(batch_idx):
                chunks[i % n_gpus].append(fi)

            dev_list = list(devices)
            phys_kw = {
                'lambda_cd': lambda_cd, 'lambda_knn': lambda_knn,
                'lambda_nproj': lambda_nproj, 'knn_k': knn_k,
                'noise_sigma': noise_sigma,
            } if phys_enabled else None
            with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                futures = []
                for gi, dev in enumerate(dev_list):
                    if not chunks[gi]:
                        continue
                    futures.append(pool.submit(
                        _process_frames_on_gpu,
                        gpu_wrappers[dev], dataset, chunks[gi],
                        injection_cache, adv_points, dev, ref_feature,
                        lw, lambda_uni, target_car_size, total_batch,
                        pts_init, normals_init, phys_kw,
                    ))

                for fut in futures:
                    grad, lds, nv = fut.result()
                    # Accumulate gradient on primary device
                    if adv_points.grad is None:
                        adv_points.grad = grad.to(device)
                    else:
                        adv_points.grad += grad.to(device)
                    for ld in lds:
                        for k, v in ld.items():
                            if k in step_loss:
                                step_loss[k] += v
                    n_valid += nv
        else:
            wrapper = gpu_wrappers[device]
            for fi in batch_idx:
                frame = _forward_one_frame(
                    wrapper, dataset, fi, injection_cache,
                    adv_points, device,
                    rpn_only=True, run_post=False,
                    noise_sigma=noise_sigma,
                )
                if frame is None:
                    continue

                r = frame['result']
                loss, ld = pointopt_loss(
                    point_cls_logits=r['point_cls_logits'],
                    rpn_box_preds=r['rpn_box_preds'],
                    point_features=r['point_features'],
                    ref_feature=ref_feature,
                    adv_points=adv_points,
                    n_scene=frame['n_scene'], n_adv=frame['n_adv'],
                    injection_pos=frame['inj_pos'], device=device,
                    kappa=lw.get('kappa', 5.0),
                    alpha_feat=lw['alpha_feat'],
                    beta_loc=lw['beta_loc'],
                    beta_size=lw['beta_size'],
                    lambda_uni=lambda_uni,
                    target_size=target_car_size,
                    pts_init=pts_init,
                    normals_init=normals_init,
                    lambda_cd=lambda_cd,
                    lambda_knn=lambda_knn,
                    lambda_nproj=lambda_nproj,
                    knn_k=knn_k,
                )

                (loss / total_batch).backward()

                for k, v in ld.items():
                    if k in step_loss:
                        step_loss[k] += v
                n_valid += 1

        if n_valid > 0:
            for k in step_loss:
                step_loss[k] /= n_valid
            optimizer.step()
            scheduler.step()

        # Project back into bounding box
        with torch.no_grad():
            bbox_projection(adv_points, half_ext)

        for k in history:
            history[k].append(step_loss.get(k, 0.0))

        pbar.set_postfix({
            'loss': f'{step_loss.get("L_total", 0):.4f}',
            'cls': f'{step_loss.get("L_cls", 0):.4f}',
            'loc': f'{step_loss.get("L_loc", 0):.4f}',
        })

        # Periodic monitoring: check proposals + RCNN confidence
        if (step + 1) % monitor_every == 0:
            mon_wrapper = gpu_wrappers[devices[0]]
            with torch.no_grad():
                check_idx = np.random.choice(
                    train_indices, size=min(4, len(train_indices)),
                    replace=False,
                )
                total_prop, total_conf, n_checked = 0, 0.0, 0
                for ci in check_idx:
                    cf = _forward_one_frame(
                        mon_wrapper, dataset, ci, injection_cache,
                        adv_points.detach(), devices[0],
                        rpn_only=False, run_post=True,
                    )
                    if cf is None:
                        continue
                    cr = cf['result']
                    total_prop += _count_proposals_near(
                        cr.get('rois'), cf['inj_pos'], 5.0)
                    total_conf += _best_rcnn_conf(
                        cr.get('pred_dicts'), cf['inj_pos'], 5.0)
                    n_checked += 1
                avg_prop = total_prop / max(n_checked, 1)
                avg_conf = total_conf / max(n_checked, 1)
            history['n_proposals_near'][-1] = avg_prop
            history['rcnn_best_conf'][-1] = avg_conf
            print(f"\n  [Step {step+1}] proposals_near={avg_prop:.1f}, "
                  f"rcnn_conf={avg_conf:.3f}")

        if (step + 1) % 200 == 0:
            method_tag = 'pointopt_physical' if phys_enabled else 'pointopt'
            _save_checkpoint(adv_points, history, save_dir, tag='latest',
                             pts_init=pts_init if phys_enabled else None,
                             method=method_tag)

    method_tag = 'pointopt_physical' if phys_enabled else 'pointopt'
    _save_checkpoint(adv_points, history, save_dir, tag='final',
                     pts_init=pts_init if phys_enabled else None,
                     method=method_tag)

    for w in gpu_wrappers.values():
        w.remove_hook()
    print(f"\nPointOpt attack complete. Results saved to {save_dir}/")
    return adv_points.detach(), history


# ---------------------------------------------------------------------------
# Apply optimised points to a sample (for evaluation)
# ---------------------------------------------------------------------------

def apply_pointopt_to_sample(sample, adv_points_ckpt, config,
                             device='cuda:0', injection_pos=None):
    """
    Apply optimised adversarial points to a single sample for evaluation.

    Compatible with the existing evaluation pipeline (metrics.py).

    Args:
        sample:           dict from KITTIDataset
        adv_points_ckpt:  path to checkpoint or (N, 3) tensor
        config:           attack config dict
        device:           torch device string
        injection_pos:    (3,) override injection position

    Returns:
        merged_pc: (N+M, 4) numpy array
        n_adv:     int
    """
    inj_cfg = config['attack']['injection']

    if isinstance(adv_points_ckpt, str):
        ckpt = torch.load(adv_points_ckpt, map_location='cpu', weights_only=False)
        adv_pts = ckpt['adv_points']
    else:
        adv_pts = adv_points_ckpt

    adv_pts_np = adv_pts.numpy() if isinstance(adv_pts, torch.Tensor) else adv_pts

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

    merged, n_adv = inject_points(
        sample['pointcloud'], adv_pts_np, injection_pos,
        remove_overlap=True,
    )
    return merged, n_adv
