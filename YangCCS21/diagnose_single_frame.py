# -*- coding: utf-8 -*-
"""
Single-frame diagnostic attack.

Optimizes adversarial mesh on ONE fixed KITTI frame for N steps,
monitoring three levels simultaneously at each step:

  Level 1: RPN per-point foreground score (sigmoid) at adversarial points
  Level 2: Number of RPN proposals within 3m of injection position
  Level 3: RCNN final detection confidence (post-NMS) near injection

Usage:
  python diagnose_single_frame.py --tag close --dist 5 10 --size 0.8 0.5 0.5
  python diagnose_single_frame.py --tag big --dist 5 35 --size 2.0 1.0 1.0
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset
from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from attack.reparameterize import reparameterize
from attack.inject import (
    build_bev_occupancy, sample_injection_position, inject_points,
)


PROXIMITY = 5.0


def find_frame_at_distance(dataset, wrapper, inj_cfg, dist_min, dist_max, device):
    """Find a frame with a valid injection position at the specified distance range."""
    rng = np.random.RandomState(777)
    for idx in range(min(100, len(dataset))):
        sample = dataset[idx]
        pc = sample['pointcloud']
        gt = sample['gt_bboxes']
        if len(gt) == 0:
            continue

        occ, gi = build_bev_occupancy(
            pc, gt,
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )

        for _ in range(20):
            pos, valid = sample_injection_position(
                occ, gi,
                min_clearance=inj_cfg['min_clearance'],
                fallback_pos=tuple(inj_cfg['fallback_pos']),
                rng=rng,
            )
            if not valid:
                continue

            dist = np.sqrt(pos[0]**2 + pos[1]**2)
            if dist < dist_min or dist > dist_max:
                continue

            pc_t = torch.from_numpy(pc.astype(np.float32)).to(device)
            boxes, scores = wrapper.detect(pc_t, score_thresh=0.1)
            if len(boxes) > 0:
                dists_det = np.sqrt((boxes[:, 0] - pos[0])**2 + (boxes[:, 1] - pos[1])**2)
                if dists_det.min() < PROXIMITY:
                    continue

            lidar_dist = np.sqrt(pos[0]**2 + pos[1]**2)
            sid = sample["sample_id"]
            print(f"  Found frame idx={idx}, id={sid}, inj_pos={pos}, dist={lidar_dist:.1f}m")
            return sample, pos

    sample = dataset[0]
    if dist_max <= 15:
        pos = np.array([7.0, -3.0, 0.0])
    else:
        pos = np.array([25.0, -10.0, 0.0])
    print(f"  Fallback: using frame 0 with pos={pos}")
    return sample, pos


def count_proposals_near(rois_np, pos, radius=3.0):
    if rois_np is None or len(rois_np) == 0:
        return 0
    dists = np.sqrt((rois_np[:, 0] - pos[0])**2 + (rois_np[:, 1] - pos[1])**2)
    return int((dists < radius).sum())


def best_rcnn_confidence_near(pred_dicts, pos, radius=3.0):
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


def run_diagnosis(tag, size_limit, dist_min, dist_max, n_iters, lr, device,
                  enable_ste=False):
    with open('configs/attack_config.yaml') as f:
        config = yaml.safe_load(f)

    atk = config['attack']
    inj_cfg = atk['injection']
    lidar_cfg = atk.get('lidar', {})

    mode_str = 'STE+RCNN' if enable_ste else 'RPN-only'
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS: {tag} ({mode_str})")
    print(f"  Object size: {size_limit}")
    print(f"  Distance range: {dist_min}-{dist_max}m")
    print(f"  Iters: {n_iters}, LR: {lr}")
    print(f"{'='*70}")

    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=device,
        enable_ste=enable_ste,
    )

    dataset = KITTIDataset(config['data']['kitti_root'], split='val',
                           pc_range=config['data']['pc_range'])

    print("Finding frame...")
    sample, inj_pos = find_frame_at_distance(
        dataset, wrapper, inj_cfg, dist_min, dist_max, device
    )

    b = torch.tensor(size_limit, dtype=torch.float32, device=device)
    c = torch.tensor([0.3, 0.3, 0.0], dtype=torch.float32, device=device)
    v0, faces, _ = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0 = v0.to(device)
    faces = faces.to(device)

    delta_v = torch.zeros_like(v0, requires_grad=True)
    t_tilde = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_v, t_tilde], lr=lr)

    R = torch.eye(3, device=device)
    sensor_pos = torch.zeros(3, device=device)
    inj_t = torch.tensor(inj_pos, dtype=torch.float32, device=device)
    pc_np = sample['pointcloud']

    hist = {
        'rpn_fg_score_mean': [], 'rpn_fg_score_max': [],
        'rpn_fg_logit_mean': [],
        'n_proposals_near': [],
        'rcnn_best_conf': [],
        'loss': [], 'L_cls': [], 'L_rcnn': [],
        'grad_norm': [], 'n_adv_pts': [],
    }

    lidar_dist = np.sqrt(inj_pos[0]**2 + inj_pos[1]**2)
    print(f"\nStarting: {n_iters} iters, size={size_limit}, dist={lidar_dist:.1f}m")
    print("-" * 100)
    print(f"{'step':>5} | {'loss':>8} | {'L_cls':>7} | {'L_rcnn':>7} | {'fg_logit':>8} | "
          f"{'#prop':>5} | {'rcnn_conf':>9} | {'grad':>8} | {'#pts':>5}")
    print("-" * 100)

    from attack.loss import L_rcnn_cls

    for step in range(n_iters):
        optimizer.zero_grad()

        vertices = reparameterize(v0, delta_v, t_tilde, R, b, c)
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
            for k in hist:
                hist[k].append(0.0)
            continue

        pc_tensor = torch.tensor(pc_np, dtype=torch.float32, device=device)
        merged_pc, n_adv = inject_points(
            pc_tensor, adv_pts, torch.zeros(3, device=device),
            remove_overlap=True,
        )
        n_scene = merged_pc.shape[0] - n_adv

        run_post = (step % 10 == 0) or (step == n_iters - 1)
        result = wrapper.forward_with_grad(merged_pc, rpn_only=False, run_post=run_post)

        fg_scores = result.get('point_cls_scores')
        fg_logits = result.get('point_cls_logits')

        if fg_scores is not None and n_adv > 0:
            adv_scores = fg_scores[n_scene:n_scene + n_adv]
            fg_mean = adv_scores.mean().item()
            fg_max = adv_scores.max().item()
        else:
            fg_mean = fg_max = 0.0

        if fg_logits is not None and n_adv > 0:
            adv_logits = fg_logits[n_scene:n_scene + n_adv]
            logit_mean = adv_logits.mean().item()
        else:
            logit_mean = 0.0

        rois = result.get('rois')
        if rois is not None:
            rois_np = rois.detach().cpu().numpy().squeeze(0)
            n_prop = count_proposals_near(rois_np, inj_pos, PROXIMITY)
        else:
            n_prop = 0

        rcnn_conf = 0.0
        if run_post:
            rcnn_conf = best_rcnn_confidence_near(result.get('pred_dicts'), inj_pos, PROXIMITY)

        # --- Loss computation ---
        l_cls_val = 0.0
        l_rcnn_val = 0.0

        loss_terms = []
        if fg_logits is not None and n_adv > 0:
            adv_logits_for_loss = fg_logits[n_scene:n_scene + n_adv]
            l_cls = -adv_logits_for_loss.mean()
            loss_terms.append(l_cls)
            l_cls_val = l_cls.item()

        if enable_ste:
            rcnn_cls = result.get('rcnn_cls_preds')
            rois_t = result.get('rois')
            if rcnn_cls is not None and rois_t is not None:
                l_rcnn = L_rcnn_cls(rcnn_cls, inj_pos, rois_t, proximity=PROXIMITY)
                loss_terms.append(2.0 * l_rcnn)
                l_rcnn_val = l_rcnn.item()

        if loss_terms:
            loss = sum(loss_terms)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()

        grad_scale = 10.0
        if delta_v.grad is not None:
            delta_v.grad.data.mul_(grad_scale)
        if t_tilde.grad is not None:
            t_tilde.grad.data.mul_(grad_scale)

        gn = 0.0
        if delta_v.grad is not None:
            gn = delta_v.grad.norm().item()

        optimizer.step()

        hist['rpn_fg_score_mean'].append(fg_mean)
        hist['rpn_fg_score_max'].append(fg_max)
        hist['rpn_fg_logit_mean'].append(logit_mean)
        hist['n_proposals_near'].append(n_prop)
        hist['rcnn_best_conf'].append(rcnn_conf)
        hist['loss'].append(loss.item())
        hist['L_cls'].append(l_cls_val)
        hist['L_rcnn'].append(l_rcnn_val)
        hist['grad_norm'].append(gn)
        hist['n_adv_pts'].append(n_adv)

        if step % 25 == 0 or step == n_iters - 1:
            print(f"{step:5d} | {loss.item():8.4f} | {l_cls_val:7.4f} | {l_rcnn_val:7.4f} | "
                  f"{logit_mean:8.4f} | {n_prop:5d} | {rcnn_conf:9.4f} | "
                  f"{gn:8.2f} | {n_adv:5d}")

    print(f"\n{'='*70}")
    print(f"SUMMARY [{tag}] ({mode_str}, size={size_limit}, dist={lidar_dist:.1f}m)")
    print(f"{'='*70}")
    print(f"  RPN fg score: {hist['rpn_fg_score_mean'][0]:.4f} -> {hist['rpn_fg_score_mean'][-1]:.4f}")
    print(f"  RPN fg logit: {hist['rpn_fg_logit_mean'][0]:.4f} -> {hist['rpn_fg_logit_mean'][-1]:.4f}")
    print(f"  Proposals near: {hist['n_proposals_near'][0]} -> max={max(hist['n_proposals_near'])}")
    rcnn_vals = [v for v in hist['rcnn_best_conf'] if v > 0]
    max_rcnn = max(rcnn_vals) if rcnn_vals else 0.0
    print(f"  RCNN conf: max={max_rcnn:.4f}")
    print(f"  Adv points: {hist['n_adv_pts'][0]} -> {hist['n_adv_pts'][-1]}")
    print(f"  Grad norm: first={hist['grad_norm'][0]:.4f}, last={hist['grad_norm'][-1]:.4f}")

    if max_rcnn >= 0.3:
        print(f"\n  *** SUCCESS: RCNN confidence reached {max_rcnn:.3f}! ***")
    elif max(hist['rpn_fg_score_mean']) > 0.5 and max(hist['n_proposals_near']) == 0:
        print("\n  BOTTLENECK: RPN scores high but no proposals generated.")
    elif max(hist['rpn_fg_score_mean']) < 0.1:
        print("\n  BOTTLENECK: RPN scores remain very low.")
    elif max(hist['n_proposals_near']) > 0 and max_rcnn < 0.3:
        print("\n  BOTTLENECK: Proposals exist but RCNN rejects them.")
    else:
        print("\n  Inconclusive.")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    sid = sample["sample_id"]
    npts = hist["n_adv_pts"][-1]
    title = f"Diagnostic [{tag}] {mode_str}, frame={sid}, size={size_limit}, dist={lidar_dist:.0f}m, #pts~{npts}"
    fig.suptitle(title, fontsize=13)

    def smooth(arr, w=20):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    ax = axes[0, 0]
    ax.plot(hist['rpn_fg_score_mean'], label='mean', alpha=0.4, color='C0')
    ax.plot(hist['rpn_fg_score_max'], label='max', alpha=0.4, color='C1')
    sm = smooth(hist['rpn_fg_score_mean'])
    ax.plot(range(19, len(sm)+19), sm, 'k-', lw=2, label='mean (smooth)')
    ax.set_title('RPN FG Score (sigmoid)')
    ax.set_xlabel('Step'); ax.set_ylabel('Score'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(hist['rpn_fg_logit_mean'], alpha=0.4, color='C0')
    sm = smooth(hist['rpn_fg_logit_mean'])
    ax.plot(range(19, len(sm)+19), sm, 'k-', lw=2)
    ax.set_title('RPN FG Logit')
    ax.set_xlabel('Step'); ax.set_ylabel('Logit'); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(hist['n_proposals_near'], color='C2')
    ax.set_title('# Proposals near injection')
    ax.set_xlabel('Step'); ax.set_ylabel('Count'); ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    ax.plot(hist['rcnn_best_conf'], color='C3')
    ax.axhline(y=0.3, color='r', ls='--', label='threshold=0.3')
    ax.set_title('RCNN Best Confidence')
    ax.set_xlabel('Step'); ax.set_ylabel('Confidence'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(hist['loss'], alpha=0.4, color='C4')
    sm = smooth(hist['loss'])
    ax.plot(range(19, len(sm)+19), sm, 'k-', lw=2)
    ax.set_title('Total Loss')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(hist['L_cls'], alpha=0.4, color='C1', label='L_cls')
    sm = smooth(hist['L_cls'])
    ax.plot(range(19, len(sm)+19), sm, 'k-', lw=2)
    ax.set_title('L_cls (RPN)')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(hist['L_rcnn'], alpha=0.4, color='C3', label='L_rcnn')
    sm = smooth(hist['L_rcnn'])
    ax.plot(range(19, len(sm)+19), sm, 'k-', lw=2)
    ax.set_title('L_rcnn (RCNN)')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    ax.plot(hist['grad_norm'], alpha=0.4, color='C5')
    ax.set_title('Gradient Norm (delta_v)')
    ax.set_xlabel('Step'); ax.set_ylabel('||grad||'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'results/diagnose_{tag}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {save_path}")

    torch.save({
        'delta_v': delta_v.detach().cpu(),
        't_tilde': t_tilde.detach().cpu(),
        'v0': v0.cpu(), 'faces': faces.cpu(),
        'history': hist,
        'frame_id': sample['sample_id'],
        'injection_pos': inj_pos.tolist() if hasattr(inj_pos, 'tolist') else list(inj_pos),
        'size_limit': size_limit,
        'lidar_dist': lidar_dist,
        'mode': mode_str,
    }, f'results/diagnose_{tag}.pth')

    wrapper.remove_hook()
    return hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--size', type=float, nargs=3, required=True)
    parser.add_argument('--dist', type=float, nargs=2, required=True,
                        help='min max distance to LiDAR')
    parser.add_argument('--n_iters', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ste', action='store_true',
                        help='Enable STE + RCNN loss')
    args = parser.parse_args()

    torch.cuda.set_device(int(args.device.split(':')[1]) if ':' in args.device else 0)
    run_diagnosis(args.tag, args.size, args.dist[0], args.dist[1],
                  args.n_iters, args.lr, args.device,
                  enable_ste=args.ste)
