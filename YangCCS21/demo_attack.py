"""
Demo: run a short white-box attack and generate visualization plots.

Outputs (in results/):
  - demo_loss_curves.png        — loss over iterations
  - demo_asr_bar.png            — ASR before/after attack on eval subset
  - demo_bev_examples.png       — BEV view of 4 example frames (clean vs adversarial)
  - demo_rpn_score_histogram.png — RPN score distribution at adversarial points

Usage:
    python demo_attack.py --device cuda:0 --n_iters 30 --batch 4 --n_eval 20
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

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
from attack.loss import appearing_loss, apply_physical_constraints


# ── Helpers ──────────────────────────────────────────────────────────────

def load_config(path='configs/attack_config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def get_injection_pos(sample, inj_cfg, rng):
    pc, gt = sample['pointcloud'], sample['gt_bboxes']
    occ, gi = build_bev_occupancy(
        pc, gt,
        x_range=tuple(inj_cfg['x_range']),
        y_range=tuple(inj_cfg['y_range']),
        resolution=inj_cfg['resolution'],
        margin=inj_cfg['margin'],
    )
    pos, valid = sample_injection_position(
        occ, gi,
        min_clearance=inj_cfg['min_clearance'],
        fallback_pos=tuple(inj_cfg['fallback_pos']),
        rng=rng,
    )
    return pos, valid


# ── Plot functions ───────────────────────────────────────────────────────

def plot_loss_curves(history, save_path):
    """4-panel loss curve plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    items = [
        ('L_total', 'Total Loss', 'black'),
        ('L_cls', 'Classification Loss (L_cls)', 'red'),
        ('L_feat', 'Feature Loss (L_feat)', 'blue'),
        ('L_area', 'Area Regularizer (L_area)', 'green'),
    ]
    for ax, (key, title, color) in zip(axes.flat, items):
        vals = history.get(key, [])
        if vals:
            ax.plot(vals, color=color, alpha=0.4, linewidth=0.8)
            if len(vals) >= 5:
                k = min(5, len(vals))
                kernel = np.ones(k) / k
                sm = np.convolve(vals, kernel, mode='valid')
                ax.plot(range(k - 1, len(vals)), sm, color=color, linewidth=2,
                        label=f'Smoothed (k={k})')
                ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    plt.suptitle('White-box Attack Loss Curves', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


def plot_asr_bar(n_eligible, n_success_clean, n_success_adv, save_path):
    """Bar chart: detection rate at injection position (clean vs adversarial)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Clean Scene', 'After Attack']
    rates = [
        n_success_clean / max(n_eligible, 1) * 100,
        n_success_adv / max(n_eligible, 1) * 100,
    ]
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(labels, rates, color=colors, width=0.5, edgecolor='black')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate at Injection Position (%)')
    ax.set_title(f'Appearing Attack Success Rate\n(evaluated on {n_eligible} eligible frames)')
    ax.set_ylim(0, max(max(rates) + 15, 30))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  Saved → {save_path}')


def plot_bev_examples(examples, save_path):
    """BEV scatter plot: clean scene + adversarial points + detection boxes."""
    n = min(len(examples), 4)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, ex in enumerate(examples[:n]):
        # Top row: clean
        ax_clean = axes[0, col]
        pc_clean = ex['pc_clean']
        ax_clean.scatter(pc_clean[:, 0], pc_clean[:, 1], s=0.05, c='gray', alpha=0.3,
                         label='Scene points')
        if len(ex['boxes_clean']) > 0:
            for bi, box in enumerate(ex['boxes_clean']):
                _draw_bev_box(ax_clean, box, color='green',
                              label='Clean detection' if bi == 0 else None)
        inj = ex['inj_pos']
        ax_clean.plot(inj[0], inj[1], 'x', color='red', markersize=12,
                      markeredgewidth=3, label='Injection position')
        ax_clean.set_title(f"Frame {ex['sid']} — Clean", fontsize=10)
        ax_clean.set_xlim(0, 70)
        ax_clean.set_ylim(-40, 40)
        ax_clean.set_aspect('equal')
        ax_clean.grid(True, alpha=0.2)
        ax_clean.legend(fontsize=7, loc='upper right', markerscale=3)

        # Bottom row: adversarial
        ax_adv = axes[1, col]
        pc_adv = ex['pc_adv']
        n_adv = ex['n_adv']
        scene_pts = pc_adv[:-n_adv] if n_adv > 0 else pc_adv
        adv_pts = pc_adv[-n_adv:] if n_adv > 0 else np.empty((0, 4))
        ax_adv.scatter(scene_pts[:, 0], scene_pts[:, 1], s=0.05, c='gray', alpha=0.3,
                       label='Scene points')
        if n_adv > 0:
            ax_adv.scatter(adv_pts[:, 0], adv_pts[:, 1], s=3, c='red', alpha=0.8,
                           label=f'Adversarial points ({n_adv})')
        if len(ex['boxes_adv']) > 0:
            for bi, box in enumerate(ex['boxes_adv']):
                _draw_bev_box(ax_adv, box, color='orange',
                              label='Detection after attack' if bi == 0 else None)
        ax_adv.plot(inj[0], inj[1], 'x', color='red', markersize=12,
                    markeredgewidth=3, label='Injection position')
        ax_adv.set_title(f"Frame {ex['sid']} — After Attack", fontsize=10)
        ax_adv.set_xlim(0, 70)
        ax_adv.set_ylim(-40, 40)
        ax_adv.set_aspect('equal')
        ax_adv.grid(True, alpha=0.2)
        ax_adv.legend(fontsize=7, loc='upper right', markerscale=3)

    axes[0, 0].set_ylabel('Clean Scene\nY (m)', fontsize=10)
    axes[1, 0].set_ylabel('After Attack\nY (m)', fontsize=10)
    plt.suptitle('BEV Visualization: Clean vs Adversarial', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


def _draw_bev_box(ax, box, color='green', label=None):
    """Draw a rotated BEV box on a matplotlib axis."""
    cx, cy, cz, l, w, h, yaw = box[:7]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [l / 2, w / 2], [-l / 2, w / 2],
        [-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2],
    ])
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated = (R @ corners.T).T + np.array([cx, cy])
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=1.5, label=label)


def plot_rpn_histogram(scores_clean, scores_adv, save_path):
    """Histogram of RPN scores at adversarial point locations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(scores_clean) > 0:
        ax.hist(scores_clean, bins=30, alpha=0.5, color='blue', label='Before attack (random pts)')
    if len(scores_adv) > 0:
        ax.hist(scores_adv, bins=30, alpha=0.5, color='red', label='After attack (adv pts)')
    ax.set_xlabel('RPN Point Classification Score')
    ax.set_ylabel('Count')
    ax.set_title('RPN Score Distribution at Adversarial Point Locations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  Saved → {save_path}')


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Demo: short attack + plots')
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_iters', type=int, default=30,
                        help='Number of optimization steps (default: 30)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Frames per iteration (default: 4)')
    parser.add_argument('--n_eval', type=int, default=20,
                        help='Frames for ASR evaluation (default: 20)')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    device = args.device

    # Set default CUDA device for OpenPCDet's custom CUDA kernels
    if device.startswith('cuda'):
        import re
        m = re.search(r'cuda:(\d+)', device)
        if m:
            torch.cuda.set_device(int(m.group(1)))
    atk = config['attack']
    inj_cfg = atk['injection']
    lidar_cfg = atk.get('lidar', {})

    # ── Load model & data ──
    print("=" * 60)
    print("Demo: Short White-box Attack + Visualization")
    print("=" * 60)
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device,
    )
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split='val',
        pc_range=config['data']['pc_range'],
    )
    print(f"  {len(dataset)} frames loaded")

    # Load ref feature
    ref_path = config.get('ref_feature', {}).get('output_path', 'results/ref_car_feature.pt')
    if os.path.exists(ref_path):
        ref_feature = torch.load(ref_path, map_location=device, weights_only=True)
        # Flatten if needed
        ref_feature = ref_feature.squeeze()
        print(f"  Loaded ref feature: {ref_feature.shape}")
    else:
        ref_feature = torch.zeros(128, device=device)
        print(f"  WARNING: ref feature not found, using zeros")

    # ── Pre-compute injection positions ──
    print("\nPre-computing injection positions...")
    rng = np.random.RandomState(42)
    n_use = min(200, len(dataset))
    inj_cache = {}
    valid_idx = []
    for i in range(n_use):
        pos, valid = get_injection_pos(dataset[i], inj_cfg, rng)
        inj_cache[i] = pos
        if valid:
            valid_idx.append(i)
    if not valid_idx:
        valid_idx = list(range(n_use))
    print(f"  {len(valid_idx)} valid injection positions")

    # ── Phase A: Short white-box attack ──
    print(f"\n{'='*60}")
    print(f"Phase A: White-box attack ({args.n_iters} iters, batch={args.batch})")
    print(f"{'='*60}")

    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0, faces = v0.to(device), faces.to(device)
    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)
    c = torch.tensor(atk['translation_limit'], dtype=torch.float32, device=device)
    R = torch.eye(3, device=device)
    sensor_pos = torch.zeros(3, device=device)

    delta_v = torch.zeros_like(v0, requires_grad=True)
    t_tilde = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_v, t_tilde], lr=atk['lr'])

    lw = atk['loss_weights']
    target_car_size = tuple(atk.get('phantom_box_size', [3.9, 1.6, 1.56]))

    history = {'L_total': [], 'L_cls': [], 'L_loc': [], 'L_size': [], 'L_feat': [], 'L_lap': []}

    for step in tqdm(range(args.n_iters), desc='Optimizing'):
        optimizer.zero_grad()
        batch_idx = np.random.choice(valid_idx, size=min(args.batch, len(valid_idx)), replace=False)
        vertices = reparameterize(v0, delta_v, t_tilde, R, b, c)

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_dict = {k: 0.0 for k in history}
        n_ok = 0

        for fi in batch_idx:
            sample = dataset[int(fi)]
            inj_pos = inj_cache[int(fi)]
            vw = vertices + torch.tensor(inj_pos, dtype=torch.float32, device=device).unsqueeze(0)
            adv_pts = render_adversarial_points(
                vw, faces, sensor_pos,
                n_elevation=lidar_cfg.get('n_elevation', 64),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -24.9),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 2.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.08),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
            )
            if adv_pts.shape[0] == 0:
                continue
            pc_t = torch.tensor(sample['pointcloud'], dtype=torch.float32, device=device)
            merged, n_adv = inject_points(pc_t, adv_pts, torch.zeros(3, device=device), remove_overlap=True)
            n_scene = merged.shape[0] - n_adv
            result = wrapper.forward_with_grad(merged, rpn_only=True, run_post=False)
            pt_cls = result.get('point_cls_logits')
            rpn_box = result.get('rpn_box_preds')
            pt_feat = result.get('point_features')
            if pt_cls is None:
                continue
            loss, ld = appearing_loss(
                pt_cls, rpn_box, pt_feat, ref_feature, vertices, faces, adj,
                n_scene, n_adv, injection_pos=inj_pos, device=device,
                kappa=lw.get('kappa', 5.0),
                alpha_feat=lw['alpha_feat'],
                beta_loc=lw['beta_loc'],
                beta_size=lw['beta_size'],
                lambda_lap=lw['lambda_lap'],
                target_size=target_car_size,
            )
            batch_loss = batch_loss + loss
            for k in step_dict:
                step_dict[k] += ld.get(k, 0.0)
            n_ok += 1

        if n_ok > 0:
            batch_loss = batch_loss / n_ok
            for k in step_dict:
                step_dict[k] /= n_ok

        if batch_loss.requires_grad:
            batch_loss.backward()
            optimizer.step()
        with torch.no_grad():
            apply_physical_constraints(delta_v, v0, tuple(atk['size_limit']))

        for k in history:
            history[k].append(step_dict.get(k, 0.0))

    # Save checkpoint
    ckpt_path = os.path.join(args.output, 'adv_mesh_demo.pth')
    torch.save({
        'delta_v': delta_v.detach().cpu(),
        't_tilde': t_tilde.detach().cpu(),
        'v0': v0.detach().cpu(),
        'faces': faces.detach().cpu(),
        'history': history,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # Plot 1: loss curves
    plot_loss_curves(history, os.path.join(args.output, 'demo_loss_curves.png'))

    # ── Phase B: Evaluate ASR + collect examples ──
    print(f"\n{'='*60}")
    print(f"Phase B: Evaluating ASR on {args.n_eval} frames")
    print(f"{'='*60}")

    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']
    rng_eval = np.random.RandomState(123)

    n_eligible = 0
    n_success_clean = 0
    n_success_adv = 0
    examples = []
    rpn_scores_clean_all = []
    rpn_scores_adv_all = []

    eval_indices = np.random.RandomState(99).choice(valid_idx, size=min(args.n_eval, len(valid_idx)), replace=False)

    for fi in tqdm(eval_indices, desc='Evaluating'):
        sample = dataset[int(fi)]
        pc_np = sample['pointcloud']
        inj_pos = inj_cache[int(fi)]

        # Clean detection
        pc_t = torch.tensor(pc_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            boxes_clean, scores_clean = wrapper.detect(pc_t, score_thresh)

        # Check clean: any detection near injection?
        def near_det(boxes, scores, pos, prox, thresh):
            if len(boxes) == 0:
                return False
            d = np.sqrt((boxes[:, 0] - pos[0])**2 + (boxes[:, 1] - pos[1])**2)
            return bool(((d < prox) & (scores >= thresh)).any())

        clean_has = near_det(boxes_clean, scores_clean, inj_pos, proximity, score_thresh)
        if clean_has:
            continue  # not eligible

        n_eligible += 1

        # Apply attack
        with torch.no_grad():
            verts = reparameterize(v0, delta_v.detach(), t_tilde.detach(), R, b, c)
            inj_t = torch.tensor(inj_pos, dtype=torch.float32, device=device)
            vw = verts + inj_t.unsqueeze(0)
            adv_pts = render_adversarial_points(
                vw, faces, sensor_pos,
                n_elevation=lidar_cfg.get('n_elevation', 64),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -24.9),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 2.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.08),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
            )
        if adv_pts.shape[0] == 0:
            continue

        adv_np = adv_pts.cpu().numpy()
        pc_adv = np.vstack([pc_np, np.hstack([adv_np, np.ones((len(adv_np), 1))])])
        n_adv = len(adv_np)

        pc_adv_t = torch.tensor(pc_adv, dtype=torch.float32, device=device)
        with torch.no_grad():
            boxes_adv, scores_adv = wrapper.detect(pc_adv_t, score_thresh)

        adv_has = near_det(boxes_adv, scores_adv, inj_pos, proximity, score_thresh)
        if adv_has:
            n_success_adv += 1

        # Collect RPN scores at adv point region (for histogram)
        with torch.no_grad():
            pc_adv_t2 = torch.tensor(pc_adv, dtype=torch.float32, device=device)
            pc_adv_t2.requires_grad_(True)
            r = wrapper.forward_with_grad(pc_adv_t2)
            pt_cls = r.get('point_cls_scores')
            if pt_cls is not None:
                adv_scores = pt_cls[-n_adv:].cpu().numpy()
                rpn_scores_adv_all.append(adv_scores)
                # Random scene points for comparison
                if pc_np.shape[0] > n_adv:
                    idx_rand = np.random.choice(pc_np.shape[0], size=n_adv, replace=False)
                    clean_scores = pt_cls[idx_rand].cpu().numpy()
                    rpn_scores_clean_all.append(clean_scores)

        # Save example for BEV plot
        if len(examples) < 4:
            examples.append({
                'sid': sample['sample_id'],
                'pc_clean': pc_np,
                'pc_adv': pc_adv,
                'n_adv': n_adv,
                'boxes_clean': boxes_clean,
                'boxes_adv': boxes_adv,
                'inj_pos': inj_pos,
            })

    asr = n_success_adv / max(n_eligible, 1) * 100
    print(f"\n  Eligible frames: {n_eligible}")
    print(f"  Attack successes: {n_success_adv}")
    print(f"  ASR: {asr:.1f}%")

    # Plot 2: ASR bar chart
    plot_asr_bar(n_eligible, n_success_clean, n_success_adv,
                 os.path.join(args.output, 'demo_asr_bar.png'))

    # Plot 3: BEV examples
    if examples:
        plot_bev_examples(examples, os.path.join(args.output, 'demo_bev_examples.png'))

    # Plot 4: RPN score histogram
    sc = np.concatenate(rpn_scores_clean_all) if rpn_scores_clean_all else np.array([])
    sa = np.concatenate(rpn_scores_adv_all) if rpn_scores_adv_all else np.array([])
    if len(sa) > 0:
        plot_rpn_histogram(sc, sa, os.path.join(args.output, 'demo_rpn_score_histogram.png'))

    print(f"\n{'='*60}")
    print(f"Demo complete! All plots saved to {args.output}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
