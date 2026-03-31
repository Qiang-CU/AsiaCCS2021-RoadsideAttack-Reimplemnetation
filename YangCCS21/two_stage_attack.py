"""
Single-stage white-box appearing attack against PointRCNN.

RPN-level optimization: push foreground scores + box center/size predictions
so proposals appear near the injection position (inspired by Tu et al. CVPR 2020).

Usage:
    python two_stage_attack.py --device cuda:0
    python two_stage_attack.py --device cuda:0 --warm_start results/adv_mesh_whitebox_latest.pth
    python two_stage_attack.py --device cuda:0 --single_frame   # quick single-frame diagnostic
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

from utils.kitti_utils import KITTIDataset
from attack.whitebox import run_whitebox_attack
from model.pointrcnn_wrapper import PointRCNNWrapper
from attack.inject import inject_points


def plot_two_stage_history(history, save_path):
    """Plot loss curves and monitoring metrics for two-stage attack."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    stages = np.array(history.get('stage', []))
    n = len(stages)
    if n == 0:
        plt.close()
        return
    x = np.arange(n)
    s1_mask = stages == 1
    s2_mask = stages == 2

    def plot_with_stages(ax, key, title, color='C0'):
        vals = history.get(key, [])
        if not vals:
            ax.set_title(title + ' (no data)')
            return
        vals = np.array(vals[:n])
        if s1_mask.any():
            ax.plot(x[s1_mask], vals[s1_mask], '.', color=color, alpha=0.3, markersize=2)
        if s2_mask.any():
            ax.plot(x[s2_mask], vals[s2_mask], '.', color='C3', alpha=0.3, markersize=2)

        w = min(20, max(3, n // 30))
        if len(vals) >= w:
            kernel = np.ones(w) / w
            sm = np.convolve(vals, kernel, mode='valid')
            ax.plot(range(w - 1, len(vals)), sm, 'k-', lw=1.5)

        if s2_mask.any():
            s2_start = np.where(s2_mask)[0][0]
            ax.axvline(x=s2_start, color='red', ls='--', alpha=0.5, label='Stage 2 start')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

    plot_with_stages(axes[0, 0], 'L_total', 'Total Loss')
    plot_with_stages(axes[0, 1], 'L_cls', 'RPN Classification Loss (L_cls)', 'C1')
    plot_with_stages(axes[0, 2], 'L_loc', 'RPN Box Location Loss (L_loc)', 'C2')
    plot_with_stages(axes[1, 0], 'L_size', 'RPN Box Size Loss (L_size)', 'C4')
    plot_with_stages(axes[1, 1], 'L_feat', 'Feature Loss (L_feat)', 'C5')
    plot_with_stages(axes[1, 2], 'L_rcnn', 'RCNN Detection Loss (L_rcnn)', 'C3')

    ax = axes[2, 0]
    vals = history.get('n_proposals_near', [])
    if vals:
        ax.plot(vals, 'C2', alpha=0.5, lw=0.8)
        if len(vals) >= 10:
            sm = np.convolve(vals, np.ones(10) / 10, mode='valid')
            ax.plot(range(9, len(vals)), sm, 'k-', lw=1.5)
    ax.set_title('# Proposals Near Injection')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    vals = history.get('rcnn_best_conf', [])
    if vals:
        ax.plot(vals, 'C3', alpha=0.5, lw=0.8)
        if len(vals) >= 10:
            sm = np.convolve(vals, np.ones(10) / 10, mode='valid')
            ax.plot(range(9, len(vals)), sm, 'k-', lw=1.5)
    ax.axhline(y=0.3, color='r', ls='--', alpha=0.5, label='threshold=0.3')
    ax.set_title('Best RCNN Confidence Near Injection')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    bx = history.get('b_x', [])
    by = history.get('b_y', [])
    bz = history.get('b_z', [])
    if bx:
        ax.plot(bx, label='b_x', color='C0')
        ax.plot(by, label='b_y', color='C1')
        ax.plot(bz, label='b_z', color='C2')
        ax.legend(fontsize=8)
    ax.set_title('Size Annealing (b)')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Two-Stage White-box Attack', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {save_path}")


def evaluate_asr(dataset, wrapper, delta_v, t_tilde, v0, faces,
                 config, injection_cache, valid_indices,
                 n_eval=50, device='cuda:0'):
    """Quick ASR evaluation after attack."""
    from attack.whitebox import apply_attack_to_sample

    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']
    rng = np.random.RandomState(999)
    eval_idx = rng.choice(valid_indices, size=min(n_eval, len(valid_indices)), replace=False)

    n_eligible = 0
    n_success = 0

    for fi in eval_idx:
        sample = dataset[int(fi)]
        inj_pos = injection_cache[int(fi)]['pos']

        pc_t = torch.tensor(sample['pointcloud'], dtype=torch.float32, device=device)
        with torch.no_grad():
            boxes_c, scores_c = wrapper.detect(pc_t, score_thresh)

        if len(boxes_c) > 0:
            d = np.sqrt((boxes_c[:, 0] - inj_pos[0])**2 + (boxes_c[:, 1] - inj_pos[1])**2)
            if (d < proximity).any():
                continue

        n_eligible += 1

        merged, n_adv = apply_attack_to_sample(
            sample, delta_v, t_tilde, v0, faces,
            config, device, injection_pos=inj_pos,
        )
        merged_t = torch.tensor(merged, dtype=torch.float32, device=device)
        with torch.no_grad():
            boxes_a, scores_a = wrapper.detect(merged_t, score_thresh)

        if len(boxes_a) > 0:
            d = np.sqrt((boxes_a[:, 0] - inj_pos[0])**2 + (boxes_a[:, 1] - inj_pos[1])**2)
            if ((d < proximity) & (scores_a >= score_thresh)).any():
                n_success += 1

    asr = n_success / max(n_eligible, 1) * 100
    print(f"\n  ASR Evaluation:")
    print(f"    Eligible frames: {n_eligible}")
    print(f"    Successes:       {n_success}")
    print(f"    ASR:             {asr:.1f}%")
    return asr, n_eligible, n_success


def generate_ref_feature(dataset, config, device, n_instances=200):
    """
    Extract average backbone feature for GT car points.

    Runs PointRCNN backbone on frames with GT car boxes, extracts
    point_features at points inside GT boxes, and averages them.
    """
    ref_path = config.get('ref_feature', {}).get('output_path', 'results/ref_car_feature.pt')
    if os.path.exists(ref_path):
        print(f"  ref_car_feature already exists at {ref_path}")
        return

    print("Generating ref_car_feature.pt from GT car instances...")
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=device,
        enable_ste=False,
    )

    all_feats = []
    count = 0
    for idx in range(min(len(dataset), 500)):
        if count >= n_instances:
            break
        sample = dataset[idx]
        gt_boxes = sample['gt_bboxes']
        if len(gt_boxes) == 0:
            continue

        pc = sample['pointcloud']
        pc_t = torch.tensor(pc, dtype=torch.float32, device=device)
        with torch.no_grad():
            result = wrapper.forward_with_grad(pc_t, rpn_only=True, run_post=False)

        pf = result.get('point_features')
        if pf is None:
            continue

        pts_xyz = pc_t[:, :3]
        for box in gt_boxes:
            cx, cy, cz, l, w, h, yaw = box[:7]
            dx = pts_xyz[:, 0] - cx
            dy = pts_xyz[:, 1] - cy
            dz = pts_xyz[:, 2] - cz
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            lx = dx * cos_y - dy * sin_y
            ly = dx * sin_y + dy * cos_y
            inside = (lx.abs() < l/2) & (ly.abs() < w/2) & (dz.abs() < h/2)
            if inside.sum() < 5:
                continue
            feat_inside = pf[inside].mean(dim=0)
            all_feats.append(feat_inside)
            count += 1
            if count >= n_instances:
                break

    wrapper.remove_hook()

    if all_feats:
        ref = torch.stack(all_feats).mean(dim=0)
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        torch.save(ref.cpu(), ref_path)
        print(f"  Saved ref_car_feature ({ref.shape}) to {ref_path}")
    else:
        print("  WARNING: no GT car features found, using zeros")
    del wrapper
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Two-stage white-box attack')
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warm_start', default=None, help='Checkpoint for warm-start')
    parser.add_argument('--output', default='results')
    parser.add_argument('--n_eval', type=int, default=50, help='Frames for ASR eval')
    parser.add_argument('--single_frame', action='store_true',
                        help='Run quick single-frame diagnostic instead')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['device'] = args.device

    if args.device.startswith('cuda'):
        import re
        m = re.search(r'cuda:(\d+)', args.device)
        if m:
            torch.cuda.set_device(int(m.group(1)))

    print("Loading KITTI dataset...")
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split='val',
        pc_range=config['data']['pc_range'],
    )
    print(f"  {len(dataset)} frames")

    generate_ref_feature(dataset, config, args.device)

    if args.single_frame:
        config['attack']['n_iters'] = 200
        config['attack']['multi_frame_batch'] = 1

    delta_v, t_tilde, history = run_whitebox_attack(
        dataset, config,
        save_dir=args.output,
        warm_start_ckpt=args.warm_start,
    )

    plot_two_stage_history(history, os.path.join(args.output, 'two_stage_curves.png'))

    print(f"\n{'='*70}")
    print(f"  Evaluating attack...")
    print(f"{'='*70}")

    from attack.whitebox import precompute_injections
    inj_cache, valid_idx = precompute_injections(dataset, config['attack']['injection'])

    eval_wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=args.device,
        enable_ste=False,
    )

    ckpt = torch.load(os.path.join(args.output, 'adv_mesh_whitebox_final.pth'),
                      map_location=args.device, weights_only=False)
    v0 = ckpt['v0']
    faces = ckpt['faces']

    asr, n_elig, n_succ = evaluate_asr(
        dataset, eval_wrapper, delta_v.to(args.device), t_tilde.to(args.device),
        v0.to(args.device), faces.to(args.device),
        config, inj_cache, valid_idx,
        n_eval=args.n_eval, device=args.device,
    )

    summary = {
        'asr': asr, 'eligible': n_elig, 'successes': n_succ,
        'n_iters': config['attack']['n_iters'],
    }
    import json
    with open(os.path.join(args.output, 'asr_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    eval_wrapper.remove_hook()
    print(f"\nDone. All results in {args.output}/")


if __name__ == '__main__':
    main()
