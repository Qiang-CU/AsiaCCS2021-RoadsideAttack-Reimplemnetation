"""
Phase 1 verification: Gradient chain validation.

Confirms that gradients from PointRCNN's classification loss flow back
to injected point cloud coordinates.

Pass criteria:
- delta_v.grad.norm() > 1e-6 with a meaningful scene
- Gradient is non-uniform across vertices
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset
from attack.mesh import create_icosphere
from attack.renderer import render_adversarial_points
from attack.reparameterize import reparameterize
from attack.inject import (
    build_bev_occupancy, sample_injection_position, inject_points,
)


def main():
    config_path = 'configs/attack_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model with STE patch
    enable_ste = config['attack'].get('enable_ste', False)
    print(f"Loading PointRCNN (STE={'ON' if enable_ste else 'OFF'})...")
    wrapper = PointRCNNWrapper(
        config_path=config['model']['pointrcnn_config'],
        ckpt_path=config['model']['pointrcnn_ckpt'],
        device=device,
        enable_ste=enable_ste,
    )

    # Load dataset
    print("Loading KITTI val split...")
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split='val',
        pc_range=config['data']['pc_range'],
    )

    atk = config['attack']
    inj_cfg = atk['injection']
    lidar_cfg = atk.get('lidar', {})

    # Create icosphere mesh
    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0 = v0.to(device)
    faces = faces.to(device)

    b = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)
    c = torch.tensor(atk['translation_limit'], dtype=torch.float32, device=device)

    # Optimizable parameters
    delta_v = torch.zeros_like(v0, requires_grad=True)
    t_tilde = torch.zeros(3, device=device, requires_grad=True)
    R = torch.eye(3, device=device)
    sensor_pos = torch.zeros(3, device=device)

    # Pick a frame with GT cars
    print("\nFinding a frame with GT cars...")
    sample = None
    inj_pos = None
    for idx in range(min(20, len(dataset))):
        s = dataset[idx]
        if len(s['gt_bboxes']) > 0:
            occ, grid_info = build_bev_occupancy(
                s['pointcloud'], s['gt_bboxes'],
                x_range=tuple(inj_cfg['x_range']),
                y_range=tuple(inj_cfg['y_range']),
                resolution=inj_cfg['resolution'],
                margin=inj_cfg['margin'],
            )
            pos, valid = sample_injection_position(
                occ, grid_info,
                min_clearance=inj_cfg['min_clearance'],
                fallback_pos=tuple(inj_cfg['fallback_pos']),
            )
            if valid:
                sample = s
                inj_pos = pos
                print(f"  Using frame {s['sample_id']} with {len(s['gt_bboxes'])} GT cars")
                print(f"  Injection position: {inj_pos}")
                break

    if sample is None:
        print("ERROR: No suitable frame found.")
        sys.exit(1)

    # Forward pass with gradient tracking
    print("\nRunning forward pass with gradient tracking...")

    vertices = reparameterize(v0, delta_v, t_tilde, R, b, c)
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
    print(f"  Rendered {adv_pts.shape[0]} adversarial points")

    if adv_pts.shape[0] == 0:
        print("ERROR: No adversarial points rendered. Check mesh/renderer.")
        sys.exit(1)

    # Inject into scene
    pc_tensor = torch.tensor(
        sample['pointcloud'], dtype=torch.float32, device=device
    )
    merged_pc, n_adv = inject_points(
        pc_tensor, adv_pts, torch.zeros(3, device=device),
        remove_overlap=True,
    )
    print(f"  Merged point cloud: {merged_pc.shape[0]} points ({n_adv} adversarial)")

    # Debug: check gradient tracking at each stage
    print(f"\n  [Debug] adv_pts requires_grad: {adv_pts.requires_grad}")
    print(f"  [Debug] adv_pts grad_fn: {adv_pts.grad_fn}")
    print(f"  [Debug] merged_pc requires_grad: {merged_pc.requires_grad}")
    print(f"  [Debug] merged_pc grad_fn: {merged_pc.grad_fn}")

    # Forward through PointRCNN (no NMS — preserves autograd graph)
    # Build batch dict manually to check gradient at each step
    batch_dict = wrapper.build_batch_dict(merged_pc)
    print(f"  [Debug] batch_dict['points'] requires_grad: {batch_dict['points'].requires_grad}")

    # Run each module and check gradient
    wrapper.last_feature = None
    for key in batch_dict:
        if isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = batch_dict[key].to(wrapper.device)

    with torch.set_grad_enabled(True):
        for i, cur_module in enumerate(wrapper.model.module_list):
            batch_dict = cur_module(batch_dict)
            mod_name = cur_module.__class__.__name__
            # Check if any new tensor has requires_grad
            for k, v in batch_dict.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    print(f"  [Debug] After {mod_name}: '{k}' has grad (shape={v.shape})")

    batch_cls_preds = batch_dict.get('batch_cls_preds')
    if batch_cls_preds is not None:
        cls_logits = batch_cls_preds.squeeze(0)
        cls_scores = torch.sigmoid(cls_logits)
        print(f"\n  Raw proposals: {cls_scores.shape[0]}")
        print(f"  Max cls score: {cls_scores.max().item():.4f}")
        print(f"  cls_scores requires_grad: {cls_scores.requires_grad}")
        print(f"  cls_scores grad_fn: {cls_scores.grad_fn}")

    # ── Test 1: RPN-level gradient (should always work) ──
    point_cls = batch_dict.get('point_cls_scores')
    print(f"\n  [RPN-level] point_cls_scores shape: "
          f"{point_cls.shape if point_cls is not None else None}")
    print(f"  [RPN-level] requires_grad: "
          f"{point_cls.requires_grad if point_cls is not None else None}")

    n_scene = merged_pc.shape[0] - n_adv

    # ── Test 2: RCNN-level gradient (requires STE) ──
    rcnn_cls = batch_dict.get('batch_cls_preds')
    rois = batch_dict.get('rois')
    print(f"\n  [RCNN-level] batch_cls_preds shape: "
          f"{rcnn_cls.shape if rcnn_cls is not None else None}")
    print(f"  [RCNN-level] requires_grad: "
          f"{rcnn_cls.requires_grad if rcnn_cls is not None else None}")
    print(f"  [RCNN-level] grad_fn: "
          f"{rcnn_cls.grad_fn if rcnn_cls is not None else None}")

    if rcnn_cls is not None and rois is not None:
        rcnn_scores = torch.sigmoid(rcnn_cls.squeeze())
        print(f"  RCNN scores: shape={rcnn_scores.shape}, "
              f"max={rcnn_scores.max().item():.4f}, "
              f"mean={rcnn_scores.mean().item():.4f}")

    # ── Test A: Direct STE gradient test (bypass proposal location issue) ──
    # Use the raw RCNN logits directly — no sigmoid (avoids saturation),
    # no proximity filtering (tests the gradient chain, not the attack logic).
    use_rcnn = (enable_ste and rcnn_cls is not None
                and rcnn_cls.requires_grad)

    if use_rcnn:
        print(f"\n  === Test A: Direct RCNN logit gradient (STE chain) ===")
        rcnn_logits = rcnn_cls.squeeze()  # (K,) raw logits
        if rcnn_logits.dim() == 0:
            rcnn_logits = rcnn_logits.unsqueeze(0)

        # Use raw logit sum — avoids sigmoid saturation entirely.
        # If gradient flows, logit.sum().backward() WILL produce nonzero grad.
        loss = rcnn_logits.sum()
        print(f"  loss = rcnn_logits.sum() = {loss.item():.4f}")
        print(f"  rcnn_logits stats: min={rcnn_logits.min().item():.4f}, "
              f"max={rcnn_logits.max().item():.4f}, "
              f"mean={rcnn_logits.mean().item():.4f}")
    elif point_cls is not None and point_cls.requires_grad:
        print(f"\n  === Test A: RPN-level gradient (fallback) ===")
        adv_point_cls = point_cls[n_scene:]
        print(f"  Adversarial point RPN scores: "
              f"mean={adv_point_cls.mean().item():.4f}, "
              f"max={adv_point_cls.max().item():.4f}")
        loss = -adv_point_cls.mean()
        print(f"  RPN loss: {loss.item():.4f}")
    else:
        print("\n  ✗ FAILED: No valid loss target found.")
        wrapper.remove_hook()
        return

    loss.backward()

    # Check gradients on delta_v
    if delta_v.grad is not None:
        grad_norm = delta_v.grad.norm().item()
        grad_std = delta_v.grad.std().item()
        grad_max = delta_v.grad.abs().max().item()
        n_nonzero = (delta_v.grad.abs() > 1e-10).sum().item()

        print(f"\n  delta_v.grad.norm()  = {grad_norm:.6e}")
        print(f"  delta_v.grad.std()   = {grad_std:.6e}")
        print(f"  delta_v.grad.max()   = {grad_max:.6e}")
        print(f"  Non-zero gradients:    {n_nonzero}/{delta_v.grad.numel()}")

        level = "RCNN (STE)" if use_rcnn else "RPN"
        if grad_norm > 1e-6:
            print(f"\n  ✓ PASSED: Gradients flow from {level} loss to delta_v.")
        else:
            print(f"\n  ✗ FAILED: Gradient norm too small ({level}).")

        if grad_std > 1e-8:
            print("  ✓ Gradient is non-uniform across vertices (good).")
        else:
            print("  ⚠ Gradient appears uniform — may indicate a problem.")
    else:
        print("\n  ✗ FAILED: delta_v.grad is None.")

    if t_tilde.grad is not None:
        print(f"\n  t_tilde.grad = {t_tilde.grad.cpu().numpy()}")
        print(f"  t_tilde.grad.norm() = {t_tilde.grad.norm().item():.6e}")

    wrapper.remove_hook()
    print("\nDone.")


if __name__ == '__main__':
    main()
