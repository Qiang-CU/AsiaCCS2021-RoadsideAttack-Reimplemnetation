# -*- coding: utf-8 -*-
"""
Debug script: trace exactly where the STE gradient chain breaks.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from model.pointrcnn_wrapper import PointRCNNWrapper, _RoIPointPool3dSTE
from utils.kitti_utils import KITTIDataset

torch.cuda.set_device(0)
device = 'cuda:0'


def main():
    with open('configs/attack_config.yaml') as f:
        config = yaml.safe_load(f)

    print("Loading PointRCNN with STE...")
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        device=device,
        enable_ste=True,
    )

    import types
    is_patched = isinstance(wrapper.model.roi_head.roipool3d_gpu, types.MethodType)
    print("STE patch applied:", is_patched)

    dataset = KITTIDataset(config['data']['kitti_root'], split='val',
                           pc_range=config['data']['pc_range'])

    sample = None
    for idx in range(20):
        s = dataset[idx]
        if len(s['gt_bboxes']) > 0:
            sample = s
            break

    pc_np = sample['pointcloud'][:, :4].astype(np.float32)
    pc = torch.tensor(pc_np, dtype=torch.float32, device=device)

    # == TEST 1: RPN gradient (baseline, should work) ==
    print("\n" + "=" * 60)
    print("TEST 1: RPN gradient (should work)")
    print("=" * 60)

    pc1 = pc.detach().requires_grad_(True)
    batch1 = wrapper.build_batch_dict(pc1)
    for k in batch1:
        if isinstance(batch1[k], torch.Tensor):
            batch1[k] = batch1[k].to(device)

    with torch.set_grad_enabled(True):
        for cur_module in wrapper.model.module_list:
            mod_name = cur_module.__class__.__name__
            batch1 = cur_module(batch1)
            if mod_name == 'PointHeadBox':
                break

    pt_cls = batch1.get('point_cls_scores')
    print("point_cls_scores requires_grad:", pt_cls.requires_grad)
    loss1 = pt_cls.sum()
    loss1.backward()
    g1 = pc1.grad
    if g1 is not None:
        print("pc.grad norm (RPN path): %.6e" % g1.norm().item())
    else:
        print("pc.grad is None (RPN path)")

    # == TEST 2: Isolated STE autograd test ==
    print("\n" + "=" * 60)
    print("TEST 2: Isolated STE autograd Function test")
    print("=" * 60)

    B, N, C, M, S = 1, 100, 10, 2, 16
    pts = torch.randn(B, N, 3, device=device)
    feats = torch.randn(B, N, C, device=device, requires_grad=True)
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0],
                           [1.0, 1.0, 0.0, 5.0, 5.0, 5.0, 0.0]]],
                         device=device)
    pool_width = 1.0
    try:
        pooled, empty = _RoIPointPool3dSTE.apply(pts, feats, boxes, pool_width, S)
        print("Pooled shape:", pooled.shape)
        print("Pooled requires_grad:", pooled.requires_grad)
        print("Pooled grad_fn:", pooled.grad_fn)
        loss2 = pooled.sum()
        loss2.backward()
        if feats.grad is not None:
            print("feats.grad norm: %.6e" % feats.grad.norm().item())
        else:
            print("feats.grad is None -- STE backward FAILED")
    except Exception as e:
        print("STE test FAILED with error:", e)
        import traceback
        traceback.print_exc()

    # == TEST 3: Full pipeline RCNN gradient ==
    print("\n" + "=" * 60)
    print("TEST 3: Full pipeline RCNN gradient")
    print("=" * 60)

    pc3 = pc.detach().requires_grad_(True)
    batch3 = wrapper.build_batch_dict(pc3)
    for k in batch3:
        if isinstance(batch3[k], torch.Tensor):
            batch3[k] = batch3[k].to(device)

    # Track intermediate states
    point_feats_after_rpn = None

    with torch.set_grad_enabled(True):
        for cur_module in wrapper.model.module_list:
            mod_name = cur_module.__class__.__name__
            batch3 = cur_module(batch3)

            if mod_name == 'PointHeadBox':
                point_feats_after_rpn = batch3.get('point_features')
                print("After PointHeadBox:")
                print("  point_features requires_grad:", point_feats_after_rpn.requires_grad)

            if mod_name == 'PointRCNNHead':
                cls = batch3.get('batch_cls_preds')
                print("After PointRCNNHead:")
                print("  batch_cls_preds requires_grad:", cls.requires_grad)
                print("  batch_cls_preds grad_fn:", cls.grad_fn)

    cls_preds = batch3['batch_cls_preds']

    # Walk grad_fn chain
    print("\nGrad_fn chain from batch_cls_preds:")
    gf = cls_preds.grad_fn
    depth = 0
    while gf is not None and depth < 25:
        print("  [%d] %s" % (depth, type(gf).__name__))
        if hasattr(gf, 'next_functions') and len(gf.next_functions) > 0:
            gf = gf.next_functions[0][0]
        else:
            break
        depth += 1

    # Backward
    loss3 = cls_preds.sum()
    loss3.backward()

    if pc3.grad is not None:
        print("\npc.grad norm (RCNN+STE path): %.6e" % pc3.grad.norm().item())
    else:
        print("\npc.grad is None (RCNN+STE path)")

    # Also check intermediate
    if point_feats_after_rpn is not None and point_feats_after_rpn.grad is not None:
        print("point_features.grad norm: %.6e" % point_feats_after_rpn.grad.norm().item())
    else:
        print("point_features.grad is None (gradient stopped before STE)")


    # == TEST 4: Full chain from delta_v through mesh render to RCNN ==
    print("\n" + "=" * 60)
    print("TEST 4: Full chain delta_v -> render -> inject -> RCNN")
    print("=" * 60)

    from attack.mesh import create_icosphere
    from attack.renderer import render_adversarial_points
    from attack.reparameterize import reparameterize
    from attack.inject import (
        build_bev_occupancy, sample_injection_position, inject_points,
    )

    atk = config['attack']
    inj_cfg = atk['injection']
    lidar_cfg = atk.get('lidar', {})

    v0, faces, adj = create_icosphere(subdivisions=atk['mesh_subdivisions'])
    v0 = v0.to(device)
    faces = faces.to(device)

    b_param = torch.tensor(atk['size_limit'], dtype=torch.float32, device=device)
    c_param = torch.tensor(atk['translation_limit'], dtype=torch.float32, device=device)

    delta_v = torch.zeros_like(v0, requires_grad=True)
    t_tilde = torch.zeros(3, device=device, requires_grad=True)
    R = torch.eye(3, device=device)
    sensor_pos = torch.zeros(3, device=device)

    # Get injection position
    occ, grid_info = build_bev_occupancy(
        sample['pointcloud'], sample['gt_bboxes'],
        x_range=tuple(inj_cfg['x_range']),
        y_range=tuple(inj_cfg['y_range']),
        resolution=inj_cfg['resolution'],
        margin=inj_cfg['margin'],
    )
    inj_pos, valid = sample_injection_position(
        occ, grid_info,
        min_clearance=inj_cfg['min_clearance'],
        fallback_pos=tuple(inj_cfg['fallback_pos']),
    )
    print("Injection pos:", inj_pos, "valid:", valid)

    # Reparameterize
    vertices = reparameterize(v0, delta_v, t_tilde, R, b_param, c_param)
    print("vertices requires_grad:", vertices.requires_grad)

    inj_t = torch.tensor(inj_pos, dtype=torch.float32, device=device)
    vertices_world = vertices + inj_t.unsqueeze(0)

    # Render
    adv_pts = render_adversarial_points(
        vertices_world, faces, sensor_pos,
        n_elevation=lidar_cfg.get('n_elevation', 64),
        elev_min_deg=lidar_cfg.get('elev_min_deg', -24.9),
        elev_max_deg=lidar_cfg.get('elev_max_deg', 2.0),
        h_step_deg=lidar_cfg.get('h_step_deg', 0.08),
        margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
    )
    print("Rendered %d adversarial points" % adv_pts.shape[0])
    print("adv_pts requires_grad:", adv_pts.requires_grad)
    print("adv_pts grad_fn:", adv_pts.grad_fn)

    if adv_pts.shape[0] == 0:
        print("ERROR: No adv points rendered")
        return

    # Inject
    pc_tensor = torch.tensor(
        sample['pointcloud'], dtype=torch.float32, device=device
    )
    merged_pc, n_adv = inject_points(
        pc_tensor, adv_pts, torch.zeros(3, device=device),
        remove_overlap=True,
    )
    print("merged_pc requires_grad:", merged_pc.requires_grad)
    print("merged_pc grad_fn:", merged_pc.grad_fn)

    # Forward through PointRCNN
    batch4 = wrapper.build_batch_dict(merged_pc)
    for k in batch4:
        if isinstance(batch4[k], torch.Tensor):
            batch4[k] = batch4[k].to(device)

    with torch.set_grad_enabled(True):
        for cur_module in wrapper.model.module_list:
            batch4 = cur_module(batch4)

    cls4 = batch4['batch_cls_preds']
    print("\nbatch_cls_preds requires_grad:", cls4.requires_grad)

    loss4 = cls4.sum()
    loss4.backward()

    if delta_v.grad is not None:
        print("delta_v.grad norm: %.6e" % delta_v.grad.norm().item())
        print("*** SUCCESS: Full gradient chain works! ***")
    else:
        print("delta_v.grad is None")
        print("Checking intermediate grads...")
        # adv_pts was created from vertices_world which depends on delta_v
        # If adv_pts.grad_fn exists, gradient should flow
        # The issue might be in inject_points
        print("  adv_pts grad_fn:", adv_pts.grad_fn)

        # Check if merged_pc has grad
        if merged_pc.grad is not None:
            print("  merged_pc.grad norm: %.6e" % merged_pc.grad.norm().item())
        else:
            print("  merged_pc.grad is None (need .retain_grad() to check)")

    if t_tilde.grad is not None:
        print("t_tilde.grad norm: %.6e" % t_tilde.grad.norm().item())
    else:
        print("t_tilde.grad is None")


if __name__ == '__main__':
    main()
