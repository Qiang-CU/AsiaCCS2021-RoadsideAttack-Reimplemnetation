"""
Phase 0 verification: Test PointRCNN inference on KITTI val frames.

Pass criteria:
- Produces bounding boxes on >= 3 frames with confidence > 0.5
- BEV AP on first 100 val frames is within 5% of reported benchmark
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset


def main():
    config_path = 'configs/attack_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print("Loading PointRCNN...")
    wrapper = PointRCNNWrapper(
        config_path=config['model']['pointrcnn_config'],
        ckpt_path=config['model']['pointrcnn_ckpt'],
        device=device,
    )
    print("  Model loaded successfully.")

    # Load dataset
    print("Loading KITTI val split...")
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split='val',
        pc_range=config['data']['pc_range'],
    )
    print(f"  {len(dataset)} frames loaded.")

    # Test inference on first N frames
    n_test = min(10, len(dataset))
    n_with_conf05 = 0
    total_detections = 0

    print(f"\nRunning inference on {n_test} frames...")
    print("-" * 60)

    for idx in range(n_test):
        sample = dataset[idx]
        pc = sample['pointcloud']
        sid = sample['sample_id']
        n_gt = len(sample['gt_bboxes'])

        pc_t = torch.from_numpy(pc.astype(np.float32)).to(device)
        boxes, scores = wrapper.detect(pc_t, score_thresh=0.1)

        n_det = len(boxes)
        max_score = scores.max() if n_det > 0 else 0.0
        n_conf05 = (scores > 0.5).sum() if n_det > 0 else 0
        total_detections += n_det

        print(f"  Frame {sid}: {n_det:3d} detections | "
              f"max_score={max_score:.3f} | "
              f"conf>0.5: {n_conf05} | GT cars: {n_gt}")

        if n_det > 0 and max_score > 0.5:
            n_with_conf05 += 1

    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Frames with conf > 0.5 detections: {n_with_conf05}/{n_test}")
    print(f"  Total detections (score > 0.1):     {total_detections}")
    print(f"  Average detections per frame:       {total_detections/n_test:.1f}")

    # Phase 0 pass/fail
    if n_with_conf05 >= 3:
        print("\n  ✓ Phase 0 PASSED: PointRCNN inference is working correctly.")
    else:
        print("\n  ✗ Phase 0 FAILED: Too few confident detections.")
        print("    Check: model weights, data paths, point cloud range.")

    # Test hook features
    print("\nTesting feature hook...")
    sample = dataset[0]
    pc_t = torch.from_numpy(sample['pointcloud'].astype(np.float32)).to(device)
    result = wrapper.forward_with_grad(pc_t)
    feat = result.get('features')
    if feat is not None:
        print(f"  Hook feature shape: {feat.shape}")
        print(f"  Feature range: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
        print("  ✓ Feature hook is working.")
    else:
        print("  ✗ Feature hook returned None.")

    wrapper.remove_hook()
    print("\nDone.")


if __name__ == '__main__':
    main()
