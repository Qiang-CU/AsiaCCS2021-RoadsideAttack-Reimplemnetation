"""
Precompute reference car features from KITTI train split.

For each car instance in KITTI train:
- Crop the point cloud to the GT bounding box (with 0.2m margin)
- Run a forward pass with that crop as a "fake scene"
- Capture the hook feature from PointRCNN's RCNN head
Average over N instances to get ref_feature (shape: D,).
Save to results/ref_car_feature.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset, get_bbox_lidar, count_points_in_bbox


def crop_points_in_bbox(pc, bbox_lidar, margin=0.2):
    """
    Crop points inside a 3D bbox with margin.

    Args:
        pc:          (N, 4) point cloud
        bbox_lidar:  (7,) [cx, cy, cz, l, w, h, yaw]
        margin:      extra padding in meters

    Returns:
        cropped: (M, 4) points inside the bbox
    """
    cx, cy, cz, l, w, h, yaw = bbox_lidar
    pts = pc[:, :3]
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy
    local_z = pts[:, 2] - cz

    mask = (
        (np.abs(local_x) < l / 2 + margin) &
        (np.abs(local_y) < w / 2 + margin) &
        (np.abs(local_z) < h / 2 + margin)
    )
    return pc[mask]


def main():
    parser = argparse.ArgumentParser(description='Precompute reference car features')
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--n_instances', type=int, default=500)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='results/ref_car_feature.pt')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load PointRCNN
    print("Loading PointRCNN...")
    wrapper = PointRCNNWrapper(
        config_path=config['model']['pointrcnn_config'],
        ckpt_path=config['model']['pointrcnn_ckpt'],
        device=args.device,
    )

    # Load KITTI train split
    print("Loading KITTI train split...")
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split='train',
        pc_range=config['data']['pc_range'],
        filter_objs=True,
    )

    features = []
    n_collected = 0

    print(f"Collecting features from {args.n_instances} car instances...")
    pbar = tqdm(total=args.n_instances, desc='Collecting car features')

    for idx in range(len(dataset)):
        if n_collected >= args.n_instances:
            break

        sample = dataset[idx]
        pc = sample['pointcloud']
        gt_boxes = sample['gt_bboxes']

        for bi, bbox in enumerate(gt_boxes):
            if n_collected >= args.n_instances:
                break

            # Crop points around this car
            cropped = crop_points_in_bbox(pc, bbox, margin=0.2)
            if len(cropped) < 10:
                continue

            # Run forward pass to capture RPN backbone features
            pc_t = torch.from_numpy(cropped.astype(np.float32)).to(args.device)
            try:
                result = wrapper.forward_with_grad(pc_t)
                # Use RPN backbone point_features (128-d), NOT RCNN hook (256-d).
                # This must match the features used in L_feat during attack.
                feat = result.get('point_features')
                if feat is not None and feat.numel() > 0:
                    feat_mean = feat.detach().mean(dim=0).cpu()
                    if torch.isfinite(feat_mean).all():
                        features.append(feat_mean)
                        n_collected += 1
                        pbar.update(1)
            except Exception as e:
                continue

    pbar.close()

    if not features:
        print("ERROR: No features collected. Check model and data paths.")
        sys.exit(1)

    # Average all features
    ref_feature = torch.stack(features).mean(dim=0)
    print(f"\nReference feature shape: {ref_feature.shape}")
    print(f"Feature norm: {ref_feature.norm().item():.4f}")
    print(f"Feature range: [{ref_feature.min().item():.4f}, {ref_feature.max().item():.4f}]")

    # Sanity check: cosine similarity between random pairs
    if len(features) >= 10:
        sims = []
        for i in range(min(50, len(features))):
            for j in range(i+1, min(50, len(features))):
                cos_sim = torch.nn.functional.cosine_similarity(
                    features[i].unsqueeze(0), features[j].unsqueeze(0)
                ).item()
                sims.append(cos_sim)
        print(f"Cosine similarity between car instances: "
              f"mean={np.mean(sims):.3f}, std={np.std(sims):.3f}")

    torch.save(ref_feature, args.output)
    print(f"Saved reference feature → {args.output}")


if __name__ == '__main__':
    main()
