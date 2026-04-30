"""
Precompute reference car features for appearing attack (AsiaCCS 2021).

Runs natural PointRCNN inference on synthetic car + ground scenes,
collects features from proposals that overlap the car position:
1. Backbone point features → ref_car_feature_backbone.pt
2. RCNN penultimate features (from NMS proposals) → ref_car_feature_rcnn.pt
3. RPN orientation → ref_car_orientations.pt
4. RPN box size → ref_car_box_size.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pointrcnn_wrapper import PointRCNNWrapper


def generate_car_box_points(center, lwh=(3.9, 1.6, 1.56), n_points=500,
                            yaw=0.0):
    """Generate points on a car-shaped box surface."""
    cx, cy, cz = center
    l, w, h = lwh
    points = []
    rng = np.random.RandomState(0)
    pts_per_face = n_points // 6

    for face in range(6):
        if face == 0:
            u = rng.uniform(-w/2, w/2, pts_per_face)
            v = rng.uniform(-h/2, h/2, pts_per_face)
            x = np.full(pts_per_face, l/2)
            y, z = u, v
        elif face == 1:
            u = rng.uniform(-w/2, w/2, pts_per_face)
            v = rng.uniform(-h/2, h/2, pts_per_face)
            x = np.full(pts_per_face, -l/2)
            y, z = u, v
        elif face == 2:
            u = rng.uniform(-l/2, l/2, pts_per_face)
            v = rng.uniform(-h/2, h/2, pts_per_face)
            x, z = u, v
            y = np.full(pts_per_face, w/2)
        elif face == 3:
            u = rng.uniform(-l/2, l/2, pts_per_face)
            v = rng.uniform(-h/2, h/2, pts_per_face)
            x, z = u, v
            y = np.full(pts_per_face, -w/2)
        elif face == 4:
            u = rng.uniform(-l/2, l/2, pts_per_face)
            v = rng.uniform(-w/2, w/2, pts_per_face)
            x, y = u, v
            z = np.full(pts_per_face, h/2)
        else:
            u = rng.uniform(-l/2, l/2, pts_per_face)
            v = rng.uniform(-w/2, w/2, pts_per_face)
            x, y = u, v
            z = np.full(pts_per_face, -h/2)

        pts = np.stack([x, y, z], axis=1)
        points.append(pts)

    pts_local = np.concatenate(points, axis=0)
    if abs(yaw) > 1e-6:
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y, 0],
                        [sin_y, cos_y, 0],
                        [0, 0, 1]])
        pts_local = pts_local @ rot.T

    pts_local[:, 0] += cx
    pts_local[:, 1] += cy
    pts_local[:, 2] += cz
    return pts_local.astype(np.float32)


def generate_ground_plane(n_points=10000, ground_z=-0.75):
    x = np.random.exponential(scale=10.0, size=n_points * 3)
    x = x[(x >= 0.0) & (x <= 40.0)]
    if len(x) > n_points:
        x = x[:n_points]
    elif len(x) < n_points:
        x = np.concatenate([x, np.random.uniform(0, 40, n_points - len(x))])
    y = np.random.uniform(-20, 20, len(x))
    z = np.full(len(x), ground_z) + np.random.normal(0, 0.02, len(x))
    intensity = np.random.uniform(0, 0.5, len(x))
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Precompute reference car features (backbone + RCNN)')
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_views', type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('results', exist_ok=True)

    atk = config['attack']
    object_pos = np.array(atk['object_pos'], dtype=np.float32)
    ps_cfg = config.get('pose_sweep', {})
    ground_z = ps_cfg.get('ground_z', -0.75)

    wrapper = PointRCNNWrapper(
        config_path=config['model']['pointrcnn_config'],
        ckpt_path=config['model']['pointrcnn_ckpt'],
        device=args.device,
        enable_ste=False,
    )

    rng = np.random.RandomState(42)
    backbone_feat_list = []
    rcnn_feat_list = []
    rpn_orient_list = []
    rpn_box_size_list = []

    print(f"Collecting features from {args.n_views} viewpoints...")
    print(f"  Object at: {object_pos.tolist()}")

    for view_i in range(args.n_views):
        lidar_x = rng.uniform(*atk['lidar_x_range'])
        lidar_y = rng.uniform(*atk['lidar_y_range'])
        lidar_z = rng.uniform(*atk['lidar_z_range'])
        lidar_pos = np.array([lidar_x, lidar_y, lidar_z], dtype=np.float32)
        rel_pos = object_pos - lidar_pos

        car_pts = generate_car_box_points(rel_pos, n_points=500)
        car_pts_4 = np.column_stack([car_pts,
                                     np.ones(len(car_pts), dtype=np.float32)])
        ground = generate_ground_plane(ground_z=ground_z)
        n_ground = len(ground)
        scene = np.concatenate([ground, car_pts_4], axis=0)
        scene_t = torch.from_numpy(scene).to(args.device)

        try:
            with torch.no_grad():
                result = wrapper.forward_with_grad(scene_t)

            # Backbone features for car points
            pf = result.get('point_features')
            if pf is not None and pf.shape[0] > n_ground:
                car_feats = pf[n_ground:].cpu()
                backbone_feat_list.append(car_feats.mean(dim=0))

            # RCNN features: find ROIs that overlap the car position
            rois = result.get('rois')
            rf = result.get('rcnn_features')
            if rois is not None and rf is not None:
                roi_boxes = rois[0]  # (m, 7)
                roi_xy = roi_boxes[:, :2]
                car_xy = torch.tensor(rel_pos[:2], device=roi_xy.device)
                dists = torch.norm(roi_xy - car_xy.unsqueeze(0), dim=1)
                near_mask = dists < 3.0
                if near_mask.any():
                    near_feats = rf[near_mask].cpu()
                    rcnn_feat_list.append(near_feats.mean(dim=0))

            # RPN box predictions for car points
            rpn_box = result.get('rpn_box_preds')
            if rpn_box is not None and rpn_box.shape[0] > n_ground:
                car_rpn = rpn_box[n_ground:].cpu()
                rpn_orient_list.append(float(car_rpn[:, 6].mean()))
                rpn_box_size_list.append(car_rpn[:, 3:6].mean(dim=0))

        except Exception as e:
            print(f"  View {view_i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (view_i + 1) % 10 == 0:
            print(f"  Processed {view_i+1}/{args.n_views}, "
                  f"backbone={len(backbone_feat_list)}, "
                  f"rcnn={len(rcnn_feat_list)}")

    if not backbone_feat_list:
        print("ERROR: No features collected.")
        sys.exit(1)

    # Save backbone features
    ref_feat = torch.stack(backbone_feat_list).mean(dim=0)
    print(f"\nBackbone ref feature: shape={ref_feat.shape}, "
          f"norm={ref_feat.norm():.4f}")
    torch.save(ref_feat, 'results/ref_car_feature_backbone.pt')
    print("Saved → results/ref_car_feature_backbone.pt")

    # Save RCNN features
    if rcnn_feat_list:
        ref_rcnn = torch.stack(rcnn_feat_list).mean(dim=0)
        print(f"RCNN ref feature: shape={ref_rcnn.shape}, "
              f"norm={ref_rcnn.norm():.4f}")
        torch.save(ref_rcnn, 'results/ref_car_feature_rcnn.pt')
        print("Saved → results/ref_car_feature_rcnn.pt")
    else:
        print("WARNING: No RCNN features collected (hook may not have fired)")

    # Save RPN orientation
    rpn_orient = float(np.mean(rpn_orient_list)) if rpn_orient_list else 0.0
    orient_dict = {'rpn_orientation': rpn_orient}
    torch.save(orient_dict, 'results/ref_car_orientations.pt')
    print(f"RPN orientation: {rpn_orient:.4f}")
    print("Saved → results/ref_car_orientations.pt")

    # Save RPN box size
    if rpn_box_size_list:
        ref_box_size = torch.stack(rpn_box_size_list).mean(dim=0)
        torch.save(ref_box_size, 'results/ref_car_box_size.pt')
        print(f"RPN box size (dx,dy,dz): {ref_box_size.tolist()}")
        print("Saved → results/ref_car_box_size.pt")

    wrapper.remove_hook()
    print("\nDone.")


if __name__ == '__main__':
    main()
