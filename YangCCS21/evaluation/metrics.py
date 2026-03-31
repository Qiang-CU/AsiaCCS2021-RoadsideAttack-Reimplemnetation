"""
Evaluation metrics for the appearing adversarial attack (AsiaCCS 2021).

- Attack Success Rate (ASR): phantom car detected at injection position
- Recall-IoU curve for appearing attack
- Defense evaluation (kNN, Gaussian noise, density SVM)

All evaluation functions support multi-GPU parallel evaluation via
``devices`` / ``wrappers`` parameters.  When omitted, they fall back
to single-device behaviour for full backward compatibility.
"""

import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.bev_iou import bev_iou_matrix, bev_iou
from attack.inject import build_bev_occupancy, sample_injection_position
from attack.loss import compute_phantom_box


# ---------------------------------------------------------------------------
# Single-sample helpers
# ---------------------------------------------------------------------------

def has_detection_near_pos(pred_bboxes, pred_scores, injection_pos,
                          proximity_thresh=1.5, score_thresh=0.3):
    """
    True if any predicted box is near the injection position with sufficient
    confidence.

    Args:
        pred_bboxes:     (D, 7) numpy array
        pred_scores:     (D,)   numpy array
        injection_pos:   (3,)   numpy array
        proximity_thresh: float, max distance in meters
        score_thresh:    float, min confidence
    """
    if len(pred_bboxes) == 0:
        return False
    dists = np.sqrt((pred_bboxes[:, 0] - injection_pos[0])**2 +
                    (pred_bboxes[:, 1] - injection_pos[1])**2)
    return bool(((dists < proximity_thresh) & (pred_scores >= score_thresh)).any())


def is_detected(pred_bboxes, gt_bbox, iou_thresh=0.7):
    """
    True if any predicted box has BEV IoU >= iou_thresh with gt_bbox.

    Args:
        pred_bboxes: (D, 7) numpy or empty array
        gt_bbox:     (7,)   numpy
        iou_thresh:  float
    """
    if len(pred_bboxes) == 0:
        return False
    ious = bev_iou_matrix(pred_bboxes, gt_bbox.reshape(1, 7))[:, 0]
    return bool(ious.max() >= iou_thresh)


# ---------------------------------------------------------------------------
# ASR for appearing attack (multi-GPU)
# ---------------------------------------------------------------------------

def compute_appearing_asr(model_wrapper, dataset, adv_mesh_ckpt, config,
                          device='cuda:0',
                          devices=None, wrappers=None):
    """
    Compute Attack Success Rate for the appearing attack.

    A frame is attack-success if:
    1. Before attack: no detection near injection position
    2. After attack:  at least one detection near injection position (conf > thresh)

    Args:
        model_wrapper:  PointRCNNWrapper instance (used when wrappers is None,
                        or as the single-GPU wrapper)
        dataset:        KITTIDataset (val split)
        adv_mesh_ckpt:  path to adv_mesh checkpoint
        config:         attack config dict
        device:         torch device (single-GPU fallback)
        devices:        list[torch.device] for multi-GPU; None = single GPU
        wrappers:       dict[torch.device, PointRCNNWrapper]; None = use model_wrapper

    Returns:
        asr:   float in [0, 1]
        stats: dict with detailed results
    """
    from attack.whitebox import apply_attack_to_sample

    if devices is None:
        devices = [torch.device(device)]
    if wrappers is None:
        wrappers = {devices[0]: model_wrapper}
    n_gpus = len(devices)

    ckpt = torch.load(adv_mesh_ckpt, map_location='cpu', weights_only=False)
    delta_v = ckpt['delta_v']
    t_tilde = ckpt['t_tilde']
    v0      = ckpt['v0']
    faces   = ckpt['faces']

    atk = config['attack']
    inj_cfg = atk['injection']
    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']

    def _eval_one_asr(idx):
        dev = devices[idx % n_gpus]
        wrapper = wrappers[dev]
        if dev.type == 'cuda':
            torch.cuda.set_device(dev)

        rng = np.random.RandomState(42 + idx)
        sample = dataset[idx]
        pc_np  = sample['pointcloud']
        gt_boxes = sample['gt_bboxes']

        occ, grid_info = build_bev_occupancy(
            pc_np, gt_boxes,
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )
        inj_pos, valid = sample_injection_position(
            occ, grid_info,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
            rng=rng,
        )

        pc_t = torch.from_numpy(pc_np.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_clean, scores_clean = wrapper.detect(pc_t, score_thresh)

        clean_has_det = has_detection_near_pos(
            pred_clean, scores_clean, inj_pos,
            proximity_thresh=proximity, score_thresh=score_thresh
        )

        if clean_has_det:
            return {
                'sample_id': sample['sample_id'],
                'eligible': False,
                'success': False,
                'injection_pos': inj_pos.tolist(),
            }

        pc_adv, n_adv = apply_attack_to_sample(
            sample, delta_v, t_tilde, v0, faces, config, str(dev),
            injection_pos=inj_pos,
        )
        pc_adv_t = torch.from_numpy(pc_adv.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_adv, scores_adv = wrapper.detect(pc_adv_t, score_thresh)

        adv_has_det = has_detection_near_pos(
            pred_adv, scores_adv, inj_pos,
            proximity_thresh=proximity, score_thresh=score_thresh
        )

        return {
            'sample_id': sample['sample_id'],
            'eligible': True,
            'success': adv_has_det,
            'n_adv_pts': n_adv,
            'injection_pos': inj_pos.tolist(),
        }

    n_frames = len(dataset)
    results = [None] * n_frames

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_eval_one_asr, idx): idx
                   for idx in range(n_frames)}
        for future in tqdm(
            futures, total=n_frames, desc='Evaluating Appearing ASR'
        ):
            idx = futures[future]
            results[idx] = future.result()

    n_eligible = sum(1 for r in results if r['eligible'])
    n_success  = sum(1 for r in results if r.get('success', False) and r['eligible'])

    asr = n_success / max(n_eligible, 1)
    print(f'\nAppearing ASR: {n_success}/{n_eligible} = {asr*100:.1f}%')

    return asr, {
        'n_eligible': n_eligible,
        'n_success': n_success,
        'per_sample': results,
    }


# ---------------------------------------------------------------------------
# Recall-IoU curve (appearing attack variant, multi-GPU)
# ---------------------------------------------------------------------------

def compute_recall_iou_curve(model_wrapper, dataset, adv_mesh_ckpt, config,
                              device='cuda:0',
                              iou_thresholds=None,
                              devices=None, wrappers=None):
    """
    For a range of IoU thresholds, compute fraction of injected frames where
    the model produces a detection with IoU > threshold with the phantom box.

    Args:
        model_wrapper:  PointRCNNWrapper instance (single-GPU fallback)
        dataset:        KITTIDataset
        adv_mesh_ckpt:  path to checkpoint
        config:         attack config dict
        device:         torch device (single-GPU fallback)
        iou_thresholds: array of IoU thresholds
        devices:        list[torch.device] for multi-GPU; None = single GPU
        wrappers:       dict[torch.device, PointRCNNWrapper]; None = use model_wrapper

    Returns:
        iou_thresholds: (T,) array
        recall_adv:     (T,) array
    """
    from attack.whitebox import apply_attack_to_sample

    if devices is None:
        devices = [torch.device(device)]
    if wrappers is None:
        wrappers = {devices[0]: model_wrapper}
    n_gpus = len(devices)

    ckpt = torch.load(adv_mesh_ckpt, map_location='cpu', weights_only=False)
    delta_v = ckpt['delta_v']
    t_tilde = ckpt['t_tilde']
    v0      = ckpt['v0']
    faces   = ckpt['faces']

    atk = config['attack']
    inj_cfg = atk['injection']
    score_thresh = config['model']['score_thresh']
    phantom_size = tuple(atk['phantom_box_size'])
    phantom_yaw  = atk.get('phantom_box_yaw', 0.0)

    if iou_thresholds is None:
        iou_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def _eval_one_recall(idx):
        dev = devices[idx % n_gpus]
        wrapper = wrappers[dev]
        if dev.type == 'cuda':
            torch.cuda.set_device(dev)

        rng = np.random.RandomState(42 + idx)
        sample = dataset[idx]
        pc_np  = sample['pointcloud']
        gt_boxes = sample['gt_bboxes']

        occ, grid_info = build_bev_occupancy(
            pc_np, gt_boxes,
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )
        inj_pos, _ = sample_injection_position(
            occ, grid_info,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
            rng=rng,
        )

        phantom_box = compute_phantom_box(inj_pos, phantom_size, phantom_yaw)

        pc_adv, _ = apply_attack_to_sample(
            sample, delta_v, t_tilde, v0, faces, config, str(dev),
            injection_pos=inj_pos,
        )
        pc_adv_t = torch.from_numpy(pc_adv.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_boxes, pred_scores = wrapper.detect(pc_adv_t, score_thresh)

        if len(pred_boxes) == 0:
            return 0.0
        ious = bev_iou_matrix(pred_boxes, phantom_box.reshape(1, 7))[:, 0]
        return float(ious.max())

    n_frames = len(dataset)
    best_ious = [0.0] * n_frames

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_eval_one_recall, idx): idx
                   for idx in range(n_frames)}
        for future in tqdm(
            futures, total=n_frames, desc='Recall-IoU', leave=False
        ):
            idx = futures[future]
            best_ious[idx] = future.result()

    best_ious = np.array(best_ious)
    recall = np.array([(best_ious >= t).mean() for t in iou_thresholds])

    return iou_thresholds, recall


# ---------------------------------------------------------------------------
# Defense evaluation
# ---------------------------------------------------------------------------

def knn_outlier_removal(points, k=5, alpha=0.1):
    """
    Remove points where mean kNN distance > mu + alpha * sigma.

    Args:
        points: (N, 3+) numpy array
        k:      number of neighbours
        alpha:  threshold multiplier

    Returns:
        filtered: (M, 3+) numpy array with outliers removed
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :3])
    dists, _ = tree.query(points[:, :3], k=k+1)  # +1 for self
    mean_dists = dists[:, 1:].mean(axis=1)  # skip self
    mu = mean_dists.mean()
    sigma = mean_dists.std()
    mask = mean_dists <= mu + alpha * sigma
    return points[mask]


def gaussian_noise_defense(points, sigma2=0.01):
    """
    Add Gaussian noise to all point coordinates.

    Args:
        points: (N, 3+) numpy array
        sigma2: variance

    Returns:
        noisy: (N, 3+) numpy array
    """
    noisy = points.copy()
    noise = np.random.normal(0, np.sqrt(sigma2), size=points[:, :3].shape)
    noisy[:, :3] += noise.astype(np.float32)
    return noisy


def density_defense_features(pred_boxes, points, lidar_pos=None, r=0.35):
    """
    Compute density features for each predicted bounding box.

    For each box: find the point nearest to LiDAR within the box,
    count points within sphere of radius r around that point.

    Args:
        pred_boxes: (D, 7) numpy array
        points:     (N, 3+) numpy array
        lidar_pos:  (3,) or None (default: origin)
        r:          sphere radius in meters

    Returns:
        counts: (D,) array of point counts (features for SVM)
    """
    if lidar_pos is None:
        lidar_pos = np.zeros(3, dtype=np.float32)

    pts = points[:, :3]
    counts = np.zeros(len(pred_boxes), dtype=np.float32)

    for bi, box in enumerate(pred_boxes):
        cx, cy, cz, l, w, h, yaw = box
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        local_x = cos_y * dx - sin_y * dy
        local_y = sin_y * dx + cos_y * dy
        local_z = pts[:, 2] - cz

        inside = ((np.abs(local_x) < l / 2) &
                  (np.abs(local_y) < w / 2) &
                  (np.abs(local_z) < h / 2))

        if not inside.any():
            counts[bi] = 0
            continue

        box_pts = pts[inside]
        # Nearest point to LiDAR
        dists_to_lidar = np.linalg.norm(box_pts - lidar_pos, axis=1)
        nearest_idx = dists_to_lidar.argmin()
        nearest_pt = box_pts[nearest_idx]

        # Count points within sphere
        sphere_dists = np.linalg.norm(pts - nearest_pt, axis=1)
        counts[bi] = float((sphere_dists <= r).sum())

    return counts
