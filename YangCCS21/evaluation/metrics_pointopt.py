"""
Evaluation metrics for the direct point-optimisation appearing attack.

Mirrors evaluation/metrics.py but uses apply_pointopt_to_sample instead of
the mesh-based apply_attack_to_sample.
"""

import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.bev_iou import bev_iou_matrix
from attack.inject import build_bev_occupancy, sample_injection_position
from attack.loss import compute_phantom_box
from evaluation.metrics import has_detection_near_pos


def compute_pointopt_asr(dataset, adv_points_ckpt, config,
                         device='cuda:0',
                         devices=None, wrappers=None):
    """
    Compute ASR for the point-optimisation attack.

    Args:
        dataset:          KITTIDataset (val split)
        adv_points_ckpt:  path to adv_points_pointopt_*.pth
        config:           attack config dict
        device:           fallback device
        devices:          list[torch.device] for multi-GPU
        wrappers:         dict[torch.device, PointRCNNWrapper]

    Returns:
        asr, stats
    """
    from attack.whitebox_pointopt import apply_pointopt_to_sample

    if devices is None:
        devices = [torch.device(device)]
    if wrappers is None:
        from model.pointrcnn_wrapper import PointRCNNWrapper
        w = PointRCNNWrapper(
            config['model']['pointrcnn_config'],
            config['model']['pointrcnn_ckpt'],
            device=str(devices[0]),
            enable_ste=False,
        )
        wrappers = {devices[0]: w}
    n_gpus = len(devices)

    atk = config['attack']
    inj_cfg = atk['injection']
    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']

    def _eval_one(idx):
        dev = devices[idx % n_gpus]
        wrapper = wrappers[dev]
        if dev.type == 'cuda':
            torch.cuda.set_device(dev)

        rng = np.random.RandomState(42 + idx)
        sample = dataset[idx]
        pc_np = sample['pointcloud']
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

        pc_adv, n_adv = apply_pointopt_to_sample(
            sample, adv_points_ckpt, config, str(dev),
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
        futures = {executor.submit(_eval_one, idx): idx
                   for idx in range(n_frames)}
        for future in tqdm(
            futures, total=n_frames, desc='Evaluating PointOpt ASR'
        ):
            idx = futures[future]
            results[idx] = future.result()

    n_eligible = sum(1 for r in results if r['eligible'])
    n_success = sum(1 for r in results if r.get('success', False) and r['eligible'])

    asr = n_success / max(n_eligible, 1)
    print(f'\nPointOpt ASR: {n_success}/{n_eligible} = {asr*100:.1f}%')

    return asr, {
        'n_eligible': n_eligible,
        'n_success': n_success,
        'per_sample': results,
    }
