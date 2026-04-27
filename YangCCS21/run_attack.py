"""
Main entry point for the AsiaCCS 2021 appearing adversarial attack replication.

Usage:
    # Phase 0: Test PointRCNN inference
    python run_attack.py --mode test_inference --device cuda:0

    # Phase 3: Precompute reference car features
    python run_attack.py --mode precompute --device cuda:0

    # Phase 5: White-box attack
    python run_attack.py --mode whitebox --device cuda:0

    # Phase 6: Black-box attack (CMA-ES, multi-GPU)
    python run_attack.py --mode blackbox --gpu 0,1,2,3

    # Phase 7: Evaluate ASR (multi-GPU)
    python run_attack.py --mode eval --gpu 0,1,2,3 --ckpt results/adv_mesh_whitebox_final.pth

    # Phase 7: Recall-IoU curve (multi-GPU)
    python run_attack.py --mode recall_iou --gpu 0,1,2,3 --ckpt results/adv_mesh_whitebox_final.pth

    # Phase 7: Defense evaluation (multi-GPU)
    python run_attack.py --mode defenses --gpu 0,1,2,3 --ckpt results/adv_mesh_whitebox_final.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_save_dir(config):
    return config.get('output', {}).get('save_dir', 'results')


def get_wrapper(config, device, enable_ste=None):
    from model.pointrcnn_wrapper import PointRCNNWrapper
    if enable_ste is None:
        enable_ste = config['attack'].get('enable_ste', False)
    return PointRCNNWrapper(
        config_path=config['model']['pointrcnn_config'],
        ckpt_path=config['model']['pointrcnn_ckpt'],
        device=device,
        enable_ste=enable_ste,
    )


def build_multi_gpu_wrappers(config, devices):
    """Create a PointRCNNWrapper replica on each device.

    Returns:
        wrappers: dict mapping torch.device -> PointRCNNWrapper
    """
    from model.pointrcnn_wrapper import PointRCNNWrapper
    wrappers = {}
    for dev in devices:
        torch.cuda.set_device(dev)
        wrappers[dev] = PointRCNNWrapper(
            config['model']['pointrcnn_config'],
            config['model']['pointrcnn_ckpt'],
            str(dev),
        )
    print(f'  Model replicas loaded on {len(devices)} GPU(s): '
          f'{[str(d) for d in devices]}')
    return wrappers


def _parse_devices(args):
    """Parse --gpu / --device into a list of torch.device."""
    if args.gpu:
        gpu_ids = [int(g.strip()) for g in args.gpu.split(',')]
        devices = [torch.device(f'cuda:{g}') for g in gpu_ids]
        torch.cuda.set_device(gpu_ids[0])
    else:
        devices = [torch.device(args.device)]
    return devices


def get_dataset(config, split=None):
    from utils.kitti_utils import KITTIDataset
    if split is None:
        split = config['data']['split']
    return KITTIDataset(
        root=config['data']['kitti_root'],
        split=split,
        pc_range=config['data']['pc_range'],
    )


def mode_test_inference(config, device):
    """Phase 0: Test PointRCNN inference on a few KITTI val frames."""
    print("=" * 60)
    print("Phase 0: Testing PointRCNN inference")
    print("=" * 60)

    wrapper = get_wrapper(config, device)
    dataset = get_dataset(config)

    n_test = min(5, len(dataset))
    n_detected = 0

    for idx in range(n_test):
        sample = dataset[idx]
        pc = sample['pointcloud']
        sid = sample['sample_id']

        pc_t = torch.from_numpy(pc.astype(np.float32)).to(device)
        boxes, scores = wrapper.detect(pc_t, score_thresh=0.3)

        print(f"  Frame {sid}: {len(boxes)} detections "
              f"(max score: {scores.max():.3f})" if len(scores) > 0
              else f"  Frame {sid}: 0 detections")

        if len(scores) > 0 and scores.max() > 0.5:
            n_detected += 1

    print(f"\n  Frames with conf > 0.5 detections: {n_detected}/{n_test}")
    if n_detected >= 3:
        print("  ✓ Phase 0 PASSED")
    else:
        print("  ✗ Phase 0 FAILED — check model weights")


def mode_precompute(config, device):
    """Phase 3: Precompute reference car features."""
    print("=" * 60)
    print("Phase 3: Precomputing reference car features")
    print("=" * 60)

    # Delegate to precompute_features.py
    from precompute_features import main as precompute_main
    sys.argv = [
        'precompute_features.py',
        '--config', 'configs/attack_config.yaml',
        '--n_instances', str(config.get('ref_feature', {}).get('n_instances', 500)),
        '--device', device,
        '--output', config.get('ref_feature', {}).get('output_path', 'results/ref_car_feature.pt'),
    ]
    precompute_main()


def mode_whitebox(config, device, warm_start_ckpt=None):
    """Phase 5: White-box appearing attack (single-stage RPN optimization)."""
    print("=" * 60)
    print("Phase 5: White-box appearing attack (single-stage)")
    print(f"  Size limit: {config['attack']['size_limit']}")
    print(f"  Loss weights: {config['attack']['loss_weights']}")
    if warm_start_ckpt:
        print(f"  Warm-start: {warm_start_ckpt}")
    print("=" * 60)

    config['device'] = device
    save_dir = get_save_dir(config)

    from attack.whitebox import run_whitebox_attack

    dataset = get_dataset(config)

    mesh_param, translation_param, history = run_whitebox_attack(
        dataset, config, save_dir=save_dir,
        warm_start_ckpt=warm_start_ckpt,
    )

    from evaluation.visualize import plot_loss_curve
    loss_keys = {
        'L_total': history['L_total'],
        'L_cls': history['L_cls'],
        'L_loc': history['L_loc'],
    }
    plot_loss_curve(loss_keys, save_path=os.path.join(save_dir, 'whitebox_loss.png'))


def mode_pointopt(config, device, warm_start_ckpt=None, devices=None):
    """Direct point-cloud optimisation attack (supports multi-GPU)."""
    print("=" * 60)
    print("Phase 5b: Direct Point Optimisation attack")
    po_cfg = config['attack'].get('pointopt', {})
    print(f"  n_points: {po_cfg.get('n_points', 400)}")
    print(f"  init: {po_cfg.get('init', 'gt')}")
    print(f"  Loss weights: {config['attack']['loss_weights']}")
    if warm_start_ckpt:
        print(f"  Warm-start: {warm_start_ckpt}")

    if devices is None:
        devices = [torch.device(device)]
    n_gpus = len(devices)
    print(f"  GPUs: {[str(d) for d in devices]}")
    print("=" * 60)

    config['device'] = str(devices[0])
    save_dir = get_save_dir(config)

    from attack.whitebox_pointopt import run_pointopt_attack

    dataset = get_dataset(config)

    wrappers = None
    if n_gpus > 1:
        wrappers = build_multi_gpu_wrappers(config, devices)

    adv_points, history = run_pointopt_attack(
        dataset, config, save_dir=save_dir,
        warm_start_ckpt=warm_start_ckpt,
        devices=devices, wrappers=wrappers,
    )

    from evaluation.visualize import plot_loss_curve
    loss_keys = {
        'L_total': history['L_total'],
        'L_cls': history['L_cls'],
        'L_loc': history['L_loc'],
    }
    plot_loss_curve(
        loss_keys, save_path=os.path.join(save_dir, 'whitebox_pointopt_loss.png')
    )


def mode_eval_pointopt(config, devices, ckpt_path):
    """Evaluate ASR for point-opt checkpoint."""
    print("=" * 60)
    print("Phase 7: Evaluating Point-Opt ASR")
    print("=" * 60)

    from evaluation.metrics_pointopt import compute_pointopt_asr

    wrappers = build_multi_gpu_wrappers(config, devices)
    dataset = get_dataset(config)
    save_dir = get_save_dir(config)

    asr, stats = compute_pointopt_asr(
        dataset=dataset, adv_points_ckpt=ckpt_path,
        config=config, device=str(devices[0]),
        devices=devices, wrappers=wrappers,
    )

    os.makedirs(save_dir, exist_ok=True)

    adv_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    adv_pts = adv_ckpt['adv_points'].numpy()
    extent = adv_pts.max(0) - adv_pts.min(0)

    result_summary = {
        'asr': asr,
        'asr_pct': f'{asr*100:.1f}%',
        'n_success': stats['n_success'],
        'n_eligible': stats['n_eligible'],
        'checkpoint': ckpt_path,
        'method': adv_ckpt.get('method', 'pointopt'),
        'n_points': len(adv_pts),
        'object_size_m': {
            'x': round(float(extent[0]), 3),
            'y': round(float(extent[1]), 3),
            'z': round(float(extent[2]), 3),
        },
        'object_bbox': {
            'min': [round(float(v), 4) for v in adv_pts.min(0)],
            'max': [round(float(v), 4) for v in adv_pts.max(0)],
        },
    }

    result_path = os.path.join(save_dir, 'asr_pointopt_results.json')
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {result_path}")
    print(f"  ASR: {stats['n_success']}/{stats['n_eligible']} = {asr*100:.1f}%")
    print(f"  Object size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")
    print(f"  Points: {len(adv_pts)}")


def mode_blackbox(config, device, devices=None):
    """Phase 6: Black-box appearing attack (CMA-ES, PointOpt) with multi-GPU."""
    print("=" * 60)
    print("Phase 6: Black-box appearing attack (CMA-ES + PointOpt)")
    bb_cfg = config['attack'].get('blackbox', {})
    print(f"  n_points: {bb_cfg.get('n_points', 200)}")
    print(f"  init: {bb_cfg.get('init', 'gt')}")
    print("=" * 60)

    from attack.blackbox_appearing import run_blackbox_appearing_attack

    if devices is None:
        devices = [torch.device(device)]
    save_dir = get_save_dir(config)

    wrapper = get_wrapper(config, str(devices[0]))
    dataset = get_dataset(config)

    adv_points, history = run_blackbox_appearing_attack(
        dataset, wrapper, config, devices, save_dir=save_dir
    )

    pts_np = adv_points.numpy()
    extent = pts_np.max(0) - pts_np.min(0)

    bb_summary = {
        'method': 'CMA-ES_pointopt',
        'best_fitness': float(min(history['best_fitness'])),
        'best_asr': float(history['best_asr'][-1]) if history['best_asr'] else None,
        'generations': len(history['best_fitness']),
        'n_points': len(pts_np),
        'object_size_m': {
            'x': round(float(extent[0]), 3),
            'y': round(float(extent[1]), 3),
            'z': round(float(extent[2]), 3),
        },
        'object_bbox': {
            'min': [round(float(v), 4) for v in pts_np.min(0)],
            'max': [round(float(v), 4) for v in pts_np.max(0)],
        },
        'final_sigma': float(history['sigma'][-1]) if history['sigma'] else None,
        'config': {
            'popsize': bb_cfg.get('popsize', 48),
            'n_eval_samples': bb_cfg.get('n_eval_samples', 8),
            'sigma0': bb_cfg.get('sigma0', 0.15),
            'maxiter': bb_cfg.get('maxiter', 300),
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'blackbox_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(bb_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Blackbox attack summary saved to {summary_path}")
    print(f"  Best fitness: {min(history['best_fitness']):.4f}")
    print(f"  Best ASR: {history['best_asr'][-1]:.1%}" if history['best_asr'] else "")
    print(f"  Object size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")
    print(f"  Points: {len(pts_np)}")


def mode_physical_verify(config, devices, ckpt_path, mesh_method='poisson',
                         n_resample=None, sample_method='both'):
    """
    Physical verification pipeline (AAAI 2020 / AsiaCCS 2021):
      1. Load optimized adversarial points
      2. Surface reconstruction (Poisson / ConvexHull / etc.)
      3. Remove small connected components (< 1024 faces)
      4. Sample points from mesh: closest (AAAI 2020) and/or uniform
      5. Inject into scenes and run detection
      6. Compute and report ASR for each sampling strategy
    """
    print("=" * 60)
    print("Physical Verification Pipeline")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Mesh method: {mesh_method}")
    print(f"  Sample method: {sample_method}")
    print("=" * 60)
    save_dir = get_save_dir(config)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    adv_pts = ckpt['adv_points'].numpy()
    n_original = len(adv_pts)
    extent = adv_pts.max(0) - adv_pts.min(0)
    print(f"  Original points: {n_original}")
    print(f"  Original size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")

    from export_mesh import (poisson_reconstruct_and_clean,
                             reconstruct_mesh_from_points,
                             sample_points_from_mesh,
                             closest_sample_points_from_mesh,
                             export_obj, export_pointopt_html)

    # Step 1-3: Reconstruct mesh
    if mesh_method == 'poisson':
        print("\nStep 1-3: Screened Poisson reconstruction + cleanup")
        o3d_mesh, verts, faces = poisson_reconstruct_and_clean(
            adv_pts, depth=8, min_component_faces=1024
        )
    else:
        print(f"\nStep 1-3: {mesh_method} reconstruction")
        verts, faces = reconstruct_mesh_from_points(adv_pts, method=mesh_method)

    if len(faces) == 0:
        print("ERROR: reconstruction produced 0 faces, cannot continue")
        return

    re_verts_extent = verts.max(0) - verts.min(0)
    print(f"  Mesh size: {re_verts_extent[0]:.3f} x {re_verts_extent[1]:.3f}"
          f" x {re_verts_extent[2]:.3f} m")

    # Export mesh for inspection
    os.makedirs(save_dir, exist_ok=True)
    mesh_dir = os.path.join(save_dir, 'mesh_export')
    os.makedirs(mesh_dir, exist_ok=True)
    obj_path = os.path.join(mesh_dir, 'physical_reconstructed.obj')
    export_obj(verts, faces, obj_path, f'{mesh_method} reconstructed')

    # Build wrappers and dataset once
    from evaluation.metrics_pointopt import compute_pointopt_asr
    wrappers = build_multi_gpu_wrappers(config, devices)
    dataset = get_dataset(config)

    # Evaluate original ASR first
    print(f"\nStep 4: Evaluating original (pre-reconstruction) ASR...")
    asr_orig, stats_orig = compute_pointopt_asr(
        dataset=dataset, adv_points_ckpt=ckpt_path,
        config=config, device=str(devices[0]),
        devices=devices, wrappers=wrappers,
    )
    print(f"  Original ASR: {stats_orig['n_success']}/{stats_orig['n_eligible']}"
          f" = {asr_orig*100:.1f}%")

    # Determine which sampling methods to run
    methods_to_run = []
    if sample_method == 'both':
        methods_to_run = ['closest', 'uniform']
    else:
        methods_to_run = [sample_method]

    if n_resample is None:
        n_resample = n_original

    results_by_method = {}

    for sm in methods_to_run:
        print(f"\n{'='*60}")
        print(f"  Sampling: {sm} ({n_resample} points)")
        print(f"{'='*60}")

        if sm == 'closest':
            resampled_pts, surf_dists = closest_sample_points_from_mesh(
                adv_pts, verts, faces)
        else:
            resampled_pts = sample_points_from_mesh(verts, faces, n_resample)

        resampled_pts = resampled_pts - resampled_pts.mean(0)
        re_extent = resampled_pts.max(0) - resampled_pts.min(0)
        print(f"  Resampled size: {re_extent[0]:.3f} x {re_extent[1]:.3f}"
              f" x {re_extent[2]:.3f} m")

        ckpt_tag = f'adv_points_physical_{sm}.pth'
        resampled_ckpt_path = os.path.join(save_dir, ckpt_tag)
        torch.save({
            'adv_points': torch.tensor(resampled_pts, dtype=torch.float32),
            'method': f'physical_verify_{sm}',
            'original_ckpt': ckpt_path,
            'mesh_method': mesh_method,
            'sample_method': sm,
            'n_original': n_original,
            'n_resampled': n_resample,
        }, resampled_ckpt_path)

        html_path = os.path.join(mesh_dir, f'physical_{sm}.html')
        export_pointopt_html(resampled_pts, verts, faces, html_path,
                             f'Physical Verify — {sm} sampling')

        print(f"  Evaluating ASR ({sm})...")
        asr, stats = compute_pointopt_asr(
            dataset=dataset, adv_points_ckpt=resampled_ckpt_path,
            config=config, device=str(devices[0]),
            devices=devices, wrappers=wrappers,
        )

        retention = (asr / max(asr_orig, 1e-6)) * 100
        results_by_method[sm] = {
            'asr': asr,
            'asr_pct': f'{asr*100:.1f}%',
            'n_success': stats['n_success'],
            'n_eligible': stats['n_eligible'],
            'n_points': len(resampled_pts),
            'retention_pct': f'{retention:.1f}%',
            'object_size_m': {
                'x': round(float(re_extent[0]), 3),
                'y': round(float(re_extent[1]), 3),
                'z': round(float(re_extent[2]), 3),
            },
        }
        if sm == 'closest':
            results_by_method[sm]['mean_surface_dist'] = round(float(surf_dists.mean()), 4)
            results_by_method[sm]['max_surface_dist'] = round(float(surf_dists.max()), 4)

        print(f"  {sm} ASR: {stats['n_success']}/{stats['n_eligible']}"
              f" = {asr*100:.1f}% (retention: {retention:.1f}%)")

    # Save combined results
    result_summary = {
        'original': {
            'asr': asr_orig,
            'asr_pct': f'{asr_orig*100:.1f}%',
            'n_success': stats_orig['n_success'],
            'n_eligible': stats_orig['n_eligible'],
            'n_points': n_original,
            'object_size_m': {
                'x': round(float(extent[0]), 3),
                'y': round(float(extent[1]), 3),
                'z': round(float(extent[2]), 3),
            },
        },
        'mesh': {
            'method': mesh_method,
            'verts': len(verts),
            'faces': len(faces),
        },
        'sampling_results': results_by_method,
    }

    result_path = os.path.join(save_dir, 'physical_verify_results.json')
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Physical Verification Summary")
    print(f"{'='*60}")
    print(f"  Original ASR:  {stats_orig['n_success']}/{stats_orig['n_eligible']}"
          f" = {asr_orig*100:.1f}%")
    print(f"  Mesh: {mesh_method}, {len(verts)} verts, {len(faces)} faces")
    for sm, r in results_by_method.items():
        print(f"  {sm:8s} ASR: {r['n_success']}/{r['n_eligible']}"
              f" = {r['asr_pct']} (retention: {r['retention_pct']})")
    print(f"  Results: {result_path}")
    print(f"  Mesh:    {obj_path}")


def mode_eval(config, devices, ckpt_path):
    """Phase 7: Evaluate ASR (multi-GPU)."""
    print("=" * 60)
    print("Phase 7: Evaluating Appearing ASR")
    print("=" * 60)

    from evaluation.metrics import compute_appearing_asr

    wrappers = build_multi_gpu_wrappers(config, devices)
    dataset = get_dataset(config)
    save_dir = get_save_dir(config)

    asr, stats = compute_appearing_asr(
        model_wrapper=None, dataset=dataset, adv_mesh_ckpt=ckpt_path,
        config=config, device=str(devices[0]),
        devices=devices, wrappers=wrappers,
    )

    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'asr_results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'asr': asr,
            'n_eligible': stats['n_eligible'],
            'n_success': stats['n_success'],
        }, f, indent=2)

    print(f"\n  Results saved to {result_path}")


def mode_recall_iou(config, devices, ckpt_path):
    """Phase 7: Recall-IoU curve (multi-GPU)."""
    print("=" * 60)
    print("Phase 7: Computing Recall-IoU curve")
    print("=" * 60)

    from evaluation.metrics import compute_recall_iou_curve
    from evaluation.visualize import plot_recall_iou

    wrappers = build_multi_gpu_wrappers(config, devices)
    dataset = get_dataset(config)
    save_dir = get_save_dir(config)

    iou_thresholds, recall = compute_recall_iou_curve(
        model_wrapper=None, dataset=dataset, adv_mesh_ckpt=ckpt_path,
        config=config, device=str(devices[0]),
        devices=devices, wrappers=wrappers,
    )

    plot_recall_iou(
        iou_thresholds, recall,
        save_path=os.path.join(save_dir, 'recall_iou_appearing.png'),
        title='Appearing Attack Recall-IoU'
    )

    np.savez(os.path.join(save_dir, 'recall_iou_data.npz'),
             iou_thresholds=iou_thresholds, recall=recall)


def mode_eval_defenses(config, devices, ckpt_path):
    """Phase 7: Evaluate defenses (multi-GPU)."""
    print("=" * 60)
    print("Phase 7: Evaluating Defenses")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from attack.whitebox import apply_attack_to_sample, load_mesh_checkpoint
    from evaluation.metrics import (
        has_detection_near_pos, knn_outlier_removal,
        gaussian_noise_defense, density_defense_features,
    )
    from attack.inject import build_bev_occupancy, sample_injection_position

    wrappers = build_multi_gpu_wrappers(config, devices)
    n_gpus = len(devices)
    dataset = get_dataset(config)
    save_dir = get_save_dir(config)

    loaded = load_mesh_checkpoint(ckpt_path, map_location='cpu')
    mesh_param = loaded['mesh_param']
    translation_param = loaded['translation_param']
    v0 = loaded['v0']
    faces = loaded['faces']
    param_mode = loaded['param_mode']

    atk = config['attack']
    inj_cfg = atk['injection']
    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']

    n_eval_frames = min(len(dataset), 200)

    def _eval_one_defense(idx):
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
        inj_pos, _ = sample_injection_position(
            occ, grid_info,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
            rng=rng,
        )

        pc_t = torch.from_numpy(pc_np.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_clean, scores_clean = wrapper.detect(pc_t, score_thresh)
        if has_detection_near_pos(pred_clean, scores_clean, inj_pos,
                                  proximity, score_thresh):
            return None  # not eligible

        pc_adv, _ = apply_attack_to_sample(
            sample, mesh_param, translation_param, v0, faces, config, str(dev),
            injection_pos=inj_pos,
            param_mode=param_mode,
        )

        result = {
            'no_defense': False,
            'knn': False,
            'gaussian_0.01': False,
        }

        pc_adv_t = torch.from_numpy(pc_adv.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred, scores = wrapper.detect(pc_adv_t, score_thresh)
        if has_detection_near_pos(pred, scores, inj_pos, proximity, score_thresh):
            result['no_defense'] = True

        pc_filtered = knn_outlier_removal(pc_adv, k=5, alpha=0.1)
        pc_f_t = torch.from_numpy(pc_filtered.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_f, scores_f = wrapper.detect(pc_f_t, score_thresh)
        if has_detection_near_pos(pred_f, scores_f, inj_pos, proximity, score_thresh):
            result['knn'] = True

        pc_noisy = gaussian_noise_defense(pc_adv, sigma2=0.01)
        pc_n_t = torch.from_numpy(pc_noisy.astype(np.float32)).to(dev)
        with torch.no_grad():
            pred_n, scores_n = wrapper.detect(pc_n_t, score_thresh)
        if has_detection_near_pos(pred_n, scores_n, inj_pos, proximity, score_thresh):
            result['gaussian_0.01'] = True

        return result

    frame_results = [None] * n_eval_frames
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_eval_one_defense, idx): idx
                   for idx in range(n_eval_frames)}
        for future in tqdm(futures, total=n_eval_frames, desc='Defense eval'):
            idx = futures[future]
            frame_results[idx] = future.result()

    defense_results = {
        'no_defense': {'eligible': 0, 'success': 0},
        'knn': {'eligible': 0, 'success': 0},
        'gaussian_0.01': {'eligible': 0, 'success': 0},
    }
    for r in frame_results:
        if r is None:
            continue
        for def_name in defense_results:
            defense_results[def_name]['eligible'] += 1
            if r[def_name]:
                defense_results[def_name]['success'] += 1

    print("\nDefense Evaluation Results:")
    print("-" * 50)
    for name, res in defense_results.items():
        n = res['eligible']
        s = res['success']
        asr = s / max(n, 1)
        print(f"  {name:20s}: ASR = {s}/{n} = {asr*100:.1f}%")

    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'defense_results.json')
    with open(result_path, 'w') as f:
        json.dump(defense_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='AsiaCCS 2021 Appearing Attack')
    parser.add_argument('--mode', required=True,
                        choices=['test_inference', 'precompute', 'whitebox',
                                 'pointopt', 'blackbox', 'eval',
                                 'eval_pointopt', 'recall_iou', 'defenses',
                                 'physical_verify'])
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu', default=None,
                        help='Comma-separated GPU IDs for multi-GPU, e.g. 0,1,2,3')
    parser.add_argument('--ckpt', default=None, help='Path to adv mesh checkpoint')
    parser.add_argument('--mesh-method', default='poisson',
                        choices=['poisson', 'convex_hull', 'alpha', 'ball_pivoting'],
                        help='Mesh reconstruction method for physical_verify')
    parser.add_argument('--n-resample', type=int, default=None,
                        help='Number of points to resample from mesh (default: same as original)')
    parser.add_argument('--sample-method', default='both',
                        choices=['closest', 'uniform', 'both'],
                        help='Sampling strategy: closest (AAAI20), uniform, or both')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set default CUDA device so that OpenPCDet's custom CUDA kernels
    # (which use torch.cuda.*Tensor() without explicit device) allocate
    # on the correct GPU.
    if args.device.startswith('cuda'):
        import re
        m = re.search(r'cuda:(\d+)', args.device)
        if m:
            torch.cuda.set_device(int(m.group(1)))

    if args.mode == 'test_inference':
        mode_test_inference(config, args.device)
    elif args.mode == 'precompute':
        mode_precompute(config, args.device)
    elif args.mode == 'whitebox':
        mode_whitebox(config, args.device, warm_start_ckpt=args.ckpt)
    elif args.mode == 'pointopt':
        devices = _parse_devices(args)
        mode_pointopt(config, args.device, warm_start_ckpt=args.ckpt,
                      devices=devices)
    elif args.mode == 'eval_pointopt':
        if not args.ckpt:
            print("ERROR: --ckpt required for eval_pointopt mode")
            sys.exit(1)
        devices = _parse_devices(args)
        mode_eval_pointopt(config, devices, args.ckpt)
    elif args.mode == 'blackbox':
        devices = _parse_devices(args)
        mode_blackbox(config, args.device, devices)
    elif args.mode == 'eval':
        if not args.ckpt:
            print("ERROR: --ckpt required for eval mode")
            sys.exit(1)
        devices = _parse_devices(args)
        mode_eval(config, devices, args.ckpt)
    elif args.mode == 'recall_iou':
        if not args.ckpt:
            print("ERROR: --ckpt required for recall_iou mode")
            sys.exit(1)
        devices = _parse_devices(args)
        mode_recall_iou(config, devices, args.ckpt)
    elif args.mode == 'defenses':
        if not args.ckpt:
            print("ERROR: --ckpt required for defenses mode")
            sys.exit(1)
        devices = _parse_devices(args)
        mode_eval_defenses(config, devices, args.ckpt)
    elif args.mode == 'physical_verify':
        if not args.ckpt:
            print("ERROR: --ckpt required for physical_verify mode")
            sys.exit(1)
        devices = _parse_devices(args)
        mode_physical_verify(config, devices, args.ckpt,
                             mesh_method=args.mesh_method,
                             n_resample=args.n_resample,
                             sample_method=args.sample_method)


if __name__ == '__main__':
    main()
