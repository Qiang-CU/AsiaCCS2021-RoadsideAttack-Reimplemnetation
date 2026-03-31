"""
Black-box appearing attack using CMA-ES with direct point optimisation.

Key design:
- Direct point-cloud parameterisation (same as whitebox_pointopt)
- Per-candidate early termination: bad candidates pruned after a few frames
- ThreadPoolExecutor distributes candidates across GPUs (CUDA releases GIL)
- Frames pre-cached ONCE to avoid repeated disk I/O
- CMA_diagonal mode for high-dimensional efficiency (dim = n_points * 3)

Usage (via run_attack.py):
    python run_attack.py --mode blackbox --gpu 0,1,2,3
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cma

from attack.inject import inject_points
from attack.whitebox_pointopt import (
    precompute_injections, init_car_surface_points, init_from_gt_cars,
)
from model.pointrcnn_wrapper import PointRCNNWrapper


def _eval_candidates_on_gpu(sol_indices, solutions, n_points, half_ext,
                            frame_pcs, frame_inj_pos, wrapper, device,
                            score_thresh, proximity, reg_weight,
                            best_fitness_ref):
    """
    Evaluate a list of CMA-ES candidates on one GPU, sequentially,
    with per-candidate early termination.

    ``best_fitness_ref`` is a mutable list [value] shared across GPUs so
    that a good result on one GPU tightens the pruning bound for others.
    """
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    n_frames = len(frame_pcs)
    results = []

    for si in sol_indices:
        vec = solutions[si]
        pts = torch.tensor(vec.reshape(n_points, 3),
                           dtype=torch.float32, device=device)
        for d in range(3):
            pts[:, d].clamp_(-half_ext[d], half_ext[d])

        reg = reg_weight * float(np.linalg.norm(vec))
        total_score = 0.0
        n_success = 0
        n_evaluated = 0

        with torch.no_grad():
            for fi in range(n_frames):
                inj_pos = frame_inj_pos[fi]
                inj_t = torch.tensor(inj_pos, dtype=torch.float32,
                                     device=device)
                merged, n_adv = inject_points(frame_pcs[fi], pts, inj_t,
                                              remove_overlap=True)
                if n_adv == 0:
                    n_evaluated += 1
                    continue

                boxes, scores = wrapper.detect(merged, score_thresh=0.1)
                n_evaluated += 1

                if len(boxes) > 0:
                    dists = np.sqrt((boxes[:, 0] - inj_pos[0])**2 +
                                    (boxes[:, 1] - inj_pos[1])**2)
                    near_mask = dists < proximity
                    if near_mask.any():
                        best = float(scores[near_mask].max())
                        total_score += best
                        if best >= score_thresh:
                            n_success += 1

                # Early termination: even if ALL remaining frames score
                # perfectly, can this candidate beat the current best?
                if n_evaluated < n_frames:
                    remaining = n_frames - n_evaluated
                    opt_score = (total_score + remaining * 1.0) / n_frames
                    opt_asr = (n_success + remaining) / n_frames
                    opt_fitness = -opt_score - 2.0 * opt_asr + reg
                    if opt_fitness >= best_fitness_ref[0]:
                        break

        mean_score = total_score / max(n_frames, 1)
        asr = n_success / max(n_frames, 1)
        fitness = -mean_score - 2.0 * asr + reg

        if fitness < best_fitness_ref[0]:
            best_fitness_ref[0] = fitness

        results.append((fitness, n_success, n_frames))

    return results


# ── Main CMA-ES attack loop (multi-GPU) ──────────────────────────────────

def run_blackbox_appearing_attack(dataset, model_wrapper, config, devices,
                                  save_dir='results'):
    """
    Black-box appearing attack using CMA-ES with direct point optimisation.

    Args:
        dataset:        KITTIDataset (val split)
        model_wrapper:  PointRCNNWrapper on primary device (used as template)
        config:         dict from attack_config.yaml
        devices:        list of torch.device
        save_dir:       output directory

    Returns:
        adv_points: (N, 3) optimised adversarial points
        history:    dict of fitness curves
    """
    os.makedirs(save_dir, exist_ok=True)
    atk = config['attack']
    bb_cfg = atk['blackbox']
    inj_cfg = atk['injection']
    n_gpus = len(devices)

    # PointOpt parameters
    n_points = bb_cfg.get('n_points', 200)
    init_mode = bb_cfg.get('init', 'gt')
    target_car_size = tuple(atk.get('phantom_box_size', [3.9, 1.6, 1.56]))
    half_ext = [s / 2.0 for s in target_car_size]

    sigma0 = bb_cfg.get('sigma0', 0.15)
    popsize = bb_cfg.get('popsize', 48)
    maxiter = bb_cfg.get('maxiter', 300)
    n_eval = bb_cfg.get('n_eval_samples', 8)
    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']

    dim = n_points * 3

    print(f'  CMA-ES black-box appearing attack (PointOpt, multi-GPU)')
    print(f'    n_points={n_points}, dim={dim}')
    print(f'    popsize={popsize}, n_eval={n_eval}, maxiter={maxiter}, '
          f'sigma0={sigma0}')
    print(f'    half_ext={half_ext}')
    print(f'    GPUs: {[str(d) for d in devices]}')

    # ── Model replicas ──
    wrappers = {}
    for dev in devices:
        if dev == devices[0]:
            wrappers[dev] = model_wrapper
        else:
            torch.cuda.set_device(dev)
            wrappers[dev] = PointRCNNWrapper(
                config['model']['pointrcnn_config'],
                config['model']['pointrcnn_ckpt'],
                str(dev),
            )
    print(f'    Model replicas loaded on {n_gpus} GPU(s)')

    # ── Initialise adversarial points ──
    if init_mode == 'gt':
        print(f"Initialising {n_points} points from GT car scans...")
        init_pts = init_from_gt_cars(dataset, n_points=n_points, device='cpu')
    else:
        print(f"Initialising {n_points} points on car-surface box...")
        init_pts = init_car_surface_points(
            n_points, size=target_car_size, device='cpu')

    x0 = init_pts.numpy().flatten().astype(np.float64)

    # ── Pre-compute injection positions (seed=42, same as pointopt) ──
    print("Pre-computing injection positions (seed=42)...")
    injection_cache, valid_indices = precompute_injections(dataset, inj_cfg)
    print(f"  {len(valid_indices)}/{len(dataset)} valid positions")

    # ── Pre-cache frames: load from disk ONCE ──
    n_pool = min(len(valid_indices), 500)
    pool_indices = valid_indices[:n_pool]
    print(f"Pre-caching {n_pool} frames...")

    frame_cache = {}
    for idx in tqdm(pool_indices, desc='Loading frames', leave=False):
        sample = dataset[int(idx)]
        frame_cache[idx] = {
            'pc_np': sample['pointcloud'],
            'inj_pos': injection_cache[idx]['pos'],
        }

    # Pre-load point clouds to each GPU (once)
    gpu_pcs = {}
    for dev in devices:
        gpu_pcs[dev] = {}
        for idx in pool_indices:
            gpu_pcs[dev][idx] = torch.tensor(
                frame_cache[idx]['pc_np'], dtype=torch.float32, device=dev
            )
    print(f"  GPU tensors pre-loaded ({n_pool} frames × {n_gpus} GPUs)")

    # ── CMA-ES ──
    opts = {
        'popsize': popsize,
        'maxiter': maxiter,
        'verb_disp': 0,
        'verb_log': 0,
        'tolx': 1e-8,
        'tolfun': bb_cfg.get('tolfun', 1e-8),
        'CMA_diagonal': bb_cfg.get('diagonal', True),
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    history = {
        'best_fitness': [],
        'gen_mean_fitness': [],
        'gen_asr': [],
        'best_asr': [],
        'sigma': [],
    }

    best_fitness = float('inf')
    best_vec = x0.copy()
    best_asr = 0.0

    pbar = tqdm(range(maxiter), desc='CMA-ES black-box (PointOpt)')
    gen = 0
    dev_list = list(devices)

    executor = ThreadPoolExecutor(max_workers=n_gpus)
    try:
        while not es.stop() and gen < maxiter:
            solutions = es.ask()

            # Sample frame indices for this generation
            gen_frame_indices = np.random.choice(
                pool_indices,
                size=min(n_eval, len(pool_indices)),
                replace=False,
            )

            gen_inj_pos_list = [frame_cache[idx]['inj_pos']
                                for idx in gen_frame_indices]

            # Split candidates round-robin across GPUs
            gpu_sol_indices = {dev: [] for dev in dev_list}
            for si in range(len(solutions)):
                gpu_sol_indices[dev_list[si % n_gpus]].append(si)

            # Shared pruning bound (mutable list so threads can update it)
            best_fitness_ref = [best_fitness]

            fitnesses = [None] * len(solutions)
            gen_successes = [0] * len(solutions)
            gen_totals = [0] * len(solutions)

            # Build per-GPU frame lists once (not inside closure)
            gen_gpu_pcs = {dev: [gpu_pcs[dev][idx]
                                 for idx in gen_frame_indices]
                           for dev in dev_list}

            def _make_task(dev, si_list, gpu_frames):
                def _task():
                    if not si_list:
                        return dev, []
                    res = _eval_candidates_on_gpu(
                        si_list, solutions, n_points, half_ext,
                        gpu_frames, gen_inj_pos_list,
                        wrappers[dev], dev,
                        score_thresh, proximity,
                        reg_weight=0.001,
                        best_fitness_ref=best_fitness_ref,
                    )
                    return dev, res
                return _task

            futures = []
            for dev in dev_list:
                task = _make_task(dev, gpu_sol_indices[dev],
                                 gen_gpu_pcs[dev])
                futures.append((dev, executor.submit(task)))

            for dev, future in futures:
                _, results = future.result()
                si_list = gpu_sol_indices[dev]
                for li, (fit, n_succ, n_tot) in enumerate(results):
                    si = si_list[li]
                    fitnesses[si] = fit
                    gen_successes[si] = n_succ
                    gen_totals[si] = n_tot

            es.tell(solutions, fitnesses)

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best = fitnesses[gen_best_idx]
            gen_best_asr = (gen_successes[gen_best_idx]
                            / max(gen_totals[gen_best_idx], 1))
            if gen_best < best_fitness:
                best_fitness = gen_best
                best_vec = solutions[gen_best_idx].copy()
                best_asr = gen_best_asr

            history['best_fitness'].append(best_fitness)
            history['gen_mean_fitness'].append(np.mean(fitnesses))
            history['gen_asr'].append(gen_best_asr)
            history['best_asr'].append(best_asr)
            history['sigma'].append(es.sigma)

            pbar.update(1)
            pbar.set_postfix({
                'best': f'{best_fitness:.3f}',
                'gen_best': f'{gen_best:.3f}',
                'asr': f'{best_asr:.1%}',
                'sigma': f'{es.sigma:.4f}',
            })

            if (gen + 1) % 20 == 0:
                _save_bb_checkpoint(best_vec, n_points, history,
                                    save_dir, tag='latest')
                print(f"\n  [Gen {gen+1}] best_fitness={best_fitness:.4f}, "
                      f"best_asr={best_asr:.1%}, sigma={es.sigma:.4f}")

            gen += 1
    finally:
        executor.shutdown(wait=False)

    pbar.close()
    _save_bb_checkpoint(best_vec, n_points, history,
                        save_dir, tag='final')

    adv_points = torch.tensor(best_vec.reshape(n_points, 3),
                              dtype=torch.float32)

    print(f"\n  CMA-ES finished: best fitness = {best_fitness:.4f}, "
          f"best ASR = {best_asr:.1%}")
    print(f"  adv_points norm = {torch.norm(adv_points).item():.3f}")

    # Cleanup GPU caches
    del gpu_pcs
    torch.cuda.empty_cache()

    return adv_points.detach(), history


def _save_bb_checkpoint(vec, n_points, history, save_dir, tag='latest'):
    """Save CMA-ES checkpoint."""
    adv_points = torch.tensor(vec.reshape(n_points, 3),
                              dtype=torch.float32)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'adv_points_blackbox_{tag}.pth')
    torch.save({
        'adv_points': adv_points,
        'history': history,
        'method': 'CMA-ES_pointopt',
    }, path)
