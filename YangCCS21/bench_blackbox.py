"""
Benchmark script for black-box attack speed (PointOpt version).
Runs 2 CMA-ES generations with reduced settings to measure per-generation time.
"""
import os, sys, time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset
from attack.inject import build_bev_occupancy, sample_injection_position
from attack.whitebox_pointopt import init_car_surface_points

import cma


def load_config():
    with open('configs/attack_config.yaml') as f:
        return yaml.safe_load(f)


def bench():
    config = load_config()
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    print("Loading model...")
    t0 = time.time()
    wrapper = PointRCNNWrapper(
        config['model']['pointrcnn_config'],
        config['model']['pointrcnn_ckpt'],
        'cuda:0',
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("Loading dataset...")
    dataset = KITTIDataset(
        root=config['data']['kitti_root'],
        split=config['data']['split'],
        pc_range=config['data']['pc_range'],
    )
    print(f"  Dataset: {len(dataset)} samples")

    atk = config['attack']
    bb_cfg = atk['blackbox']
    inj_cfg = atk['injection']
    score_thresh = config['model']['score_thresh']
    proximity = config['eval']['proximity_thresh']

    n_points = bb_cfg.get('n_points', 200)
    target_car_size = tuple(atk.get('phantom_box_size', [3.9, 1.6, 1.56]))
    half_ext = [s / 2.0 for s in target_car_size]
    dim = n_points * 3

    # Bench settings
    popsize = 8
    n_eval = 5
    maxiter = 2

    # Pre-compute injection positions for a few frames
    rng = np.random.RandomState(42)
    inj_cache = {}
    valid_indices = []
    for idx in range(min(50, len(dataset))):
        sample = dataset[idx]
        occ, gi = build_bev_occupancy(
            sample['pointcloud'], sample['gt_bboxes'],
            x_range=tuple(inj_cfg['x_range']),
            y_range=tuple(inj_cfg['y_range']),
            resolution=inj_cfg['resolution'],
            margin=inj_cfg['margin'],
        )
        pos, valid = sample_injection_position(
            occ, gi,
            min_clearance=inj_cfg['min_clearance'],
            fallback_pos=tuple(inj_cfg['fallback_pos']),
            rng=rng,
        )
        inj_cache[idx] = pos
        if valid:
            valid_indices.append(idx)
    if not valid_indices:
        valid_indices = list(range(min(50, len(dataset))))

    # Initialise points
    init_pts = init_car_surface_points(n_points, size=target_car_size, device='cpu')
    x0 = init_pts.numpy().flatten().astype(np.float64)

    opts = {'popsize': popsize, 'maxiter': maxiter, 'verb_disp': 0,
            'verb_log': 0, 'CMA_diagonal': True}
    es = cma.CMAEvolutionStrategy(x0, 0.15, opts)

    from attack.blackbox_appearing import _eval_candidates_on_gpu

    # Warmup GPU
    print("Warming up GPU...")
    sample0 = dataset[0]
    pc_t = torch.tensor(sample0['pointcloud'], dtype=torch.float32, device=device)
    with torch.no_grad():
        wrapper.detect(pc_t, score_thresh)
    torch.cuda.synchronize()

    print(f"\nBenchmark: popsize={popsize}, n_eval={n_eval}, maxiter={maxiter}, "
          f"n_points={n_points}, dim={dim}")
    print("=" * 60)

    gen_times = []
    for gen in range(maxiter):
        solutions = es.ask()

        sample_indices = np.random.choice(
            valid_indices, size=min(n_eval, len(valid_indices)), replace=False)

        frame_pcs = [
            torch.tensor(dataset[int(si)]['pointcloud'],
                         dtype=torch.float32, device=device)
            for si in sample_indices
        ]
        frame_inj_pos = [inj_cache[int(si)] for si in sample_indices]

        torch.cuda.synchronize()
        t_gen_start = time.time()

        sol_indices = list(range(len(solutions)))
        best_fitness_ref = [float('inf')]
        results = _eval_candidates_on_gpu(
            sol_indices, solutions, n_points, half_ext,
            frame_pcs, frame_inj_pos,
            wrapper, device,
            score_thresh, proximity,
            reg_weight=0.001,
            best_fitness_ref=best_fitness_ref,
        )
        fitnesses = [r[0] for r in results]

        torch.cuda.synchronize()
        t_gen = time.time() - t_gen_start
        gen_times.append(t_gen)

        es.tell(solutions, fitnesses)
        n_inferences = popsize * n_eval
        print(f"  Gen {gen}: {t_gen:.2f}s  (n_inferences={n_inferences}, "
              f"{t_gen/n_inferences:.4f}s/inference)")

    avg = np.mean(gen_times)
    print(f"\nAverage: {avg:.2f}s/gen")
    print(f"  Per inference:  {avg/popsize/n_eval:.4f}s")

    # Extrapolate to full config
    full_popsize = bb_cfg.get('popsize', 128)
    full_n_eval = bb_cfg.get('n_eval_samples', 20)
    full_maxiter = bb_cfg.get('maxiter', 300)
    per_inf = avg / popsize / n_eval
    full_time = per_inf * full_popsize * full_n_eval * full_maxiter
    print(f"\nExtrapolated full run ({full_popsize}×{full_n_eval}×{full_maxiter}):")
    print(f"  ~{full_time:.0f}s = {full_time/3600:.1f}h")


if __name__ == '__main__':
    bench()
