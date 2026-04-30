"""
Genetic-evolving black-box attack (paper Section 3.3, Algorithm 1).

Generates adversarial meshes against PointPillar via genetic algorithm.
No gradient required — only input-output (point cloud → detection) pairs.

Paper parameters:
  - Population size: 160
  - Mutation stddev: 0.05
  - Mutation probability: 0.2
  - Leftover ratio: 0.5
  - Fitness: l_cw + ω1*l_lap + ω2*l_edge + ω3*l_nor
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.mesh import create_icosphere, build_adjacency
from attack.renderer import render_adversarial_points
from attack.ground import generate_ground_raytrace
from attack.inject import inject_points


def _compute_laplacian_loss(verts_np, adj):
    """Laplacian smoothing loss (numpy)."""
    V = len(verts_np)
    total = 0.0
    for i in range(V):
        nbrs = adj[i]
        if not nbrs:
            continue
        mean_nbr = verts_np[nbrs].mean(axis=0)
        diff = verts_np[i] - mean_nbr
        total += np.dot(diff, diff)
    return total


def _compute_edge_loss(verts_np, faces_np):
    """Edge length regularization: penalize long edges."""
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    return float((e1**2 + e2**2 + e3**2).mean())


def _compute_normal_loss(verts_np, faces_np):
    """Normal consistency loss: penalize inconsistent face normals."""
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms
    # Variance of normals as consistency measure
    return float(normals.var())


def apply_physical_constraints_np(verts, size_limit):
    """Clamp mesh vertices to size_limit (paper Eq. 7)."""
    sx, sy, sz = size_limit
    centre = verts.mean(axis=0, keepdims=True)
    offset = verts - centre
    offset[:, 0] = np.clip(offset[:, 0], -sx, sx)
    offset[:, 1] = np.clip(offset[:, 1], -sy, sy)
    offset[:, 2] = np.clip(offset[:, 2], -sz, sz)
    return centre + offset


def evaluate_fitness(model, verts_np, faces, adj, faces_np,
                     object_pos, ground_pts, lidar_cfg,
                     n_views=10, score_thresh=0.3, device='cuda:0',
                     rng=None):
    """
    Evaluate fitness of a single mesh.

    Fitness = detection score (averaged over views)
              - ω1*l_lap - ω2*l_edge - ω3*l_nor

    Higher fitness = better adversarial mesh.
    Paper coefficients: ω1=0.1, ω2=1.0, ω3=0.01
    """
    if rng is None:
        rng = np.random.RandomState()

    dev = torch.device(device)
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=dev)
    sensor = torch.zeros(3, device=dev)

    x_range = [-3.0, 3.0]
    y_range = [-1.0, 1.0]
    z_range = [0.7, 0.8]

    total_score = 0.0
    n_detected = 0

    for _ in range(n_views):
        lx = rng.uniform(x_range[0], x_range[1])
        ly = rng.uniform(y_range[0], y_range[1])
        lz = rng.uniform(z_range[0], z_range[1])
        lidar_pos = np.array([lx, ly, lz], dtype=np.float32)
        rel_pos = object_pos - lidar_pos

        vw = verts_t + torch.tensor(rel_pos, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            pts = render_adversarial_points(
                vw, faces, sensor,
                n_elevation=lidar_cfg.get('n_elevation', 16),
                elev_min_deg=lidar_cfg.get('elev_min_deg', -15.0),
                elev_max_deg=lidar_cfg.get('elev_max_deg', 15.0),
                h_step_deg=lidar_cfg.get('h_step_deg', 0.2),
                margin_deg=lidar_cfg.get('ray_margin_deg', 2.0),
            )

        if pts.shape[0] == 0:
            continue

        adv_pts_np = pts.cpu().numpy()  # (M, 3) already in LiDAR frame
        merged, _ = inject_points(ground_pts.copy(), adv_pts_np, np.zeros(3), remove_overlap=True)

        max_score, n_det = model.detect_score(
            merged, target_pos=rel_pos, radius=3.0, score_thresh=0.0
        )
        total_score += max_score
        if max_score >= score_thresh:
            n_detected += 1

    avg_score = total_score / max(n_views, 1)

    det_rate = n_detected / max(n_views, 1)

    # Paper Eq. 12: fitness = l_cw + ω1*l_lap + ω2*l_edge + ω3*l_nor
    # ω1=0.1, ω2=1.0, ω3=0.01
    l_lap = _compute_laplacian_loss(verts_np, adj)
    l_edge = _compute_edge_loss(verts_np, faces_np)
    l_nor = _compute_normal_loss(verts_np, faces_np)

    fitness = (avg_score + 0.3 * det_rate
               - 0.1 * l_lap - 1.0 * l_edge - 0.01 * l_nor)
    info = {
        'avg_score': avg_score,
        'n_detected': n_detected,
        'det_rate': det_rate,
        'l_lap': l_lap,
        'l_edge': l_edge,
        'l_nor': l_nor,
    }
    return fitness, info


def mutate(verts_np, mutation_std, mutation_prob, rng):
    """Mutate mesh vertices with probability mutation_prob."""
    mask = rng.random(verts_np.shape[0]) < mutation_prob
    noise = rng.randn(mask.sum(), 3) * mutation_std
    verts_new = verts_np.copy()
    verts_new[mask] += noise
    return verts_new


def crossover(parent1, parent2, rng):
    """Single-point crossover of vertex arrays."""
    n = len(parent1)
    point = rng.randint(1, n)
    child1 = np.vstack([parent1[:point], parent2[point:]])
    child2 = np.vstack([parent2[:point], parent1[point:]])
    return child1, child2


def run_genetic_attack(config_path, output_dir='results',
                       device='cuda:0',
                       population_size=160,
                       n_generations=500,
                       mutation_std_init=0.05,
                       mutation_prob=0.2,
                       leftover_ratio=0.5,
                       n_eval_views=10,
                       checkpoint_interval=50,
                       seed=42):
    """
    Run genetic algorithm attack (paper Algorithm 1).

    Args:
        config_path: path to attack_config.yaml
        output_dir: results directory
        device: torch device
        population_size: N (paper: 160)
        n_generations: T (paper: 1000 queries → ~500 generations)
        mutation_std_init: initial mutation stddev (paper: 0.05)
        mutation_prob: mutation probability (paper: 0.2)
        leftover_ratio: fraction to keep (paper: 0.5)
        n_eval_views: viewpoints per fitness evaluation
        checkpoint_interval: save every N generations
        seed: random seed
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rng = np.random.RandomState(seed)
    atk = config['attack']
    lidar_cfg = atk.get('lidar', {})
    size_limit = tuple(atk['size_limit'])
    mesh_radius = atk.get('mesh_radius', 0.4)
    subdivisions = atk.get('mesh_subdivisions', 3)
    object_pos = np.array(atk['object_pos'], dtype=np.float32)
    score_thresh = config['model']['score_thresh']

    os.makedirs(output_dir, exist_ok=True)

    # Create base icosphere
    v0, faces_t, adj = create_icosphere(subdivisions=subdivisions, radius=mesh_radius)
    v0_np = v0.numpy()
    faces_t = faces_t.to(device)
    faces_np = faces_t.cpu().numpy()
    n_verts = v0_np.shape[0]

    print(f"Genetic Attack Config:")
    print(f"  Population: {population_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Mutation std: {mutation_std_init}, prob: {mutation_prob}")
    print(f"  Leftover ratio: {leftover_ratio}")
    print(f"  Mesh: {n_verts} verts, radius={mesh_radius}")
    print(f"  Size limit: {size_limit}")
    print(f"  Eval views/member: {n_eval_views}")

    # Load PointPillar model
    from model.pointpillar_wrapper import PointPillarWrapper
    pp_config = config['model'].get('pointpillar_config',
        '/root/AsiaCCS2021-RoadsideAttack-Reimplemnetation/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml')
    pp_ckpt = config['model'].get('pointpillar_ckpt',
        'model/checkpoint/pointpillar_7728.pth')
    model = PointPillarWrapper(pp_config, pp_ckpt, device=device)
    print(f"  Model: PointPillar loaded")

    # Ground points (deterministic)
    ground_z = config.get('pose_sweep', {}).get('ground_z', -0.75)
    ground_pts = generate_ground_raytrace(ground_z=ground_z, h_step_deg=0.5)
    print(f"  Ground: {len(ground_pts)} deterministic points")

    # Load warm-start seed if available
    seed_path = os.path.join(output_dir, '..', 'random_search_best.pth')
    if not os.path.exists(seed_path):
        seed_path = 'results/random_search_best.pth'
    warm_seed = None
    if os.path.exists(seed_path):
        ws = torch.load(seed_path, map_location='cpu')
        warm_seed = (ws['v0'].numpy() + ws['mesh_param'].numpy()).astype(np.float32)
        print(f"  Warm start from {seed_path} (score={ws.get('best_score', '?')})")

    population = []
    for i in range(population_size):
        if warm_seed is not None and i < population_size * 3 // 4:
            # 75% seeded from warm-start with varying mutation
            noise_std = rng.uniform(0.01, 0.15)
            member = warm_seed.copy() + rng.randn(n_verts, 3).astype(np.float32) * noise_std
        else:
            # 25% random exploration
            member = v0_np.copy() + rng.randn(n_verts, 3).astype(np.float32) * rng.uniform(0.05, 0.3)
            if rng.random() < 0.5:
                member[:, 1] *= rng.uniform(0.2, 0.6)
        member = apply_physical_constraints_np(member, size_limit)
        population.append(member)
    print(f"  Population initialized: {len(population)} members"
          f" ({'warm-seeded' if warm_seed is not None else 'random'})")

    n_keep = int(population_size * leftover_ratio)
    best_fitness = -float('inf')
    best_verts = None
    best_info = None
    mutation_std = mutation_std_init

    log_path = os.path.join(output_dir, 'genetic_attack_log.txt')
    with open(log_path, 'w') as log_f:
        log_f.write(f"gen,best_fit,avg_fit,best_score,avg_score,n_det,mut_std\n")

    t0 = time.time()

    for gen in range(n_generations):
        # Evaluate fitness of all population members
        fitnesses = []
        infos = []
        for member in population:
            fit, info = evaluate_fitness(
                model, member, faces_t, adj, faces_np,
                object_pos, ground_pts, lidar_cfg,
                n_views=n_eval_views, score_thresh=score_thresh,
                device=device, rng=rng,
            )
            fitnesses.append(fit)
            infos.append(info)

        fitnesses = np.array(fitnesses)
        sorted_idx = np.argsort(-fitnesses)  # descending

        gen_best = fitnesses[sorted_idx[0]]
        gen_avg = fitnesses.mean()
        gen_best_info = infos[sorted_idx[0]]
        avg_score = np.mean([info['avg_score'] for info in infos])

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_verts = population[sorted_idx[0]].copy()
            best_info = gen_best_info

        total_det = sum(info['n_detected'] for info in infos)

        elapsed = time.time() - t0
        print(f"Gen {gen:4d} | best_fit={gen_best:.6f} avg_fit={gen_avg:.6f} "
              f"best_score={gen_best_info['avg_score']:.4f} avg_score={avg_score:.4f} "
              f"det={total_det}/{population_size*n_eval_views} "
              f"mut_std={mutation_std:.4f} [{elapsed:.0f}s]")

        with open(log_path, 'a') as log_f:
            log_f.write(f"{gen},{gen_best:.6f},{gen_avg:.6f},"
                        f"{gen_best_info['avg_score']:.4f},{avg_score:.4f},"
                        f"{total_det},{mutation_std:.4f}\n")

        # Save checkpoints
        if gen % checkpoint_interval == 0 or gen == n_generations - 1:
            ckpt = {
                'generation': gen,
                'v0': torch.tensor(v0_np),
                'mesh_param': torch.tensor(best_verts - v0_np),
                'translation_param': torch.zeros(3),
                'faces': faces_t.cpu(),
                'mesh_radius': mesh_radius,
                'size_limit': list(size_limit),
                'best_fitness': best_fitness,
                'best_info': best_info,
                'population_size': population_size,
                'mutation_std': mutation_std,
            }
            ckpt_path = os.path.join(output_dir, 'genetic_best.pth')
            torch.save(ckpt, ckpt_path)
            if gen % (checkpoint_interval * 2) == 0:
                torch.save(ckpt, os.path.join(output_dir, f'genetic_gen{gen:04d}.pth'))

        # Selection: keep top n_keep
        survivors = [population[i].copy() for i in sorted_idx[:n_keep]]

        new_population = list(survivors)

        # Inject fresh random members (5%) to maintain diversity
        n_fresh = max(1, population_size // 20)
        for _ in range(n_fresh):
            noise = rng.randn(n_verts, 3).astype(np.float32) * 0.1
            fresh = v0_np + noise
            fresh = apply_physical_constraints_np(fresh, size_limit)
            new_population.append(fresh)

        while len(new_population) < population_size:
            i1, i2 = rng.choice(n_keep, 2, replace=False)
            child1, child2 = crossover(survivors[i1], survivors[i2], rng)

            child1 = mutate(child1, mutation_std, mutation_prob, rng)
            child1 = apply_physical_constraints_np(child1, size_limit)
            new_population.append(child1)

            if len(new_population) < population_size:
                child2 = mutate(child2, mutation_std, mutation_prob, rng)
                child2 = apply_physical_constraints_np(child2, size_limit)
                new_population.append(child2)

        population = new_population[:population_size]

        # Adaptive mutation
        if gen > 0 and gen % 30 == 0:
            if best_info['avg_score'] < 0.2:
                mutation_std = min(mutation_std * 1.3, 0.3)
            elif best_info['avg_score'] > 0.4:
                mutation_std = max(mutation_std * 0.8, 0.01)

    # Final save
    total_time = time.time() - t0
    print(f"\nGenetic attack complete in {total_time:.1f}s")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Best avg score: {best_info['avg_score']:.4f}")
    print(f"Best checkpoint: {os.path.join(output_dir, 'genetic_best.pth')}")

    return best_verts, best_fitness, best_info


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/attack_config.yaml')
    parser.add_argument('--output', default='results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--population', type=int, default=160)
    parser.add_argument('--generations', type=int, default=500)
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_genetic_attack(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        population_size=args.population,
        n_generations=args.generations,
        n_eval_views=args.views,
        seed=args.seed,
    )
