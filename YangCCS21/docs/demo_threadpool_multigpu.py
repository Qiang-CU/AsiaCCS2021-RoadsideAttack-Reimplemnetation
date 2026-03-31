"""
ThreadPoolExecutor 多卡并行 — 最小闭环学习 Demo

场景：用多张 GPU 并行训练一个 MLP（每卡一个模型副本）。
和任何具体项目无关，纯粹演示 ThreadPoolExecutor 多卡模式。

覆盖三个核心用法：
  Demo 1 — 多卡并行推理
  Demo 2 — 多卡并行训练（模型权重优化）
  Demo 3 — 多卡并行输入优化（固定模型，优化输入）

运行：
  python docs/demo_threadpool_multigpu.py --gpus 0,1
  python docs/demo_threadpool_multigpu.py --gpus 0,1,2,3
  python docs/demo_threadpool_multigpu.py --gpus 0          # 单卡基线
"""

import argparse
import time
import copy
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor


# =====================================================================
# 一个普通的 MLP
# =====================================================================

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 工具函数
# =====================================================================

def make_fake_data(n_samples=512, in_dim=784, n_classes=10, device='cpu'):
    """生成假数据（模拟 MNIST）。"""
    X = torch.randn(n_samples, in_dim, device=device)
    y = torch.randint(0, n_classes, (n_samples,), device=device)
    return X, y


def create_replicas(model, devices):
    """在每张 GPU 上放一个模型副本。返回 {device: model}。"""
    replicas = {}
    for dev in devices:
        torch.cuda.set_device(dev)
        m = copy.deepcopy(model)
        m.to(dev)
        replicas[dev] = m
        print(f"  Replica on {dev}")
    return replicas


def sync_replicas(replicas, source_model, devices):
    """把 source_model 的权重同步到所有副本。"""
    sd = source_model.state_dict()
    for dev in devices:
        replicas[dev].load_state_dict(sd)


# =====================================================================
# Demo 1: 多卡并行推理
# =====================================================================

def demo_inference(replicas, devices, n_samples=2048):
    print(f"\n{'─'*50}")
    print(f"Demo 1: 并行推理 ({n_samples} 样本, {len(devices)} GPU)")
    print(f"{'─'*50}")

    # 准备数据：每个样本是一个 numpy-like tensor（模拟从磁盘读取）
    all_data = [torch.randn(784) for _ in range(n_samples)]

    def infer_batch_on_gpu(model, batch, device):
        """一批样本在一张卡上推理。"""
        torch.cuda.set_device(device)
        X = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(X)
        return logits.cpu()

    n_gpus = len(devices)

    # 按 GPU 分组
    chunks = [[] for _ in range(n_gpus)]
    for i, x in enumerate(all_data):
        chunks[i % n_gpus].append(x)

    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = []
        for gi, dev in enumerate(devices):
            fut = pool.submit(
                infer_batch_on_gpu, replicas[dev], chunks[gi], dev
            )
            futures.append(fut)

        for fut in futures:
            results.append(fut.result())

    all_logits = torch.cat(results, dim=0)
    elapsed = time.time() - t0
    print(f"  {n_samples} 样本 → {elapsed*1000:.1f}ms, "
          f"输出 shape: {all_logits.shape}")
    return elapsed


# =====================================================================
# Demo 2: 多卡并行训练（优化模型权重）
# =====================================================================

def demo_training(model, replicas, devices, n_steps=100, batch_size=256):
    print(f"\n{'─'*50}")
    print(f"Demo 2: 并行训练 ({n_steps} 步, batch={batch_size}, "
          f"{len(devices)} GPU)")
    print(f"{'─'*50}")

    n_gpus = len(devices)
    primary = devices[0]
    model.to(primary).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def train_chunk_on_gpu(replica, X_chunk, y_chunk, device, total_n):
        """在一张卡上算一部分 batch 的梯度。"""
        torch.cuda.set_device(device)
        X = X_chunk.to(device)
        y = y_chunk.to(device)
        replica.zero_grad()
        logits = replica(X)
        loss = loss_fn(logits, y) * (len(X) / total_n)
        loss.backward()
        # 收集梯度
        grads = {name: p.grad.clone()
                 for name, p in replica.named_parameters()
                 if p.grad is not None}
        return grads, loss.item() * total_n / len(X)

    t0 = time.time()
    for step in range(n_steps):
        X, y = make_fake_data(batch_size, device='cpu')

        # 分块
        chunk_size = batch_size // n_gpus
        X_chunks = X.split(chunk_size)
        y_chunks = y.split(chunk_size)

        # 并行 forward + backward
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = []
            for gi, dev in enumerate(devices):
                if gi >= len(X_chunks):
                    break
                fut = pool.submit(
                    train_chunk_on_gpu,
                    replicas[dev], X_chunks[gi], y_chunks[gi],
                    dev, batch_size,
                )
                futures.append(fut)

            # 聚合梯度到主模型
            optimizer.zero_grad()
            total_loss = 0.0
            for fut in futures:
                grads, loss_val = fut.result()
                for name, p in model.named_parameters():
                    g = grads[name].to(primary)
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad += g
                total_loss += loss_val

        optimizer.step()

        # 同步权重到所有副本
        sync_replicas(replicas, model, devices)

        if (step + 1) % 25 == 0:
            print(f"  Step {step+1:3d}: loss={total_loss/n_gpus:.4f}")

    elapsed = time.time() - t0
    ms_per_step = elapsed / n_steps * 1000
    print(f"  {n_steps} 步 → {elapsed:.2f}s ({ms_per_step:.1f}ms/step)")
    return elapsed


# =====================================================================
# Demo 3: 多卡并行输入优化（固定模型，优化输入 tensor）
# =====================================================================

def demo_input_optimization(replicas, devices, n_steps=100):
    print(f"\n{'─'*50}")
    print(f"Demo 3: 并行输入优化 ({n_steps} 步, {len(devices)} GPU)")
    print(f"  场景：固定模型，优化输入让模型输出逼近目标")
    print(f"{'─'*50}")

    n_gpus = len(devices)
    primary = devices[0]

    # 待优化的输入，在主卡上
    adv_input = torch.randn(64, 784, device=primary, requires_grad=True)
    target_label = 7  # 想让模型把这个输入分类为 7
    optimizer = torch.optim.Adam([adv_input], lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    # 模拟多个 "条件"（如不同场景/不同噪声），每个条件在一张卡上算
    n_conditions = n_gpus * 2

    def compute_loss_on_gpu(model, adv_input_main, noise_scale, device,
                            target, total_n):
        torch.cuda.set_device(device)
        inp = adv_input_main.detach().to(device).requires_grad_(True)
        # 加不同噪声模拟不同条件
        noisy = inp + torch.randn_like(inp) * noise_scale
        logits = model(noisy)
        target_t = torch.full((logits.shape[0],), target,
                              dtype=torch.long, device=device)
        loss = loss_fn(logits, target_t) / total_n
        loss.backward()
        return inp.grad.clone(), loss.item() * total_n

    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()

        conditions = [(0.1 * (i + 1)) for i in range(n_conditions)]

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = []
            for ci, noise_scale in enumerate(conditions):
                dev = devices[ci % n_gpus]
                fut = pool.submit(
                    compute_loss_on_gpu,
                    replicas[dev], adv_input, noise_scale, dev,
                    target_label, n_conditions,
                )
                futures.append(fut)

            total_loss = 0.0
            for fut in futures:
                grad, loss_val = fut.result()
                if adv_input.grad is None:
                    adv_input.grad = grad.to(primary)
                else:
                    adv_input.grad += grad.to(primary)
                total_loss += loss_val

        optimizer.step()

        if (step + 1) % 25 == 0:
            avg_loss = total_loss / n_conditions
            print(f"  Step {step+1:3d}: loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    ms_per_step = elapsed / n_steps * 1000
    print(f"  {n_steps} 步 → {elapsed:.2f}s ({ms_per_step:.1f}ms/step)")
    return elapsed


# =====================================================================
# 性能对比：多卡 vs 单卡
# =====================================================================

def benchmark(replicas, devices, model, n_steps=50, batch_size=512):
    print(f"\n{'='*50}")
    print(f"Benchmark: 多卡 vs 单卡 ({n_steps} 步)")
    print(f"{'='*50}")

    n_gpus = len(devices)
    primary = devices[0]
    loss_fn = nn.CrossEntropyLoss()

    # ── 多卡 ──
    model_multi = copy.deepcopy(model).to(primary).train()
    opt_multi = torch.optim.SGD(model_multi.parameters(), lr=0.01)
    sync_replicas(replicas, model_multi, devices)

    def train_chunk(replica, X_c, y_c, dev, total_n):
        torch.cuda.set_device(dev)
        replica.zero_grad()
        logits = replica(X_c.to(dev))
        loss = loss_fn(logits, y_c.to(dev)) * (len(X_c) / total_n)
        loss.backward()
        return {n: p.grad.clone() for n, p in replica.named_parameters()
                if p.grad is not None}

    # 预热
    X, y = make_fake_data(batch_size, device='cpu')
    chunk_size = batch_size // n_gpus
    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futs = [pool.submit(train_chunk, replicas[devices[gi]],
                            X[gi*chunk_size:(gi+1)*chunk_size],
                            y[gi*chunk_size:(gi+1)*chunk_size],
                            devices[gi], batch_size)
                for gi in range(n_gpus)]
        for f in futs:
            f.result()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_steps):
        X, y = make_fake_data(batch_size, device='cpu')
        X_cs, y_cs = X.split(chunk_size), y.split(chunk_size)
        opt_multi.zero_grad()
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futs = [pool.submit(train_chunk, replicas[devices[gi]],
                                X_cs[gi], y_cs[gi], devices[gi], batch_size)
                    for gi in range(min(n_gpus, len(X_cs)))]
            for f in futs:
                for n, p in model_multi.named_parameters():
                    g = f.result()[n].to(primary)
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad += g
        opt_multi.step()
        sync_replicas(replicas, model_multi, devices)
    torch.cuda.synchronize()
    t_multi = time.time() - t0

    # ── 单卡 ──
    model_single = copy.deepcopy(model).to(primary).train()
    opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)
    # 预热
    X, y = make_fake_data(batch_size, device=primary)
    model_single.zero_grad()
    loss_fn(model_single(X), y).backward()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_steps):
        X, y = make_fake_data(batch_size, device=primary)
        opt_single.zero_grad()
        loss = loss_fn(model_single(X), y)
        loss.backward()
        opt_single.step()
    torch.cuda.synchronize()
    t_single = time.time() - t0

    ms_multi = t_multi / n_steps * 1000
    ms_single = t_single / n_steps * 1000
    speedup = t_single / t_multi

    print(f"\n  单卡: {ms_single:.1f} ms/step")
    print(f"  {n_gpus}卡:  {ms_multi:.1f} ms/step")
    print(f"  加速比: {speedup:.2f}x")

    if speedup < 1.0:
        print(f"\n  注意：模型太小 ({sum(p.numel() for p in model.parameters())} 参数)，")
        print(f"  线程开销 > 计算量。换大模型/大 batch 才能看到加速。")
        print(f"  试试 --hidden 2048 --batch 4096")

    return speedup


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ThreadPoolExecutor 多卡并行学习 Demo')
    parser.add_argument('--gpus', default='0',
                        help='GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--hidden', type=int, default=512,
                        help='MLP hidden size (加大可看到更好加速比)')
    parser.add_argument('--batch', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--steps', type=int, default=50,
                        help='Steps per demo')
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    devices = [torch.device(f'cuda:{g}') for g in gpu_ids]
    primary = devices[0]
    torch.cuda.set_device(primary)

    print(f"\n{'='*50}")
    print(f"  ThreadPoolExecutor Multi-GPU Demo")
    print(f"  GPUs: {gpu_ids}, hidden={args.hidden}, batch={args.batch}")
    print(f"{'='*50}")

    # 创建模型和副本
    model = MLP(hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  MLP: {n_params:,} parameters")

    print("\n创建模型副本...")
    replicas = create_replicas(model, devices)

    # 三个 Demo
    demo_inference(replicas, devices, n_samples=args.batch * 4)
    demo_training(model, replicas, devices, n_steps=args.steps,
                  batch_size=args.batch)
    demo_input_optimization(replicas, devices, n_steps=args.steps)

    # 性能对比
    if len(devices) > 1:
        benchmark(replicas, devices, model, n_steps=args.steps,
                  batch_size=args.batch)
    else:
        print(f"\n  单卡模式，跳过对比。试试 --gpus 0,1 看加速比。")


if __name__ == '__main__':
    main()
