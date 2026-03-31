# ThreadPoolExecutor 多卡并行：最小闭环实践

## 核心原理

Python 有 GIL（全局解释器锁），多线程无法真正并行执行 Python 代码。但 **PyTorch 的 CUDA 操作会释放 GIL**——当 GPU 在执行 kernel 时，Python 线程是不持有 GIL 的。这意味着：

- 多个线程可以**同时**向不同 GPU 提交 CUDA kernel
- GPU 上的计算（forward / backward）是真正并行的
- CPU 端的轻量操作（数据预处理、dict 拼装）虽然受 GIL 限制，但占比很小

相比 `torch.multiprocessing` / `DistributedDataParallel`，ThreadPoolExecutor 的优势：

| | ThreadPoolExecutor | multiprocessing / DDP |
|--|--------------------|-----------------------|
| 启动开销 | 无（线程创建 ~μs） | 高（进程 fork/spawn ~ms） |
| 内存共享 | 天然共享（同一进程） | 需要 shared memory / NCCL |
| 通信开销 | 零（直接读写变量） | 序列化 + IPC |
| 代码复杂度 | 低 | 高（需要 rank/world_size/init_process_group） |
| 适用规模 | 单机 1~8 卡 | 单机多机均可 |

**适用场景**：单机多卡、模型不大（单卡能放下完整模型）、需要多帧/多样本并行推理或梯度计算。PointPillar、PIXOR、PointRCNN 这类 3D 检测模型都适合。

---

## 最小闭环示例

### 1. 创建模型副本

每张 GPU 上放一个独立的模型副本：

```python
import torch
from concurrent.futures import ThreadPoolExecutor

def create_model_replicas(model_class, ckpt_path, devices):
    """在每张 GPU 上创建一个模型副本。"""
    replicas = {}
    for dev in devices:
        torch.cuda.set_device(dev)
        model = model_class()
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(dev)
        model.eval()
        replicas[dev] = model
    return replicas
```

注意 `map_location='cpu'`——先加载到 CPU，再 `.to(dev)` 移动到目标 GPU。这避免 checkpoint 原始设备（通常 cuda:0）导致的设备不匹配。

### 2. 并行推理

最简单的场景：N 个样本分配到 K 张卡并行推理。

```python
def infer_on_gpu(model, sample, device):
    """单个样本在指定 GPU 上推理。"""
    # 关键：设置当前 CUDA 设备
    # 某些库的自定义 CUDA op 用 torch.cuda.*Tensor() 构造器，
    # 会分配到 current_device 而非 tensor 所在设备
    torch.cuda.set_device(device)

    x = torch.tensor(sample, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(x.unsqueeze(0))
    return output.cpu()


def parallel_inference(replicas, samples):
    """多卡并行推理。"""
    devices = list(replicas.keys())
    n_gpus = len(devices)

    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = {}
        for i, sample in enumerate(samples):
            dev = devices[i % n_gpus]  # round-robin 分配
            fut = pool.submit(infer_on_gpu, replicas[dev], sample, dev)
            futures[fut] = i

        for fut in futures:
            results[futures[fut]] = fut.result()

    return results
```

### 3. 并行梯度计算（对抗攻击 / 输入优化）

当优化目标是**输入**（不是模型权重）时，需要在每张卡上独立计算梯度，再汇总：

```python
def compute_grad_on_gpu(model, shared_input, sample_data, device, loss_fn,
                        total_batch_size):
    """在指定 GPU 上算 forward + backward，返回梯度。"""
    torch.cuda.set_device(device)

    # 从主设备拷贝一份到当前 GPU，开启梯度
    local_input = shared_input.detach().to(device).requires_grad_(True)
    x = torch.tensor(sample_data, dtype=torch.float32, device=device)

    output = model(x, local_input)
    loss = loss_fn(output)
    (loss / total_batch_size).backward()

    # 返回梯度（还在当前 GPU 上）和标量 loss
    return local_input.grad.clone(), loss.item()


def parallel_gradient_step(replicas, shared_input, batch_samples,
                           loss_fn, optimizer):
    """一步优化：多卡并行算梯度 → 主卡聚合 → optimizer.step()"""
    devices = list(replicas.keys())
    n_gpus = len(devices)
    primary_device = devices[0]

    # 按 round-robin 分配样本到各卡
    chunks = [[] for _ in range(n_gpus)]
    for i, sample in enumerate(batch_samples):
        chunks[i % n_gpus].append(sample)

    optimizer.zero_grad()

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = []
        for gi, dev in enumerate(devices):
            if not chunks[gi]:
                continue
            for sample in chunks[gi]:
                futures.append(pool.submit(
                    compute_grad_on_gpu,
                    replicas[dev], shared_input, sample, dev,
                    loss_fn, len(batch_samples),
                ))

        # 聚合梯度到主设备
        for fut in futures:
            grad, loss_val = fut.result()
            if shared_input.grad is None:
                shared_input.grad = grad.to(primary_device)
            else:
                shared_input.grad += grad.to(primary_device)

    optimizer.step()
```

---

## 必须注意的坑

### 坑 1：`torch.cuda.set_device()` 是线程局部的

```python
# 主线程
torch.cuda.set_device(0)  # 主线程 current_device = cuda:0

# 子线程
def worker():
    torch.cuda.set_device(3)  # 只影响当前线程
    # ...

# 主线程的 current_device 仍然是 cuda:0
```

**但是**，如果你在主线程中遍历多 GPU 创建模型：

```python
for dev in [cuda:0, cuda:1, ..., cuda:7]:
    torch.cuda.set_device(dev)
    models[dev] = create_model(dev)
# 循环结束后，主线程 current_device = cuda:7 !!
```

之后在主线程用 `models[cuda:0]` 做推理，如果模型内部用了 `torch.cuda.IntTensor()` 等过时构造器，中间 tensor 会跑到 cuda:7 而不是 cuda:0。

**解决**：在每次 forward 前显式 `torch.cuda.set_device(self.device)`。

### 坑 2：Checkpoint 设备

```python
# 错误：checkpoint 保存时在 cuda:0，加载时直接到 cuda:0
state = torch.load('model.pth')  # map_location 默认 None → 原始设备

# 正确：先到 CPU，再移到目标设备
state = torch.load('model.pth', map_location='cpu')
model.load_state_dict(state)
model.to(target_device)
```

### 坑 3：自定义 CUDA op 的设备分配

OpenPCDet、PointNet++ 等库的自定义 CUDA 算子常用过时 API：

```python
# 这些会分配到 torch.cuda.current_device()，不是输入 tensor 的设备！
output = torch.cuda.IntTensor(B, npoint)
output = torch.cuda.FloatTensor(B, C, N)
```

**解决**：在调用包含自定义 CUDA op 的模型前，确保 `torch.cuda.set_device` 和模型设备一致。最好在模型的 forward 入口统一设置：

```python
class MyModelWrapper:
    def forward(self, x):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        # ... 正常 forward
```

### 坑 4：线程数不要超过 GPU 数

```python
# 正确：max_workers = GPU 数量
ThreadPoolExecutor(max_workers=n_gpus)

# 错误：超过 GPU 数量，多个线程抢同一张卡
ThreadPoolExecutor(max_workers=32)  # 8 卡机器上开 32 线程
```

同一张卡上并发提交 kernel 不会加速，反而增加 CUDA context 切换开销。

### 坑 5：异常处理

```python
with ThreadPoolExecutor(max_workers=n_gpus) as pool:
    futures = [pool.submit(work, dev) for dev in devices]

    for fut in futures:
        try:
            result = fut.result()  # 这里会抛出子线程的异常
        except RuntimeError as e:
            print(f"GPU error: {e}")
            # 某张卡 OOM 不会影响其他卡
```

子线程的异常在 `fut.result()` 时才会抛出。如果不调用 `result()`，异常会被静默吞掉。

---

## 完整模板：多卡对抗攻击优化

```python
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def multi_gpu_attack(model_class, ckpt_path, dataset, n_iters=1000,
                     lr=0.01, gpu_ids=[0, 1, 2, 3]):
    devices = [torch.device(f'cuda:{g}') for g in gpu_ids]
    primary = devices[0]
    n_gpus = len(devices)

    # 1. 每卡一个模型副本
    replicas = {}
    for dev in devices:
        torch.cuda.set_device(dev)
        model = model_class()
        sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd)
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        replicas[dev] = model

    # 2. 待优化的输入（在主卡上）
    adv_input = torch.randn(400, 3, device=primary, requires_grad=True)
    optimizer = torch.optim.Adam([adv_input], lr=lr)

    # 3. 训练循环
    for step in range(n_iters):
        optimizer.zero_grad()

        batch = np.random.choice(len(dataset), size=8, replace=False)
        chunks = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(batch):
            chunks[i % n_gpus].append(idx)

        def process_chunk(gpu_idx):
            dev = devices[gpu_idx]
            torch.cuda.set_device(dev)
            local = adv_input.detach().to(dev).requires_grad_(True)

            total_loss = 0.0
            for idx in chunks[gpu_idx]:
                sample = torch.tensor(
                    dataset[idx], dtype=torch.float32, device=dev
                )
                out = replicas[dev](sample, local)
                loss = compute_loss(out)
                (loss / len(batch)).backward()
                total_loss += loss.item()

            grad = local.grad.clone() if local.grad is not None else \
                   torch.zeros_like(local)
            return grad, total_loss

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(process_chunk, gi)
                       for gi in range(n_gpus) if chunks[gi]]

            for fut in futures:
                grad, _ = fut.result()
                if adv_input.grad is None:
                    adv_input.grad = grad.to(primary)
                else:
                    adv_input.grad += grad.to(primary)

        optimizer.step()

    return adv_input.detach()
```

---

## 性能预期

以 PointRCNN（~6M 参数）为例，单帧 forward+backward ~50ms/GPU：

| GPU 数 | 每步耗时 (8帧 batch) | 加速比 |
|--------|----------------------|--------|
| 1 | ~400ms | 1.0x |
| 2 | ~220ms | 1.8x |
| 4 | ~130ms | 3.1x |
| 8 | ~90ms | 4.4x |

加速比不是线性的，因为：
- 梯度聚合有 GPU→CPU→GPU 拷贝开销
- ThreadPoolExecutor 有线程调度开销
- GIL 在 CPU 端数据准备时仍有竞争

对于更小的模型（PointPillar ~2M，PIXOR ~1M），单帧更快，线程/通信开销占比更高，加速比会略低。但 4 卡仍然能稳定 2.5x+ 加速。

---

## 什么时候不该用这个方案

1. **模型太大单卡放不下** → 用模型并行或 FSDP
2. **需要跨机器** → 用 DDP + NCCL
3. **训练模型权重（不是优化输入）** → 用 DDP，它有优化过的梯度同步
4. **batch 内样本有依赖** → 无法并行，老老实实串行
5. **纯 CPU 模型** → GIL 限制，ThreadPool 无法并行，用 ProcessPool
