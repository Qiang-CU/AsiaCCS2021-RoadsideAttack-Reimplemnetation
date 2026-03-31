# 开发日志：Appearing Attack 物理可实现性

## 1. Mesh 白盒 → PointOpt 白盒

**问题**：复现 Yang et al. AsiaCCS 2021 时，借用了 Tu et al. CVPR 2020 的 sigmoid 重参数化 + 可微 LiDAR 渲染器。ASR 仅 0.1%（4/3720）。

**原因分析**：
- sigmoid 重参数化压缩梯度（σ'(x) 最大 0.25）
- 可微渲染器在光线命中/未命中边界有离散跳变，梯度噪声大
- 这两个设计来自 hiding attack（让检测消失），hiding 优化容易；appearing（从零生成检测）对梯度质量要求高得多

**做法**：去掉 mesh/renderer，直接优化 (N, 3) 点坐标，梯度链路 L → RPN → PointNet2 → input points，无中间噪声源。

**效果**：ASR 0.1% → **47.1%**（1752/3720）

---

## 2. 黑盒 PointOpt

**做法**：黑盒也从 mesh 参数化改为直接点优化，用 CMA-ES 搜索 (N×3) 维空间。配合 `CMA_diagonal` 模式、early termination、多 GPU 并行评估。

**效果**：运行时间从预估 5h 降至 ~30min（8 GPU），ASR 待测。

---

## 3. 物理约束引入（AAAI 2020 pipeline）

**目标**：让优化出的点云可以 3D 打印。参考 Tsai et al. AAAI 2020 的物理实现流程。

**新增三个可微正则项**（`attack/physical_constraints.py`）：
- **Chamfer distance** (`L_cd`)：限制点偏离初始形状（GT 车扫描），保持物理合理的外形
- **kNN smoothing** (`L_knn`)：惩罚 k 近邻距离的方差，保证点均匀分布（可打印表面）
- **Normal projection** (`L_nproj`)：用初始点 PCA 估计法向量，阻止点穿入物体内部

**集成方式**：在 `pointopt_loss` 中作为额外 loss 项，权重可配置。初始法向量预计算一次。

**效果**：ASR 从 47.1% 降至 44.7%（合理，搜索空间被约束了）。

---

## 4. 物理验证 Pipeline

**流程**（`run_attack.py --mode physical_verify`）：
1. 加载优化后的对抗点
2. Screened Poisson 表面重建（depth=8）
3. 去除面数 < 1024 的碎片连通分量
4. 从水密 mesh 均匀采样点云
5. 注入场景 → PointRCNN 检测 → 计算 ASR
6. 对比原始 ASR vs 重建后 ASR

**结果**：

| | 原始 ASR | Poisson 重建后 ASR | ConvexHull 重建后 ASR |
|---|---|---|---|
| 无物理约束 | 47.1% | 0.0% | — |
| 有物理约束 | 44.7% | 0.1% | 0.0% |

**问题**：重建后 ASR 全部归零。

---

## 5. 重建失败的根因分析

**现象**：Poisson 把 400 个稀疏点膨胀为 ~3m 球状包络面（原始尺寸 2.85×1.15×1.20m → 重建后 3.12×2.93×2.93m）。ConvexHull 更紧但同样失败。

**根因**：对抗效果**严重依赖每个点的精确坐标**，而非整体形状。

优化找到的 400 个点是一个精确的 1200 维"密码"，经过 PointNet2 的处理链：

```
FPS（选哪些点）→ Ball Query（怎么分组）→ PointNet（提特征）→ RPN（分类+回归）
```

每一步都对点的精确位置敏感。重采样后的点不在原来的位置上，密码就废了。

**与 AAAI 2020 的本质区别**：

| | AAAI 2020（分类攻击） | 我们（appearing attack） |
|---|---|---|
| 起点 | 已有 3D 物体（数千顶点） | 400 个散点从 GT 车初始化 |
| 任务 | 误分类（微小扰动） | 从零生成检测（需要大幅改变） |
| 扰动幅度 | 小（Chamfer 约束有效） | 大（优化需要自由度） |
| 重建误差 vs 扰动 | 重建误差 ≪ 扰动 → 保留 | 重建误差 ≫ 有效信号 → 失效 |

**关于 EoT 的理解**：
- 现有 EoT（多帧优化）让攻击对"放在哪个场景"鲁棒 → 整体平移没问题
- 但 400 个点的**内部相对坐标**在每次前向传播中完全不变 → 优化器找到了一个只对精确坐标有效的解
- 重采样改变了内部坐标 → 失效

---

## 6. 解决方案：点级别 EoT（noise_sigma）

**思路**：在优化过程中，每次前向传播时给对抗点加 N(0, σ²) 随机噪声。梯度仍然流向原始 `adv_points`（通过 `adv_points + noise`），但优化器被迫找到在 ±σ 范围内都有效的解。

**实现**：`_forward_one_frame` 中 `pts_for_inject = adv_points + torch.randn_like(adv_points) * noise_sigma`

**配置**：`attack.pointopt.physical.noise_sigma: 0.02`（2cm）

**预期**：
- 数字域 ASR 会下降（搜索空间被噪声约束）
- 重建后 ASR retention 显著提升（解不再依赖精确坐标）
- σ 越大，鲁棒性越强但 ASR 越低，需要调参

**状态**：待实验验证（noise_sigma 方案暂未跑通，先解决了采样策略问题）。

---

## 7. 关键发现：Closest Sampling（AAAI 2020 论文核心细节）

**背景**：重新精读 AAAI 2020 论文全文后，发现我们遗漏了重采样策略的关键区别。

论文 Section 4.4 明确使用了两种采样方式：
- **Closest**: 对每个对抗点，在重建 mesh 表面找最近的点（保留对抗位置）
- **Random**: 在 mesh 表面均匀随机采样

论文结果显示，对 car 类：
- Closest untargeted ASR: 95.3%
- Random untargeted ASR: **18.1%**
- Random target ASR: **0%**

> 原文: "The attack remains effective in the closest sampled point clouds... the success rates occasionally drop in the randomly sampled cases."

**我们之前的 physical_verify 全程使用了 uniform（random）采样 → 正好是 ASR 最差的方式。**

**修改**：在 `export_mesh.py` 新增 `closest_sample_points_from_mesh()`，用 Open3D RaycastingScene.compute_closest_points() 实现。`physical_verify` 支持 `--sample-method closest/uniform/both`。

**实验结果**（有物理约束的白盒 PointOpt checkpoint）：

| 采样方式 | ASR | Retention | 物体尺寸 |
|---|---|---|---|
| 原始 400 点（无重建） | **50.4%** (1874/3720) | — | 2.81 × 1.22 × 1.32 m |
| **Closest 采样** | **57.2%** (2127/3720) | **113.5%** | 2.83 × 1.22 × 1.26 m |
| Uniform 采样 | 0.0% (0/3720) | 0.0% | 3.02 × 2.96 × 2.75 m |

**关键数字**：
- Closest 采样后平均表面距离: 0.037 m，最大: 0.158 m
- Closest 采样的 ASR **反而提升**了 7 个百分点（113.5% retention）

**为什么 Closest > 原始**：
Poisson 重建生成了光滑表面，closest 采样将原始对抗点"吸附"到该表面上。微小的位移（均值 3.7cm）没有破坏对抗效果，反而因为点落在了更合理的表面上，部分边界帧从失败变为成功。

**为什么 Uniform 归零**：
Uniform 采样完全忽略对抗点的原始位置，在 3m 大小的膨胀包络面上随机撒点 → 400 个点失去了"密码"排列 → 检测器完全不响应。这与 AAAI 2020 论文的 car 类 random target 结果（0%）完全吻合。

**物理实现含义**：
- 3D 打印后 LiDAR 扫描到的点在 mesh 表面上 → 更接近 "closest" 还是 "uniform"？
- LiDAR 光线以固定角度扫描表面 → 返回点分布取决于表面朝向和距离 → 不是纯 uniform，但也不是 closest
- 实际效果取决于物体放置距离和角度，近距离（更多光线命中 → 更密集覆盖 → 更接近 closest）效果应该更好

---

## 文件命名规范（已更新）

| 输出文件 | 命名 |
|---|---|
| 白盒 checkpoint | `results/whitebox_pointopt_{latest/final}.pth` |
| 黑盒 checkpoint | `results/blackbox_pointopt_{latest/final}.pth` |
| 白盒 loss 曲线 | `results/whitebox_pointopt_loss.png` |
| 评估结果 | `results/asr_pointopt_results.json` |
| 物理验证 | `results/physical_verify_results.json` |
| 重建 mesh | `results/mesh_export/physical_reconstructed.obj` |

---

## 8. Box 初始化 + 1000 点实验

**问题**：v2（GT LiDAR 扫描初始化, 400 点）生成的对抗点云呈"两坨"分离形态，重建 mesh 为非闭合面片。原因：GT 扫描仅覆盖车辆可见面，不是完整 3D 模型，优化器可以自由地将点拆散。

**改动**：
- `init: gt` → `init: box`：从完整 3.9×1.6×1.56m 的 6 面 box 表面均匀采样，起点是封闭形状
- `n_points: 400` → `n_points: 1000`：更密集覆盖，给检测器更强信号
- 其他超参不变（物理约束权重、学习率等）

**结果**：

| 指标 | v2 (GT, 400pts) | v3 (Box, 1000pts) | 变化 |
|------|----------------|-------------------|------|
| 原始 ASR | 50.4% | **82.8%** | +32.4pp (绝对) |
| Closest ASR | 57.2% (113.5%) | **57.3%** (69.1%) | 绝对值持平 |
| Uniform ASR | 0.0% | 0.0% | 无变化 |
| rcnn_conf | 0.95 | 0.998 | 更高 |

**分析**：
- Box 初始化大幅提升了数字 ASR（82.8% vs 50.4%），因为完整封闭形状为优化器提供了更好的起点
- Closest 采样 ASR 绝对值基本持平（57.3% vs 57.2%），但保留率从 113.5% 降至 69.1%
- 保留率下降原因：1000 点在更大的 box 上变形后，Poisson 重建的表面偏差更大（mean_dist=0.028m, max_dist=0.34m），部分点投影后偏移过大
- `ref_car_feature.pt` 是 (256,1) 的 RPN 特征向量（用于 L_feat loss），不是点云数据，不能用于初始化
