# LiDAR 对抗攻击复现：Appearing Attack (AsiaCCS 2021)

> **论文**: *Robust Roadside Physical Adversarial Attack Against Deep Learning in LiDAR Perception Modules* — Yang et al., AsiaCCS 2021
>
> **攻击类型**: Appearing Attack（在空白区域生成幻影车辆检测）
>
> **目标模型**: PointRCNN（白盒 + 黑盒）
>
> **评估范围**: KITTI val 数字域评估，含物理打印尺寸约束

---

## 目录结构

```
YangCCS21/
├── configs/
│   └── attack_config.yaml          # 全局攻击/模型/数据配置
├── data/
│   └── kitti -> ../../data/kitti    # 软链接到已有 KITTI 数据集
├── model/
│   ├── checkpoint/
│   │   └── pointrcnn_7870.pth      # 预训练 PointRCNN 权重
│   └── pointrcnn_wrapper.py        # OpenPCDet PointRCNN 封装
├── attack/
│   ├── mesh.py                     # 二十面体网格创建（仅 mesh 白盒用）
│   ├── renderer.py                 # 可微分 LiDAR 渲染器（仅 mesh 白盒用）
│   ├── reparameterize.py           # 网格顶点重参数化（仅 mesh 白盒用）
│   ├── inject.py                   # BEV 占用图 + 空白区域采样 + 点云注入
│   ├── loss.py                     # L_cls + L_feat + L_loc + L_size + L_lap 损失函数
│   ├── whitebox.py                 # 白盒 mesh 攻击（ASR 0.1%，已废弃）
│   ├── whitebox_pointopt.py        # 白盒 point-opt 直接点优化（推荐）
│   ├── blackbox.py                 # 原始黑盒 CMA-ES（隐藏攻击，遗留）
│   └── blackbox_appearing.py       # 黑盒 CMA-ES + PointOpt（直接点优化）
├── evaluation/
│   ├── metrics.py                  # ASR、Recall-IoU、防御评估（mesh）
│   ├── metrics_pointopt.py         # ASR 评估（point-opt / 黑盒 point-opt）
│   └── visualize.py                # Open3D 可视化 + 曲线绘制
├── utils/
│   ├── kitti_utils.py              # KITTI 数据加载、标定、标签解析
│   └── bev_iou.py                  # BEV IoU 计算 (Shapely)
├── precompute_features.py          # 预计算参考车辆特征向量
├── run_attack.py                   # 主入口：统一调度所有阶段
├── bench_blackbox.py               # 黑盒攻击速度基准测试
├── test_inference.py               # Phase 0 验证：模型推理
├── test_gradient.py                # Phase 1 验证：梯度链路
└── results/                        # 输出目录（检查点、曲线、JSON）
```

---

## 环境依赖

```bash
# 已在 carbu conda 环境中安装：
#   - Python 3.8+, PyTorch 1.x + CUDA
#   - spconv (用于 OpenPCDet)
#   - OpenPCDet (已编译安装至 /mnt/.../carbu/OpenPCDet)
#   - cma, shapely, scipy, tqdm, pyyaml, open3d (可选可视化)
conda activate carbu
```

---

## 关键架构决策：RPN 级别梯度流

PointRCNN 的梯度链路在 **ROI Head (PointRCNNHead)** 处断裂，因为 ROI Pooling
使用了不可微的 CUDA 索引操作。白盒攻击改为使用 **RPN (PointHeadBox) 级别的逐点输出**：

| 输出 | 形状 | 有梯度 | 说明 |
|------|------|--------|------|
| `point_cls_scores` | (N_pts,) | ✓ | RPN 逐点分类得分 |
| `point_features` | (N_pts, 128) | ✓ | PointNet2 骨干特征 |
| `rpn_box_preds` | (N_pts, 7) | ✓ | RPN 逐点框预测 |
| `rcnn_cls_preds` | (1, K, 1) | ✗ | RCNN 分类（仅用于评估） |

对抗点在合并点云中位于**末尾**，通过 `[n_scene : n_scene + n_adv]` 索引选取。

---

## 三条攻击路径对比

| | Mesh 白盒 (`whitebox.py`) | Point-Opt 白盒 (`whitebox_pointopt.py`) | Point-Opt 黑盒 (`blackbox_appearing.py`) |
|--|--------------------------|-----------------------------------|--------------------------------------|
| **优化算法** | Adam 梯度下降 | Adam 梯度下降 | CMA-ES 进化策略（无梯度） |
| **优化变量** | 网格顶点偏移 δv + 平移 t | 点坐标 (N, 3) 直接优化 | 点坐标 (N, 3) 直接搜索 |
| **渲染** | 可微分光线-三角形求交 → 点云 | 无渲染，直接注入 | 无渲染，直接注入 |
| **正则项** | L_lap (Laplacian smoothing) | L_uni (点均匀性) | L2 正则 |
| **梯度噪声** | 高（mesh 重参数化 + 光线求交） | 低（直接链路） | 无（不用梯度） |
| **是否需要模型内部访问** | 是（RPN 梯度） | 是（RPN 梯度） | 否（只看检测输出） |
| **ASR** | 0.1% (4/3720) | **39.5%** (1471/3720) | 待测 |

**结论**：Mesh 路径因渲染器梯度噪声导致 ASR 极低，已废弃。Point-Opt 白盒是推荐的白盒路径。黑盒也已改用 Point-Opt 参数化。

---

## 运行命令

所有命令均在 `YangCCS21/` 目录下执行：

```bash
cd /mnt/file-206-user-disk-m/cpii.local/qli/carbu/YangCCS21
conda activate carbu
```

### Phase 0 — 模型推理验证

```bash
python test_inference.py
```

预期输出：多数帧检测到 Car（conf > 0.5），hook feature 形状 `(K, 256, 1)`。

### Phase 1 — 梯度链路验证

```bash
python test_gradient.py
```

预期输出：`delta_v.grad.norm() > 0`，`t_tilde.grad.norm() > 0`，打印 `✓ Phase 1 PASSED`。

### Phase 3 — 预计算参考车辆特征

```bash
python run_attack.py --mode precompute --device cuda:0
```

或直接：

```bash
python precompute_features.py \
    --config configs/attack_config.yaml \
    --n_instances 500 \
    --device cuda:0 \
    --output results/ref_car_feature.pt
```

输出 `results/ref_car_feature.pt`（参考车辆特征向量）。

### Phase 5b — Point-Opt 直接点优化（推荐白盒路径）

跳过 mesh/renderer，直接优化注入点的 (x, y, z) 坐标。

**Round 1 — 从头训练**：

```bash
# 单卡
python run_attack.py --mode pointopt --device cuda:0

# 多卡并行（batch frames 分配到各 GPU 并行前向+反向，主卡聚合梯度）
python run_attack.py --mode pointopt --gpu 0,1,2,3
```

Round 1 结果：
- ASR = **39.5%** (1471/3720)

**Round 2 — Warm-start 精调**：

```bash
# 多卡
python run_attack.py --mode pointopt --gpu 0,1,2,3 \
    --ckpt results/adv_points_pointopt_final.pth
```

输出：
- `results/adv_points_pointopt_final.pth` — 优化后的对抗点
- `results/pointopt_loss.png` — 损失曲线

> **提示**：运行 Round 2 前请备份 Round 1 的 checkpoint：
> ```bash
> cp results/adv_points_pointopt_final.pth results/adv_points_pointopt_round1.pth
> ```

### Phase 6 — 黑盒攻击（CMA-ES + PointOpt，支持多 GPU）

黑盒攻击同样使用直接点优化参数化，用 CMA-ES 进化策略搜索最优点坐标。

```bash
# 单卡
python run_attack.py --mode blackbox --device cuda:0

# 多卡并行（推荐，CMA-ES 候选解分布到多卡评估）
python run_attack.py --mode blackbox --gpu 0,1,2,3,4,5,6,7
```

输出 `results/adv_points_blackbox_final.pth`。

> 多 GPU 原理：每代 `popsize` 个候选解按 round-robin 分配到各 GPU 上的 PointRCNN 副本并行评估，支持 per-candidate early termination 跳过差的候选。

黑盒关键配置（`attack_config.yaml`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blackbox.n_points` | 200 | 对抗点数（少于白盒 400，降低搜索维度） |
| `blackbox.popsize` | 48 | CMA-ES 种群大小 |
| `blackbox.n_eval_samples` | 8 | 每候选每代评估帧数 |
| `blackbox.sigma0` | 0.15 | CMA-ES 初始步长 |
| `blackbox.maxiter` | 300 | 最大代数 |
| `blackbox.init` | gt | 初始化方式：gt（GT 车扫描）或 box（车表面采样） |
| `blackbox.diagonal` | true | CMA_diagonal 模式，高维搜索更高效 |

### Phase 7 — 评估（支持多 GPU 并行）

**Point-Opt 路径评估**（白盒或黑盒 checkpoint 均可用）：

```bash
python run_attack.py --mode eval_pointopt --gpu 0,1,2,3 \
    --ckpt results/adv_points_pointopt_final.pth

# 黑盒 checkpoint 也用同样方式评估
python run_attack.py --mode eval_pointopt --gpu 0,1,2,3 \
    --ckpt results/adv_points_blackbox_final.pth
```

**Mesh 路径评估**（仅用于历史对比）：

```bash
python run_attack.py --mode eval --gpu 0,1,2,3 \
    --ckpt results/adv_mesh_whitebox_final.pth
```

输出：
- `results/asr_pointopt_results.json` / `results/asr_results.json`

---

## 实验结果汇总

### ASR 对比

| 攻击路径 | 迭代 | ASR | n_success / n_eligible |
|----------|------|-----|------------------------|
| Mesh 白盒 | 1000 | 0.1% | 4 / 3720 |
| **Point-Opt 白盒 Round 1** | 1000 | **39.5%** | 1471 / 3720 |
| Point-Opt 白盒 Round 2 | 2000 (warm) | *待测* | — |
| Point-Opt 黑盒 (CMA-ES) | 300 代 | *待测* | — |

---

## 配置说明

所有超参数集中在 `configs/attack_config.yaml`，关键项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack.n_iters` | 1000 | mesh 白盒优化迭代数 |
| `attack.lr` | 0.02 | Adam 学习率 |
| `attack.size_limit` | [1.2, 0.8, 0.7] | 物理打印尺寸约束 (m) |
| `attack.multi_frame_batch` | 8 | 每步采样帧数 |
| `attack.loss_weights.kappa` | 2.0 | CW margin |
| `attack.loss_weights.beta_loc` | 0.5 | L_loc 权重 |
| `attack.loss_weights.alpha_feat` | 0.1 | L_feat 权重 |
| `attack.pointopt.n_points` | 400 | 白盒对抗点数 |
| `attack.pointopt.n_iters` | 1500 | point-opt 迭代数 |
| `attack.pointopt.lr` | 0.01 | point-opt 学习率 |
| `attack.pointopt.lambda_uni` | 0.001 | 均匀性正则权重 |
| `attack.blackbox.n_points` | 200 | 黑盒对抗点数 |
| `attack.blackbox.popsize` | 48 | CMA-ES 种群大小 |
| `attack.blackbox.n_eval_samples` | 8 | 每候选评估帧数 |
| `eval.proximity_thresh` | 1.5 | ASR 判定距离 (m) |

---

## 注意事项

1. **路径**：`attack_config.yaml` 中的 `pointrcnn_config` 为绝对路径，迁移环境时需更新。
2. **显存**：白盒攻击单帧前向 + 反向约需 4–6 GB 显存；`multi_frame_batch` 过大时可能 OOM。
3. **数据**：KITTI 数据通过软链接引用，确保 `data/kitti/training/` 下有 `velodyne/`、`label_2/`、`calib/` 和 `ImageSets/` 目录。
4. **OpenPCDet 补丁**：已修改 `OpenPCDet/pcdet/datasets/__init__.py` 使非 KITTI 数据集导入变为可选，避免缺少 `av2` 等库时报错。
5. **遗留文件**：`attack/blackbox.py` 是原始隐藏攻击的代码，依赖不存在的 `attack.rooftop` 和 `models.pixor_net`，**不属于本 appearing attack 流水线**。黑盒攻击请使用 `attack/blackbox_appearing.py`。
6. **Mesh 白盒已废弃**：`whitebox.py` 和 `two_stage_attack.py` 都是基于 mesh 参数化的攻击，ASR 仅 0.1%，因可微渲染器梯度噪声过大导致优化失败。保留代码仅供参考。
7. **Warm-start**：Point-Opt 支持 `--ckpt` 加载已有 checkpoint 继续训练。运行新 round 前建议备份旧 checkpoint。


### 初始化方案
不是随机初始化。默认 init_mode = 'gt'，有两种初始化方式：
gt（默认）：init_from_gt_cars — 从 KITTI 数据集中提取真实 GT 车辆 bounding box 内的 LiDAR 点云（最多 50 辆车），去中心化后合并，随机采样到 N=400 个点。起点就是真实车辆的 LiDAR 扫描形状。
box（备选）：init_car_surface_points — 在车辆尺寸的长方体 (3.9×1.6×1.56m) 六个面上均匀采样点，模拟车表面形状。