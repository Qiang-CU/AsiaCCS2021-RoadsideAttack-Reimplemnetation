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
│   ├── attack_config.yaml          # 默认配置（当前 mesh 主线 = offset+t）
│   └── experiments/               # 四组对照实验配置
├── data/
│   └── kitti -> ../../data/kitti    # 软链接到已有 KITTI 数据集
├── model/
│   ├── checkpoint/
│   │   └── pointrcnn_7870.pth      # 预训练 PointRCNN 权重
│   └── pointrcnn_wrapper.py        # OpenPCDet PointRCNN 封装
├── attack/
│   ├── mesh.py                     # 二十面体网格创建（仅 mesh 白盒用）
│   ├── renderer.py                 # 可微分 LiDAR 渲染器（仅 mesh 白盒用）
│   ├── reparameterize.py           # 旧 mesh baseline 的重参数化
│   ├── inject.py                   # BEV 占用图 + 空白区域采样 + 点云注入
│   ├── loss.py                     # L_cls + L_feat + L_loc + L_size + L_lap 损失函数
│   ├── whitebox.py                 # 白盒 mesh 攻击（支持 mesh_offset / reparameterize）
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
└── results/                        # 默认输出目录；实验配置会写到各自子目录
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
| **优化变量** | 网格顶点偏移 + 显式平移（可切回旧 reparameterize baseline） | 点坐标 (N, 3) 直接优化 | 点坐标 (N, 3) 直接搜索 |
| **渲染** | 可微分光线-三角形求交 → 点云 | 无渲染，直接注入 | 无渲染，直接注入 |
| **正则项** | L_lap (Laplacian smoothing) | L_uni (点均匀性) | L2 正则 |
| **梯度噪声** | 中高（默认 offset+t；旧 baseline 额外包含重参数化噪声） | 低（直接链路） | 无（不用梯度） |
| **是否需要模型内部访问** | 是（RPN 梯度） | 是（RPN 梯度） | 否（只看检测输出） |
| **ASR** | 0.1% (4/3720) | **47.1%** (1752/3720) | 待测 |

**结论**：当前仓库保留两条 mesh 白盒参数化：默认 `mesh_offset`（不使用重参数化）和旧 `reparameterize` baseline。Point-Opt 仍然是效果最强、最稳定的白盒路径；黑盒也已改用 Point-Opt 参数化。

---

## Mesh 参数化与实验矩阵

### Mesh 白盒两种参数化

- 默认 `configs/attack_config.yaml` 使用 `attack.mesh.param_mode: mesh_offset`
- 旧复现基线使用 `attack.mesh.param_mode: reparameterize`
- 新的 `mesh_offset` 形式为：`vertices = v0 + offset + t`
- 旧的 baseline 形式为：`delta_v, t_tilde -> reparameterize() -> vertices`

### LiDAR 命中点保存

`whitebox.py` 现在支持保存渲染器命中的对抗点云 `adv_pts`。配置位于：

```yaml
attack:
  mesh:
    save_hit_points:
      enabled: false
      when: [apply]   # apply / train / monitor / all
      subdir: debug_mesh_hits
      max_files: 32
```

保存的是 **LiDAR 真正打到 mesh 后得到的对抗点云**，不是整帧注入后的场景点云。

### 四组对照实验

建议直接使用下面四份配置：

```bash
configs/experiments/mesh_reparam_baseline.yaml
configs/experiments/pointopt_basic.yaml
configs/experiments/pointopt_physical.yaml
configs/experiments/pointopt_engineering.yaml
```

每份配置都带独立 `output.save_dir`，结果分别写到：

```bash
results/mesh_reparam_baseline/
results/pointopt_basic/
results/pointopt_physical/
results/pointopt_engineering/
```

---

## 完整运行流程

所有命令均在 `YangCCS21/` 目录下执行：

```bash
cd /mnt/file-206-user-disk-m/cpii.local/qli/carbu/YangCCS21
conda activate carbu
```

### Step 0 — 环境验证

```bash
# 验证模型推理
python run_attack.py --mode test_inference --device cuda:0
```

### Step 1 — 预计算参考车辆特征

```bash
python run_attack.py --mode precompute --device cuda:0
```

输出 `results/ref_car_feature.pt`，后续所有攻击都依赖此文件。

### Step 1.5 — 运行对照实验配置

下面四条命令覆盖当前实验矩阵：

```bash
# 旧 mesh + reparameterize baseline
python run_attack.py --mode whitebox \
    --config configs/experiments/mesh_reparam_baseline.yaml --device cuda:0

# pointopt-basic
python run_attack.py --mode pointopt \
    --config configs/experiments/pointopt_basic.yaml --gpu 0,1,2,3

# pointopt+physical
python run_attack.py --mode pointopt \
    --config configs/experiments/pointopt_physical.yaml --gpu 0,1,2,3

# pointopt+engineering
python run_attack.py --mode pointopt \
    --config configs/experiments/pointopt_engineering.yaml --gpu 0,1,2,3
```

如果你要跑默认的 **mesh 无重参数化主线**，直接使用：

```bash
python run_attack.py --mode whitebox --config configs/attack_config.yaml --device cuda:0
```

### Step 2 — 白盒 Point-Opt 优化（数字域 ASR 最大化）

不带物理约束，纯数字域优化，ASR 最高但不可打印：

```bash
# 先关闭物理约束（configs/attack_config.yaml 中 pointopt.physical.enabled: false）
python run_attack.py --mode pointopt --gpu 0,1,2,3,4,5,6,7
```

输出：
- `results/whitebox_pointopt_final.pth` — 优化后的对抗点
- `results/whitebox_pointopt_loss.png` — 损失曲线

### Step 3 — 评估 ASR

```bash
python run_attack.py --mode eval_pointopt --gpu 0,1,2,3,4,5,6,7 \
    --ckpt results/whitebox_pointopt_final.pth
```

输出 `results/asr_pointopt_results.json`。

### Step 4 — 带物理约束的 Point-Opt 优化（可打印物体）

开启 Chamfer + kNN smoothing + 法向量投影约束，生成可 3D 打印的对抗物体：

```bash
# 确保 configs/attack_config.yaml 中 pointopt.physical.enabled: true
# 当前默认已开启，关键权重：lambda_cd=0.1, lambda_knn=0.05, lambda_nproj=0.1

# 从头训练（推荐，物理约束从一开始就参与优化）
python run_attack.py --mode pointopt --gpu 0,1,2,3,4,5,6,7

# 或基于数字域 checkpoint warm-start（可能效果更好）
python run_attack.py --mode pointopt --gpu 0,1,2,3,4,5,6,7 \
    --ckpt results/whitebox_pointopt_final.pth
```

输出同 Step 2，checkpoint 中额外保存 `pts_init`（初始点，用于对比）。

> **提示**：运行前备份数字域 checkpoint：
> ```bash
> cp results/whitebox_pointopt_final.pth results/whitebox_pointopt_digital.pth
> ```

### Step 5 — 物理验证 Pipeline

验证对抗物体经 Poisson 表面重建 → 重采样后，ASR 是否保留：

```bash
python run_attack.py --mode physical_verify --gpu 0,1,2,3,4,5,6,7 \
    --ckpt results/whitebox_pointopt_final.pth \
    --mesh-method poisson
```

Pipeline 自动执行：
1. 加载优化后的对抗点
2. Screened Poisson 表面重建（depth=8）
3. 去除面数 < 1024 的碎片连通分量
4. 从水密 mesh 表面均匀采样点云
5. 注入场景 → PointRCNN 检测 → 计算 ASR
6. 对比原始 ASR vs 重建后 ASR

输出：
- `results/physical_verify_results.json` — 重建前后 ASR 对比 + ASR retention
- `results/adv_points_physical_resampled.pth` — 重建后采样的点云 checkpoint
- `results/mesh_export/physical_reconstructed.obj` — 水密 mesh（可直接 3D 打印）
- `results/mesh_export/physical_reconstructed.html` — 3D 交互查看器

### Step 6 — 导出可视化 / 3D 打印文件

```bash
# 导出 point-opt checkpoint 为 OBJ + PLY + HTML
python export_mesh.py --pointopt results/whitebox_pointopt_final.pth \
    --out-dir results/mesh_export --mesh-method poisson
```

### Step 7 — 黑盒攻击（可选）

```bash
python run_attack.py --mode blackbox --gpu 0,1,2,3,4,5,6,7
```

输出 `results/blackbox_pointopt_final.pth`。同样可用 Step 5 的 physical_verify 验证。

### Step 8 — 可视化攻击效果

```bash
# 只画攻击成功的帧（推荐）
python visualize_attack.py --ckpt results/whitebox_pointopt_final.pth \
    --gpu 0 --n-samples 10 --detect --bev --success-only

# 画所有帧（含失败帧，标题会标注 ✓ SUCCESS / ✗ FAIL）
python visualize_attack.py --ckpt results/whitebox_pointopt_final.pth \
    --gpu 0 --n-samples 10 --detect --bev
```

输出到 `results/`：
- `attack_3d_{sample_id}.png` — 3D 近景三面板图
- `attack_scene_{sample_id}.png` — BEV 俯视图（clean vs adversarial）

### Step 9 — Recall-IoU 曲线 / 防御评估（可选）

```bash
# Recall-IoU（mesh checkpoint）
python run_attack.py --mode recall_iou --gpu 0,1,2,3 \
    --ckpt results/adv_mesh_whitebox_final.pth

# 防御评估（kNN outlier removal + Gaussian noise）
python run_attack.py --mode defenses --gpu 0,1,2,3 \
    --ckpt results/adv_mesh_whitebox_final.pth
```

---

## 推荐完整执行顺序

```bash
# ① 环境验证 + 特征预计算
python run_attack.py --mode test_inference --device cuda:0
python run_attack.py --mode precompute --device cuda:0

# ② 数字域优化（不带物理约束，先拿到高 ASR baseline）
#    编辑 configs/attack_config.yaml: pointopt.physical.enabled: false
python run_attack.py --mode pointopt --gpu 0,1,2,3,4,5,6,7
python run_attack.py --mode eval_pointopt --gpu 0,1,2,3,4,5,6,7 \
    --ckpt results/whitebox_pointopt_final.pth
cp results/whitebox_pointopt_final.pth results/whitebox_pointopt_digital.pth

# ③ 带物理约束优化（可打印物体）
#    编辑 configs/attack_config.yaml: pointopt.physical.enabled: true
python run_attack.py --mode pointopt --gpu 0,1,2,3,4,5,6,7

# ④ 物理验证（Poisson 重建 → closest/uniform 重采样 → ASR 对比）
python run_attack.py --mode physical_verify --gpu 0,1,2,3,4,5,6,7 \
    --ckpt results/whitebox_pointopt_final.pth --mesh-method poisson \
    --sample-method both

# ⑤ 导出 3D 打印文件
python export_mesh.py --pointopt results/whitebox_pointopt_final.pth \
    --out-dir results/mesh_export --mesh-method poisson

# ⑥ 可视化攻击成功帧（原始点）
python visualize_attack.py --ckpt results/whitebox_pointopt_final.pth \
    --gpu 0 --n-samples 10 --detect --bev --success-only \
    --out-dir results/vis_original

# ⑦ 可视化物理重建后的攻击效果（closest 采样点）
python visualize_attack.py --ckpt results/adv_points_physical_closest.pth \
    --gpu 0 --n-samples 10 --detect --bev --success-only \
    --out-dir results/vis_physical_closest
```

---

## 物理约束说明（AAAI 2020 Pipeline）

配置位于 `configs/attack_config.yaml` → `attack.pointopt.physical`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 总开关 |
| `lambda_cd` | 0.1 | Chamfer distance — 限制点偏离初始形状的幅度 |
| `lambda_knn` | 0.05 | kNN smoothing — 惩罚点间距方差，保证表面均匀 |
| `lambda_nproj` | 0.1 | Normal projection — 阻止点穿入物体内部 |
| `knn_k` | 10 | kNN 操作的近邻数 |

**物理约束的作用**：

```
无约束 PointOpt → 优化出浮空散点 → Poisson 重建失败（ASR 0%）
带约束 PointOpt → 优化出贴合表面的点 → Poisson 重建保持形状 → ASR 保留
```

**loss 函数**：

```
L = L_cls + β_loc·L_loc + β_size·L_size + α_feat·L_feat + λ_uni·L_uni
  + λ_cd·L_cd + λ_knn·L_knn + λ_nproj·L_nproj
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                物理约束项（enabled: true 时激活）
```

---

## 实验结果汇总

### ASR 对比

| 攻击路径 | 迭代 | ASR | n_success / n_eligible |
|----------|------|-----|------------------------|
| Mesh 白盒 | 1000 | 0.1% | 4 / 3720 |
| **Point-Opt 白盒（无物理约束）** | 1500 | **47.1%** | 1752 / 3720 |
| **Point-Opt 白盒（带物理约束）** | 1500 | **50.4%** | 1874 / 3720 |
| Point-Opt 黑盒 (CMA-ES) | 300 代 | *待测* | — |

### 物理验证结果（Poisson 重建 → 重采样 → ASR）

| 采样方式 | ASR | Retention | 说明 |
|----------|-----|-----------|------|
| 原始 400 点（无重建） | 50.4% | — | 直接注入 |
| **Closest（AAAI 2020）** | **57.2%** | **113.5%** | mesh 表面最近点 |
| Uniform | 0.0% | 0.0% | mesh 表面随机撒点 |

> Closest 采样平均距离 3.7cm，最大 15.8cm，完全保留对抗性。

### 可视化输出

```bash
results/vis_original/attack_{3d,scene}_*.png          # 原始点
results/vis_physical_closest/attack_{3d,scene}_*.png  # closest 采样后
```

---

## 其他配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack.pointopt.n_points` | 400 | 白盒对抗点数 |
| `attack.pointopt.n_iters` | 1500 | point-opt 迭代数 |
| `attack.pointopt.lr` | 0.01 | point-opt 学习率 |
| `attack.pointopt.init` | gt | 初始化：gt（GT 车扫描）或 box（车表面采样） |
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
5. **遗留文件**：`attack/blackbox.py` 是原始隐藏攻击的代码，**不属于本 appearing attack 流水线**。黑盒攻击请使用 `attack/blackbox_appearing.py`。
6. **Mesh 白盒现支持双模式**：`configs/attack_config.yaml` 默认是 `mesh_offset`，`configs/experiments/mesh_reparam_baseline.yaml` 可回到旧 `reparameterize` baseline。
7. **Warm-start**：Point-Opt 支持 `--ckpt` 加载已有 checkpoint 继续训练。运行新 round 前建议备份旧 checkpoint。

### 初始化方案

默认 `init: gt`，两种方式：
- **gt**：`init_from_gt_cars` — 从 KITTI GT 车辆 bounding box 内提取 LiDAR 点云，去中心化后采样到 N=400 个点。起点即真实车辆扫描形状。
- **box**：`init_car_surface_points` — 在车辆尺寸长方体 (3.9×1.6×1.56m) 六面上均匀采样。