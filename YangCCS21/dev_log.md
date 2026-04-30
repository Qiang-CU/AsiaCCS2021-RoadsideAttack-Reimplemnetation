# AsiaCCS 2021 Appearing Attack — 复现开发日志

## 1. 攻击策略总览

论文定义了两种攻击路线，针对不同 3D 目标检测器：

| | 白盒 (White-box) | 黑盒 (Black-box) |
|---|---|---|
| **目标模型** | PointRCNN | PointPillar / PV-RCNN |
| **优化方式** | 梯度反传 (Adam) | 遗传算法 (Genetic Algorithm) |
| **优化变量** | mesh vertices offset δ | mesh vertices (population) |
| **需要模型内部** | 是 (backbone features, RPN/RCNN logits) | 否 (仅 input→output) |
| **Loss** | L_cls + α·L_feat + β·L_box + γ·L_area | CW-like score + Laplacian + edge + normal |
| **代码入口** | `attack/whitebox.py` | `attack/genetic_attack.py` |

---

## 2. 白盒攻击 (PointRCNN)

### 2.1 优化变量与固定量

```
优化: mesh_param (δ)  — shape (V, 3), requires_grad=True
固定: v0 (icosphere base vertices), faces, object_pos
随机: LiDAR position (每步从 config range 采样)
```

- **不优化 translation**：论文优化的是 mesh shape perturbation δ，物体位置固定。如果优化位移，loss 可能通过"把物体挪到 detector 更敏感的位置"来作弊，而非通过形状本身成功，削弱物理复现意义。
- **物理约束**：每步 clamp mesh 到 `size_limit = [0.45, 0.45, 0.41]` 半轴范围内。

### 2.2 梯度链路

```
mesh vertices (v0 + δ)
    │
    ▼
LiDAR renderer (Möller-Trumbore ray-triangle intersection)
    │  交点坐标可微，hit/no-hit mask 离散但不切断梯度
    ▼
adv_pts (M, 3) — 数量可变
    │
    ▼ concat with ground points
    │
PointNet++ backbone (FPS → SA layers)
    │
    ├──▶ point_features ──▶ L_feat_backbone (与 ref car backbone 特征对齐)
    │
    ▼
PointHeadBox (RPN)
    │
    ├──▶ point_cls_logits ──▶ L_rpn_cls (top-K adversarial points 的 foreground logits)
    │
    ├──▶ batch_box_preds ──▶ proposal_layer (NMS top-100)
    │                              │
    │                              ▼ rois (位置 detach 于 NMS 选择操作)
    │                              │
    ▼                              ▼
backbone features ──────▶ STE ROI pooling (gather 操作保留梯度)
                                   │
                                   ▼
                           RCNN head (cls_layers, reg_layers)
                                   │
                                   ├──▶ rcnn_cls_preds ──▶ L_rcnn_cls
                                   ├──▶ rcnn_features ──▶ L_rcnn_feat (与 ref car RCNN 特征对齐)
                                   └──▶ rcnn_box_preds ──▶ L_box (orientation + size matching)

mesh vertices ──▶ L_area (maximize bottom face area)
```

### 2.3 Loss 函数 (Paper Eq. 10)

```
L = L_rcnn_cls
  + α_rpn · L_rpn_cls
  + α_feat · (L_rcnn_feat + L_backbone_feat)
  + β_box · L_box
  + γ_area · L_area
```

| Loss 项 | 含义 | 梯度来源 |
|---------|------|---------|
| `L_rcnn_cls` | CW-style: 推高 top-m ROI 的 Car confidence | RCNN head → STE ROI pooling → backbone |
| `L_rpn_cls` | CW-style: 推高 adversarial points 的 foreground logits (top-K) | RPN head → backbone (直接) |
| `L_rcnn_feat` | MSE: RCNN penultimate features vs precomputed ref car features | RCNN head → STE ROI pooling → backbone |
| `L_backbone_feat` | MSE: backbone features of adv points vs ref car backbone features | backbone (直接) |
| `L_box` | orientation (cosine) + size (MSE) matching to target car | RCNN head → STE ROI pooling → backbone |
| `L_area` | 最大化底部三角面面积 (物理稳定性) | 直接作用于 mesh vertices |

**注意**：白盒 loss 中**没有** Laplacian / edge / normal 正则化。这些只用于黑盒 genetic fitness。

### 2.4 Reference Features (precompute)

运行 `python run_attack.py --mode precompute` 生成：
- `ref_car_feature_backbone.pt` — 正常 car 的 PointNet++ backbone 特征均值
- `ref_car_feature_rcnn.pt` — 正常 car 的 RCNN penultimate 特征均值
- `ref_car_orientations.pt` — 正常 car 的 RPN 预测朝向均值
- `ref_car_box_size.pt` — 正常 car 的 RPN 预测 box 尺寸均值

这些通过自然推理（NMS 产生 proposal → 找距离目标位置 <3m 的 ROI）获取，不使用 forced ROI。

如果文件缺失，`L_feat` 和 `L_box` 会静默降级为 0 并打印 WARNING。

---

## 3. 黑盒攻击 (PointPillar / PV-RCNN)

### 3.1 遗传算法 (Paper Algorithm 1)

```
population = [icosphere + Gaussian noise] × 160
for each generation:
    evaluate fitness of each member (no gradient)
    selection: keep top 50%
    crossover: single-point vertex crossover
    mutation: Gaussian noise with adaptive stddev
    physical constraints: clamp to size_limit
```

### 3.2 Fitness 函数

```
fitness = avg_detection_score + 0.3 · detection_rate
        - ω1 · L_laplacian     (ω1 = 0.1)
        - ω2 · L_edge           (ω2 = 1.0)
        - ω3 · L_normal         (ω3 = 0.01)
```

- Detection score: 在多个随机 LiDAR viewpoints 下，mesh → ray trace → inject → PointPillar detect，取目标位置附近的最高 confidence
- Laplacian / edge / normal: mesh 正则化，防止退化为杂乱点云

### 3.3 评估流程

每个 population member 在 `n_eval_views` 个随机视角下：
1. Mesh → Möller-Trumbore ray tracing → adversarial points
2. Inject into ground scene
3. PointPillar forward (torch.no_grad)
4. 查找目标位置附近的检测结果

---

## 4. 工程取舍与原文对比

### 4.1 STE ROI Pooling (非原文，工程 workaround)

**问题**：PointRCNN 原始代码中 `roipool3d_gpu` 被 `torch.no_grad()` 包裹，`point_cls_scores` 被 `.detach()`。这导致 RCNN head 对 backbone features 没有梯度，无法将 `L_rcnn_cls` / `L_rcnn_feat` / `L_box` 的信号传回 mesh vertices。

**论文**：没有讨论 ROI pooling 的可微性问题，也没有提到 STE。

**我们的做法**：Monkey-patch `roipool3d_gpu` 为 STE 版本（`_ste_roipool3d_gpu`）：
- 去掉 `torch.no_grad()` 和 `.detach()`
- CUDA kernel 的 gather 操作用 STE 近似：forward 正常执行，backward 按 gather index 将 output gradient scatter 回 input features
- 通过 matching pooled xyz 到 input xyz 重建 gather indices（CUDA kernel 不暴露 indices）

**合理性**：如果 STE 梯度噪声过大或训练不稳定，可以降低 RCNN 相关 loss 的权重（alpha_feat, beta_box），或完全退回到只用 RPN loss 驱动。

### 4.2 NMS → Detach ROI 位置 (工程近似)

**论文**：说 "focus on top-m ROIs, m=100"，没有明确说 NMS 的梯度处理方式。

**我们的做法**：让 `proposal_layer` 正常执行 NMS → 产生 top-100 ROI。ROI 的位置/选择是 detach 的（NMS 是离散操作），但 STE ROI pooling 在这些位置上 gather 的 backbone features 保留梯度。

**之前的错误做法（已修复）**：曾经用 `forced_roi` 手动注入已知位置的 ROI，绕过 NMS。问题是训练时 RCNN 永远看到"正确"的 ROI，推理时 RPN/NMS 不一定产生这个 proposal。已改为自然 proposal 流程。

**梯度链路分析**：
- `L_box` 对 RCNN 预测的绝对坐标有梯度（通过 `roi_xyz` 加回），但 box refinement 的 decode 使用了 `local_rois.clone().detach()`，所以 orientation/size 的驱动主要来自 RCNN head 特征，而非 proposal 位置。这是 OpenPCDet 原始实现的行为。

### 4.3 Möller-Trumbore 硬射线求交 (符合原文精神)

**论文**：说 "line-plane intersection is differentiable"。

**实际情况**：
- ✅ 交点坐标可微：`vertices → edge vectors → t parameter → hit point`，计算图完整
- ✅ 交点坐标没有 `.detach()`
- ⚠️ hit/no-hit mask 是离散的：`u >= 0 && v >= 0 && u+v <= 1`，barycentric 边界不可微
- ⚠️ 最近面选择 (`argmin(t)`) 是离散的

**实际影响有限**：梯度通过选中的交点坐标正确流动。单条 ray 的 hit/no-hit 边界效应被多 ray（几十到上百个 hit points）平均掉。如果后续发现梯度质量不好，可考虑 soft boundary（SoftRas 风格），但论文没有这么做。

### 4.4 不优化 translation (修正原代码偏差)

**原代码**：同时优化 `mesh_param` (δ) 和 `translation_param` (3-dim 平移)。

**问题**：loss 可能通过移动物体到 detector 更敏感的位置来"成功"，而不是通过形状本身。这削弱物理复现意义。

**修正**：`translation_param` 固定为 `zeros(3)`，只优化 mesh shape δ。物体位置由 `object_pos` 固定，通过随机 LiDAR 位置模拟不同观测视角。

### 4.5 白盒不加 Laplacian (修正原代码偏差)

**原代码**：白盒 loss 包含 `λ·L_laplacian`。

**论文**：白盒物理约束只有 size limit 和 L_area。Laplacian / edge / normal 是黑盒 genetic fitness 的 mesh 正则化项。

**修正**：从 `appearing_loss` 中移除 `L_laplacian`。`L_laplacian` 函数保留在 `loss.py` 中供黑盒攻击使用。

---

## 5. 代码结构

```
attack/
├── whitebox.py          # 白盒主循环：mesh opt → ray trace → PointRCNN forward → loss → backward
├── whitebox_rpn.py      # RPN-only 白盒 (Stage 1 调试用)
├── loss.py              # 所有 loss 函数 + appearing_loss 组合
├── renderer.py          # Möller-Trumbore 可微 LiDAR renderer
├── mesh.py              # icosphere 生成 + adjacency
├── ground.py            # 确定性地面点生成 (ray tracing)
├── inject.py            # 点云注入 + overlap removal
├── genetic_attack.py    # 黑盒遗传算法 (targeting PointPillar)
├── hillclimb_attack.py  # 黑盒爬山法 (exploratory)
└── cmaes_attack.py      # 黑盒 CMA-ES (exploratory)

model/
├── pointrcnn_wrapper.py  # PointRCNN wrapper + STE ROI pooling patch
└── pointpillar_wrapper.py # PointPillar wrapper (黑盒用)

configs/
└── attack_config.yaml    # 统一配置

run_attack.py             # 主入口 (--mode whitebox/genetic/precompute/eval/...)
precompute_features.py    # 生成 ref car features (自然推理，无 forced ROI)
```

---

## 6. 运行命令

```bash
# 1. 预计算 reference features (必须先跑)
python run_attack.py --mode precompute --device cuda:0

# 2. 白盒攻击 (PointRCNN)
python run_attack.py --mode whitebox --device cuda:0

# 3. 黑盒攻击 (PointPillar, genetic algorithm)
python run_attack.py --mode genetic --device cuda:0

# 4. 评估
python run_attack.py --mode eval --gpu 0 --ckpt results/adv_mesh_whitebox_final.pth

# 5. Pose sweep (Table 3)
python run_attack.py --mode pose_sweep --ckpt results/adv_mesh_whitebox_final.pth
```
