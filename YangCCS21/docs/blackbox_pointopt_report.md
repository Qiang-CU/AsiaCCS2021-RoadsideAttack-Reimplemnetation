# Yang et al. (AsiaCCS 2021) 复现与改进报告

> **复现论文**: Yang et al., AsiaCCS 2021 — *Robust Roadside Physical Adversarial Attack Against Deep Learning in LiDAR Perception Modules*
>
> **攻击类型**: Appearing Attack（路边放置对抗物体，使检测器误检出幻影车辆）
>
> **目标模型**: PointRCNN（白盒）、PointPillar / PV-RCNN（原论文黑盒目标）

---

## 1. 原论文方法

Yang et al. (Kaichen Yang, Tzungyu Tsai, Honggang Yu, Max Panoff, Tsung-Yi Ho, Yier Jin) 的方法：

- **白盒**：假设可访问目标模型 PointRCNN 内部，操作 3D mesh 顶点生成对抗点云，通过梯度下降优化 mesh 形状
- **黑盒**：不访问模型内部，使用遗传进化算法 (genetic algorithm) 搜索对抗物体参数，目标模型为 PointPillar、PV-RCNN
- **物理验证**：对抗物体 3D 打印后放置路边，在 LGSVL 仿真 + Baidu Apollo 及实际道路测试中验证
- **LiDAR**：Velodyne VLP-16

原论文未开源代码。该团队在 AAAI 2020 发表的前置工作 "Robust Adversarial Objects against Deep Learning Models" (Tsai, Yang, Ho, Jin) 中，攻击 PointNet++ 分类器的方法是**直接扰动点云坐标**（C&W formulation + Chamfer distance + kNN smoothing），优化完后用 Poisson Surface Reconstruction 重建 mesh 再 3D 打印。**没有使用 sigmoid 重参数化，也没有可微分渲染器**。AsiaCCS 2021 大概率沿用了同一技术路线。

---

## 2. 复现实现及问题

### 2.1 复现方案

由于原论文未开源代码，复现时**没有沿用原团队的直接点扰动路线**，而是参照了 Tu et al. (CVPR 2020) 的可微 mesh 攻击框架。Tu et al. 的方法针对的是 hiding attack（让已有车辆检测消失），其核心组件：

- **sigmoid 重参数化**（来自 Tu et al.）：`vᵢ = R(b ⊙ sign(v̄₀ᵢ) ⊙ σ(|v̄₀ᵢ| + Δv̄ᵢ)) + c ⊙ tanh(t̄)`
- **可微 LiDAR 渲染器**：Möller-Trumbore 光线-三角形求交

复现时将 Tu et al. 的 hiding attack 框架移植到 appearing attack 场景。

### 2.2 复现结果

| 方案 | 优化算法 | ASR |
|------|----------|-----|
| Mesh 白盒 | Adam | **0.1%** (4/3720) |
| Mesh 黑盒 | CMA-ES | **0%** |
| Mesh 两阶段白盒 | Adam 分段 | **≈0%** |

### 2.3 失败原因

复现 ASR 极低的原因是**复现时引入的 sigmoid 重参数化 + 可微渲染器链路**，而非原论文方法本身的问题：

1. **sigmoid 重参数化压缩梯度**：σ'(x) 最大 0.25，初始球体顶点处约 0.20，梯度幅值直接缩小到 1/5；tanh 对平移梯度同样压缩
2. **可微渲染器梯度噪声大**：顶点微小移动可能导致光线从"命中"变为"未命中"（离散跳变），求交梯度在边界附近不稳定
3. **场景不匹配**：sigmoid 重参数化来自 Tu et al. 的 hiding attack，hiding 只需让已有检测置信度降到阈值以下，优化容易；appearing 需要从零生成高置信度检测，对梯度信号质量要求高得多

两者叠加：**梯度方向不准（渲染器噪声）+ 梯度幅值被压缩（sigmoid）→ 优化基本走不动。**

> 注：Tu et al. 自己也提到 *"In this black box setting, we find re-parameterization unnecessary for gradient-free optimization."*

---

## 3. 改进方案：直接点优化（PointOpt）

### 3.1 核心思路

回归原团队 AAAI 2020 的技术路线——**直接优化点坐标**，去掉 sigmoid 重参数化和可微渲染器，梯度从 RPN loss 经 PointNet2 骨干网络直达输入点坐标。

```
复现方案: δv, t → sigmoid 重参数化 → mesh 顶点 → 可微渲染器 → 点云 → 检测器 → Loss
改进方案: 点坐标 (N, 3) ──────────────────────────→ 注入场景 → 检测器 → Loss
```

### 3.2 白盒 PointOpt

| 项目 | 设计 |
|------|------|
| **优化变量** | N=1000 个点的 (x, y, z)，共 3000 维 |
| **优化器** | Adam, lr=0.01, CosineAnnealing 衰减 |
| **Loss** | L_cls (CW-margin) + L_loc (框中心对齐) + L_feat (特征拉近参考车辆) + L_uni (点均匀性正则) + L_cd (Chamfer distance) + L_knn (kNN smoothing) + L_nproj (法向量投影) |
| **梯度来源** | RPN 逐点输出（因 ROI Pooling 不可微，不用 RCNN 级别梯度） |
| **初始化** | 完整车型 Box 表面采样（6 面均匀覆盖，起点即封闭车型形状） |
| **约束** | bbox clamp [1.95, 0.8, 0.78]m + Chamfer distance + kNN smoothing + 法向量投影 |
| **点级 EoT** | 训练时给点坐标加 N(0, 0.02²) 高斯噪声，提高重采样鲁棒性 |

### 3.3 黑盒 PointOpt

与白盒相同的参数化，优化算法替换为 CMA-ES，不需要模型梯度。

| 项目 | 设计 |
|------|------|
| **优化变量** | N=200 个点的 (x, y, z)，共 600 维 |
| **搜索算法** | CMA-ES (CMA_diagonal 模式) |
| **适应度** | 注入 → PointRCNN 推理 → 注入位置附近检测置信度 + L2 正则 |
| **sigma0** | 0.15 |
| **popsize** | 48，多 GPU 并行 + per-candidate early termination |

### 3.4 对比

| | 复现方案 (Mesh + sigmoid + 渲染器) | 改进方案 (PointOpt) |
|--|--------------------------------------|---------------------|
| **搜索空间** | 间接（δv → sigmoid → 渲染 → 点云） | 直接（点坐标即搜索变量） |
| **梯度链路** | 穿过渲染器 + sigmoid，噪声大 + 衰减 | PointNet2 → 输入点，干净 |
| **初始化** | 零向量（标准球体） | GT 车辆扫描（类车形状） |
| **约束** | sigmoid 隐式（扭曲空间） | bbox clamp（线性，不扭曲） |

---

## 4. 速度优化（黑盒）

黑盒首版在 8 GPU 上每代约 85s，300 代需约 7 小时。优化后：

| 措施 | 效果 |
|------|------|
| popsize 128 → 48 | 推理量减少 62.5% |
| n_eval_samples 20 → 8 | 再减少 60% |
| per-candidate early termination | 劣质候选 2-3 帧即跳过，省 50-70% |
| 单帧 detect() 替代 detect_batch() | 消除 FPS 点云填充开销 |
| **总计** | 每代从 ~85s 降至 ~5s，总时间约 25-50 分钟 |

---

## 5. 实验结果

### 5.1 数字攻击 ASR

| 攻击方案 | 优化算法 | 参数化 | 初始化 | 点数 | ASR |
|----------|----------|--------|--------|------|-----|
| Mesh 白盒（复现，Tu et al. 框架） | Adam | sigmoid + 渲染器 → 点云 | 标准球体 | ~150 | 0.1% (4/3720) |
| Mesh 黑盒（复现） | CMA-ES | sigmoid + 渲染器 → 点云 | 标准球体 | ~150 | 0% |
| PointOpt 白盒 v1（GT 初始化） | Adam | 直接点坐标 | GT LiDAR 扫描 | 400 | 39.5% (1471/3720) |
| PointOpt 白盒 v2（+物理约束，GT初始化） | Adam | 直接点坐标 + 物理约束 | GT LiDAR 扫描 | 400 | 50.4% |
| **PointOpt 白盒 v3（Box 初始化 + 1000点）** | Adam | 直接点坐标 + 物理约束 | **Box 表面（封闭）** | **1000** | **82.8%** (3081/3720, 较 v2 绝对提升 +32.4pp) |

### 5.2 物理验证 ASR（Poisson 重建 → 重采样 → 再评估）

| 实验版本 | 原始 ASR | Closest 采样 | 保留率 | Uniform 采样 | 保留率 |
|----------|----------|-------------|--------|-------------|--------|
| v2 (GT 初始化, 400pts) | 50.4% | 57.2% | 113.5% | 0.0% | 0% |
| **v3 (Box 初始化, 1000pts)** | **82.8%** | **57.3%** | **69.1%** | 0.0% | 0% |

- **Closest 采样**：对每个原始对抗点，在重建 mesh 表面找最近点投影，平均投影距离 0.028m，最大 0.34m
- **Uniform 采样**：从 mesh 均匀随机采样，完全破坏原始点位空间排列，ASR 归零
- 重建 Mesh：Poisson depth=8，清理后 5555 顶点 10945 面，非水密（开曲面）

### 5.3 关键发现

1. **初始化方式**对 ASR 影响巨大：Box 封闭表面初始化（82.8%）远优于 GT LiDAR 部分扫描初始化（50.4%），因为 Box 提供了完整的 6 面覆盖，优化器起点更好
2. **点数增加**（400 → 1000）给检测器提供更密集的"车辆证据"，1000 点的 PointRCNN 响应显著更强（训练时 rcnn_conf 稳定 > 0.95）
3. **Closest 采样**是物理验证的关键——保留了原始对抗点的近似位置，ASR 保留率 69.1%
4. **Uniform 采样**完全失败（0.0%）——对抗效果严格依赖精确点位，随机重采样等于销毁"密码"

---

## 6. 物理实现 Pipeline

沿用 AAAI 2020 (Tsai, Yang, Ho, Jin) 的物理实现路线：

```
优化点云 (1000pts) → Screened Poisson 重建 → 去碎片 → Closest 重采样 → ASR 验证 → 3D 打印
```

### 6.1 优化中的物理约束

| 约束 | 公式 | 权重 | 作用 |
|------|------|------|------|
| Chamfer distance | CD(pts, pts_init) | λ_cd = 0.1 | 限制点不偏离初始形状太远 |
| kNN smoothing | kNN 距离方差 | λ_knn = 0.05 | 保证点间距均匀，避免离群点 |
| 法向量投影 | max(0, δ·n) | λ_nproj = 0.1 | 防止点被推入物体内部 |
| 点级 EoT | N(0, σ²), σ=0.02m | - | 训练时加位置噪声，提高重采样鲁棒性 |

### 6.2 重建与验证

1. **Screened Poisson Surface Reconstruction**（depth=8）
2. **去碎片**：删除面数 < 1024 的孤立连通分量
3. **Closest 采样**：对每个原始对抗点，找重建 mesh 表面最近点（Open3D RaycastingScene）
4. **验证**：注入重采样点云 → PointRCNN 检测 → 计算 ASR

---

## 7. 当前状态与后续方向

### 7.1 已完成
- 白盒 PointOpt 白盒攻击（Box 初始化, 1000pts, 物理约束）：ASR **82.8%**
- 物理验证 Pipeline（Poisson 重建 + Closest/Uniform 对比）
- 可视化（3D + BEV，成功样例标注检测框和置信度）

### 7.2 待改进
- **Closest 采样保留率 69.1%**：v2 实验中保留率 113.5%，v3 反而下降，可能因为 1000 点在 box 初始化后扰动幅度更大，重建曲面偏差更大（max_surface_dist = 0.34m）
- 增大物理约束权重（lambda_cd 0.1→1.0+, lambda_knn 0.05→0.5+）可能进一步提高重建保留率，但会牺牲数字 ASR
- 重建 mesh 仍非水密，需调参或后处理使其闭合，以满足 3D 打印要求
- L∞ 逐点扰动限制（如每点最大偏移 0.1-0.15m）有助于生成更紧凑的形状
