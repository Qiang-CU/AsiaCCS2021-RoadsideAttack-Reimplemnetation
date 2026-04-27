# AsiaCCS'21 复现讨论记录

---

## 1. GPU 选择与迁移

选择 H20 单卡运行，理由：
- Hopper (sm_90) 在主流 CUDA 工具链中支持成熟，Blackwell (sm_100, 如 5090) 太新，spconv 可能有兼容问题
- 96GB 显存完全不用担心 OOM
- 迁移到 V100 步骤清晰：同版本 PyTorch/CUDA，只需重编译 CUDA extensions

核心约束：项目依赖 OpenPCDet 的自定义 CUDA 算子（PointNet2 的 ball_query、FPS、grouping），不同 GPU 架构 = 重新编译。

迁移方案：Docker 固定环境（PyTorch + CUDA + spconv 版本），H20 和 V100 各自编译 CUDA extensions，其余代码和 checkpoint 通用。

---

## 2. 重参数化：为什么存在 & 为什么要删

### 来源

来自 Tu et al. CVPR 2020（另一篇 LiDAR 对抗攻击论文），不是 AsiaCCS'21 原文的技术。

### 公式

```
v_i = R @ (b ⊙ sign(v0_i) ⊙ σ(|v0_i| + Δv_i)) + c ⊙ tanh(t̃)
```

### 直觉解释

icosphere 初始是个单位球，每个顶点在空间某个象限里（比如 x>0, y>0, z>0 = 右上前方）。

重参数化做两件事：
1. **`sign(v0)` 锁定象限** — 右上前方的顶点永远只能在右上前方移动，不能跑到左边或下边
2. **`sigmoid` 压幅** — 把移动范围压到 (0, b)，b 是半边长

最终效果：球只能变形为"星凸体"（从中心往任意方向看都能看到表面），类似圆角长方体。

### 和直接优化顶点坐标的区别

| | 直接优化 (mesh_offset) | 重参数化 (reparameterize) |
|--|--|--|
| 顶点能跑多远 | 任意位置，事后 clamp | 被 sigmoid 约束在 (0, b) 内 |
| 能否自交 | 能（右边的点跑到左边，面互相穿透） | 不能（象限锁定，拓扑保持） |
| 梯度质量 | 好（线性链路） | 差（sigmoid 饱和时梯度趋零） |
| 优化空间 | 大（形状自由） | 小（只能做星凸变形） |

### 为什么不需要它

重参数化解决的问题是"mesh 自交"。但实际上：
- 一个 162 顶点的粗糙 icosphere 很难自交
- 即使自交，对 LiDAR ray tracing 的影响微乎其微（ray 取最近交点）
- 原文 AsiaCCS'21 不用这个，直接 clamp 顶点到 45×45×41cm 范围内
- sigmoid 带来的梯度损失远大于防自交的收益

**结论：删除重参数化，统一用 mesh_offset + clamp。**

---

## 3. PointRCNN 梯度链路

### 架构

```
输入点云 → PointNet2 backbone → 逐点 cls_logits (N,) + 逐点 box_preds (N,7)
                                              ↓ NMS（不可微）
                                        proposals (K, 7)
                                              ↓ ROI Pooling（OpenPCDet: backward 未实现）
                                        RCNN head → final boxes + scores
```

### 原文 loss 公式

```
L_cls(x) = Σ_{i=1}^{m} k_i · [Z_bg(x)_i - Z_rpn(x)_i - Z_t(x)_i]
L_box(x, x_t) = d(φ_r(x), φ_r(x_t)) + d(Z_r(x), Z_r(x_t))
```

- Z_rpn = RPN objectness score（第一阶段）
- Z_bg, Z_t = RCNN 的 background 和 target class logit（第二阶段）
- k_i = 指示函数：target confidence < 0.9 时为 1
- m = 100（top-100 ROI）
- φ_r = RPN orientation, Z_r = RCNN refined orientation

**原文确实同时用了 RPN 和 RCNN 两阶段的输出。**

### 为什么 OpenPCDet 里 RCNN 梯度断了

梯度断裂不是因为"在优化模型"，而是梯度需要**穿过**冻结模型才能到达输入：

```
loss → RCNN 输出 → RCNN FC layers → ROI features → [断裂点] → backbone features → 输入点云 → mesh 顶点
```

OpenPCDet 的三个断裂点：
1. `proposal_layer` 用 `@torch.no_grad()` 装饰
2. `RoIPointPool3dFunction.backward()` 直接 `raise NotImplementedError`
3. `point_cls_scores.detach()` 在 roipool3d_gpu 中

原文大概率用的是 Shi et al. 2019 年的原始 PointRCNN 仓库，不是 OpenPCDet。原始仓库的 ROI pooling 实现可能有 backward 支持。`NotImplementedError` 是 OpenPCDet 统一重构时的工程选择，不是 PointRCNN 的本质限制。

### Mesh 攻击 0.1% ASR 的真正瓶颈

期望 ASR ~85%，实际 0.1%，差距主因**不在 RCNN 梯度**：

1. **Mesh 太粗** — 2 次细分 icosphere = 162 顶点 / 320 面。10-20m 距离上 LiDAR ray 能命中 30-80 个点，远少于 Point-Opt 的 400-1000 个
2. **硬 Möller-Trumbore 梯度不稳定** — hit/miss 边界处梯度不连续，微小顶点移动导致 ray 从命中变未命中，梯度剧烈震荡
3. **反面证据**：RPN-only 的 Point-Opt 拿到 47-50% ASR → RPN 级别 loss 本身够用，问题出在 mesh → renderer → points 梯度链路

---

## 4. 评估协议差异（关键）

### 原文评估 (Table 3)

原文数字域仿真不是跑 KITTI val 全集。做法是：
1. 固定对抗物体在 (4, -2, 0)
2. LiDAR 在预设范围内做 pose sweep（变换距离、角度）
3. 共 900 次扫描
4. 检测阈值 0.3，统计被误检为 vehicle 的比例
5. 结果：798/900 (88.7%)、754/900 (83.8%)、790/900 (87.8%) 等

### 当前代码评估

遍历 KITTI val 每一帧，每帧找空白区域注入，统计 eligible-frame ASR。

### 差异对比

| | 原文 Table 3 | 当前代码 |
|--|--|--|
| 对抗物体位置 | 固定 (4, -2, 0) | 每帧从 BEV 空白区随机采样 |
| LiDAR 视角 | 预设范围 sweep 900 次 | KITTI val 帧真实 ego pose |
| 场景内容 | 控制变量（可能是空场景/单一场景 + 对抗物体） | KITTI 真实复杂场景 + 注入 |
| 分母 | 900 次扫描 | ~3720 eligible frames |
| 典型结果 | 798/900 (88.7%) | 1752/3720 (47.1%) |

这两种评估测的东西不一样：
- **原文**：固定物体，测"不同视角下的鲁棒性"（角度/距离泛化）
- **当前**：固定物体形状，测"不同场景下的泛化性"（场景多样性）

### 数据集需求

- **KITTI** — 仍然需要。PointRCNN 在 KITTI 上训练，权重依赖 KITTI 的点云分布
- **NuScenes** — 不需要。原文不用 NuScenes，PointRCNN 也不是在 NuScenes 上训练的
- **LGSVL / Apollo** — 原文物理域实验用了 LGSVL 模拟器 + Apollo 栈，但属于物理域验证，当前 scope 排除
- **自建 LiDAR sweep** — 需要。要匹配原文 Table 3，需实现一个 LiDAR pose sweep 评估器（固定物体，sweep sensor pose，合成 ray tracing 点云）。这不依赖任何外部数据集，用我们已有的可微 LiDAR 渲染器即可实现

---

## 5. 当前代码与原文的完整偏差总结

| 维度 | 原文 AsiaCCS'21 | 当前代码 |
|------|-----------------|----------|
| 优化变量 | Mesh 顶点 | ❌ Point-Opt: 自由浮动点 / Mesh: 162 顶点太粗 |
| LiDAR 物理 | mesh → ray tracing → 点云 | ❌ Point-Opt 跳过渲染直接注入 |
| Loss 范围 | RPN + RCNN 两阶段 | ❌ 只用 RPN（RCNN 梯度断裂） |
| 物理约束 | 45×45×41cm clamp + L_area 底面面积 | ❌ 用了 Chamfer/kNN/normal projection（来自 AAAI'20） |
| 可打印 mesh | 优化过程中就是 mesh | ❌ 事后 Poisson 重建（原文明确反对这种做法） |
| 重参数化 | 无 | ❌ 保留了 CVPR'20 的 sigmoid/sign-locked（待删除） |
| 评估协议 | 固定物体位置 + LiDAR pose sweep 900 次 | ❌ KITTI val 全集 eligible-frame 遍历 |

---

## 6. 修复计划

### Phase 1：修复 Mesh 白盒攻击（短期）

目标：让 mesh → renderer → PointRCNN 的梯度链路 work，拿到 >10% ASR

- [ ] 删除重参数化代码，统一 mesh_offset + clamp
- [ ] 提高 mesh 分辨率到 3-4 次细分（642-2562 顶点）
- [ ] 替换物理约束为原文设定：45×45×41cm clamp + L_area
- [ ] 去掉 Chamfer/kNN/normal projection
- [ ] mesh 白盒 loss 暂时保持 RPN 级别

### Phase 2：匹配原文评估协议

目标：实现 LiDAR pose sweep 评估器，和 Table 3 对齐

- [ ] 实现 LiDAR pose sweep：固定物体在 (4, -2, 0)，ego 在预设范围采样
- [ ] 每个 pose 用可微 LiDAR 渲染器生成点云 → 注入空场景/简单场景 → 检测
- [ ] 统计 900 次扫描的误检率
- [ ] 保留当前 KITTI val 全集评估作为补充指标（测场景泛化性）

### Phase 3：恢复 RCNN 梯度

目标：实现原文完整 loss，ASR 接近论文报告值

- [ ] 用 PyTorch 原生操作替换 `RoIPointPool3dFunction`
- [ ] 移除 `@torch.no_grad()` 和 `.detach()` 断点
- [ ] 加入原文完整 loss：Z_bg - Z_rpn - Z_t + L_box

### Phase 4：端到端严格复现

```
Mesh 顶点优化 → 可微 LiDAR 渲染 → 点云注入 → PointRCNN (含 RCNN) → 原文完整 loss → 梯度回传到 mesh
```

### Point-Opt 的定位

Point-Opt 不是原文方法的复现，而是 diagnostic baseline / ablation：
- 用于验证 RPN 级别 loss 是否足够
- 用于快速验证攻击管线的其他组件（注入、评估等）
- 论文里可以报告为"upper bound without physical rendering constraint"

---

## 7. H20 服务器部署计划

### 服务器现状

```
GPU:        NVIDIA H20 96GB (sm_90, Hopper)
Driver:     580.76.05 (支持 CUDA ≤ 13.0)
PyTorch:    1.11.0+cu113 ← ❌ 不支持 sm_90，必须升级
CUDA TK:    11.3 ← 需要 12.x
spconv:     未安装
OpenPCDet:  未安装
Python:     3.8.10 (miniconda3)
KITTI:      /root/autodl-tmp/kitti_raw/ (7481帧, 27GB, 已解压)
旧工作:      /root/autodl-tmp/work/lidar/lidar_attack/ (CVPR复现)
已有包:      cma, open3d, scipy, shapely (pip)
磁盘:        /root ~745GB 可用
```

**核心问题：PyTorch 1.11+cu113 不兼容 H20 (sm_90)**。必须升级到 PyTorch 2.x + CUDA 12.x。

### 部署步骤

```bash
# ========== Step 1: 创建新 conda 环境 ==========
conda create -n asiaccs python=3.8 -y
conda activate asiaccs

# ========== Step 2: 安装 PyTorch 2.x + CUDA 12.4 ==========
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ========== Step 3: 安装 spconv ==========
pip install spconv-cu120   # spconv 2.x for CUDA 12

# ========== Step 4: 克隆代码 ==========
cd /root
git clone https://github.com/Qiang-CU/AsiaCCS2021-RoadsideAttack-Reimplemnetation.git
cd AsiaCCS2021-RoadsideAttack-Reimplemnetation

# ========== Step 5: 编译 OpenPCDet ==========
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
cd ..

# 验证
python -c "from pcdet.models import build_network; print('OpenPCDet OK')"

# ========== Step 6: 安装攻击代码依赖 ==========
pip install cma shapely scipy tqdm pyyaml open3d

# ========== Step 7: 设置 KITTI 数据软链接 ==========
cd YangCCS21
mkdir -p data
ln -s /root/autodl-tmp/kitti_raw data/kitti

# 验证目录结构
ls data/kitti/training/velodyne/ | wc -l   # 应该 7481
ls data/kitti/training/label_2/ | wc -l    # 应该 7481
ls data/kitti/ImageSets/                    # 应该有 train.txt, val.txt

# ========== Step 8: 下载 PointRCNN 预训练权重 ==========
mkdir -p model/checkpoint
# 从 OpenPCDet model zoo 下载 pointrcnn_7870.pth
# https://drive.google.com/file/d/... (具体链接见 OpenPCDet README)

# ========== Step 9: 更新配置路径 ==========
# configs/attack_config.yaml 中 pointrcnn_config 改为:
#   /root/AsiaCCS2021-RoadsideAttack-Reimplemnetation/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml

# ========== Step 10: 验证推理 ==========
conda activate asiaccs
cd /root/AsiaCCS2021-RoadsideAttack-Reimplemnetation/YangCCS21
python run_attack.py --mode test_inference --device cuda:0
```

### 从 H20 迁移到 V100×8 的步骤

同样的步骤，只是 Step 2 中 PyTorch 换成 V100 兼容版本（sm_70）：
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
# sm_70 (Volta) 在 PyTorch 2.4+cu124 中仍然支持
```
OpenPCDet 和 spconv 需要在 V100 上重新编译（`python setup.py develop`），其余代码和 checkpoint 通用。
