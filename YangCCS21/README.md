# Appearing Attack 复现 (AsiaCCS 2021)

> Yang et al., *Robust Roadside Physical Adversarial Attack Against Deep Learning in LiDAR Perception Modules*, AsiaCCS 2021

白盒攻击 PointRCNN，黑盒攻击 PointPillar / PV-RCNN。在空白路面区域放置优化后的 3D mesh，使 LiDAR 检测器产生幻影车辆检测。

技术细节和设计取舍见 [dev_log.md](dev_log.md)。

---

## 环境

```bash
# 依赖: PyTorch + CUDA, OpenPCDet (已编译), spconv
pip install cma shapely scipy tqdm pyyaml

# 编译安装 OpenPCDet
cd /root/AsiaCCS2021-RoadsideAttack-Reimplemnetation/OpenPCDet
pip install -e .

cd /root/AsiaCCS2021-RoadsideAttack-Reimplemnetation/YangCCS21
```

KITTI 数据：
```bash
ls data/kitti/training/{velodyne,label_2,calib}
ls data/kitti/ImageSets/{train,val}.txt
```

模型权重：
```bash
ls model/checkpoint/pointrcnn_7870.pth
ls model/checkpoint/pointpillar_7728.pth   # 黑盒攻击用
```

---

## 快速开始

```bash
# 1. 验证模型推理正常
python run_attack.py --mode test_inference --device cuda:0

# 2. 预计算 reference car features (白盒必须先跑)
python run_attack.py --mode precompute --device cuda:0

# 3. 白盒攻击 (PointRCNN)
python run_attack.py --mode whitebox --device cuda:0

# 4. 黑盒攻击 (PointPillar, genetic algorithm)
python run_attack.py --mode genetic --device cuda:0
```

---

## 所有运行模式

### 白盒攻击 (PointRCNN)

```bash
# 预计算 reference features — 生成 4 个 .pt 文件到 results/
python run_attack.py --mode precompute --device cuda:0

# 白盒 mesh 优化 (论文 Section 3.2)
python run_attack.py --mode whitebox --device cuda:0

# warm-start 从已有 checkpoint 继续
python run_attack.py --mode whitebox --device cuda:0 \
    --ckpt results/adv_mesh_whitebox_latest.pth
```

### 黑盒攻击 (PointPillar)

```bash
# 遗传算法 (论文 Section 3.3, Algorithm 1)
python run_attack.py --mode genetic --device cuda:0
```

### 评估

```bash
# ASR 评估 (多 GPU)
python run_attack.py --mode eval --gpu 0,1,2,3 \
    --ckpt results/adv_mesh_whitebox_final.pth

# Pose sweep — 论文 Table 3 协议 (30×30 = 900 poses)
python run_attack.py --mode pose_sweep --gpu 0 \
    --ckpt results/adv_mesh_whitebox_final.pth
```

### 导出

```bash
# 导出 mesh 为 .obj / .html
python export_mesh.py --ckpt results/adv_mesh_whitebox_final.pth \
    --out-dir results/mesh_export
```

---

## 配置 (`configs/attack_config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack.n_iters` | 5000 | 白盒优化迭代次数 |
| `attack.lr` | 0.003 | 学习率 (Adam) |
| `attack.mesh_subdivisions` | 3 | Icosphere 细分 (642 顶点, 1280 面) |
| `attack.size_limit` | [0.45, 0.45, 0.41] | 物理尺寸半轴限制 (cm) |
| `attack.n_views_per_step` | 4 | 每步随机 LiDAR 视角数 |
| `attack.grad_scale` | 10.0 | 梯度放大因子 |
| `loss_weights.alpha_rpn` | 0.1 | L_rpn_cls 权重 |
| `loss_weights.alpha_feat` | 0.1 | L_feat 权重 |
| `loss_weights.beta_box` | 0.1 | L_box 权重 |
| `loss_weights.gamma_area` | -0.01 | L_area 权重 (负值 = 最大化) |

---

## 目录结构

```
YangCCS21/
├── run_attack.py                  # 主入口
├── precompute_features.py         # 预计算 ref car features
├── export_mesh.py                 # 导出 mesh
├── configs/
│   └── attack_config.yaml         # 统一配置
├── attack/
│   ├── whitebox.py                # 白盒 mesh 优化 (PointRCNN)
│   ├── genetic_attack.py          # 黑盒遗传算法 (PointPillar)
│   ├── loss.py                    # Loss 函数 (L_cls / L_feat / L_box / L_area)
│   ├── renderer.py                # 可微 LiDAR ray tracing (Möller-Trumbore)
│   ├── mesh.py                    # Icosphere 生成 + 邻接表
│   ├── ground.py                  # 确定性地面点生成
│   └── inject.py                  # 点云注入 + overlap removal
├── model/
│   ├── pointrcnn_wrapper.py       # PointRCNN 封装 + STE ROI pooling patch
│   ├── pointpillar_wrapper.py     # PointPillar 封装 (黑盒用)
│   └── checkpoint/                # 预训练权重
├── evaluation/
│   ├── metrics.py                 # ASR 评估
│   ├── pose_sweep.py              # Pose sweep (论文 Table 3)
│   └── visualize.py               # 曲线绘制
├── utils/
│   ├── kitti_utils.py             # KITTI 数据加载
│   └── bev_iou.py                 # BEV IoU
├── dev_log.md                     # 技术细节 & 设计取舍
└── results/                       # 输出目录
```

---

## 注意事项

1. `attack_config.yaml` 中 `pointrcnn_config` / `pointpillar_config` 是绝对路径，迁移环境需更新
2. `precompute` 必须在 `whitebox` 之前运行，否则 L_feat / L_box 会静默降为 0
3. 白盒单帧约 4-6 GB 显存
4. `data/kitti` 是软链接，确保指向正确的 KITTI 数据目录
