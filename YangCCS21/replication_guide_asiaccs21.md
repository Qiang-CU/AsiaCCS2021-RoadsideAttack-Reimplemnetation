# Replication Guide: Robust Roadside Physical Adversarial Attack Against Deep Learning in LiDAR Perception Modules
**Paper**: AsiaCCS 2021 — Yang et al.  
**Target**: Appearing Attack against PointRCNN (white-box) + CMA-ES black-box  
**Scope**: Digital domain evaluation on KITTI val set; physical printing constraints included; Apollo/LGSVL excluded

---

## Repository Structure

```
lidar_attack_asiaccs/
├── configs/
│   └── attack_config.yaml
├── data/
│   └── kitti -> ../../lidar_attack/data/kitti   # symlink to existing KITTI
├── models/
│   └── pointrcnn_wrapper.py     # OpenPCDet hook + feature extractor
├── attack/
│   ├── mesh.py                  # REUSE from lidar_attack/attack/mesh.py
│   ├── renderer.py              # REUSE (adjust ray sampling origin)
│   ├── reparameterize.py        # REUSE (update size constraints)
│   ├── inject.py                # NEW: BEV occupancy + blank region sampler
│   ├── loss.py                  # NEW: L_cls + L_feat + L_box + L_area
│   ├── whitebox.py              # NEW: appearing attack loop
│   └── blackbox.py              # REUSE CMA-ES (new fitness function)
├── evaluation/
│   ├── metrics.py               # REUSE (update ASR definition)
│   └── visualize.py             # REUSE
├── utils/
│   ├── kitti_utils.py           # REUSE from lidar_attack/
│   └── bev_iou.py               # REUSE from lidar_attack/
├── precompute_features.py       # ONE-TIME: extract reference car features
├── run_attack.py
└── results/
```

**Setup**: Create symlink for KITTI data to avoid duplication:
```bash
cd lidar_attack_asiaccs/data
ln -s ../../lidar_attack/data/kitti kitti
```

---

## Phase 0 — Environment & OpenPCDet Setup

**Goal**: PointRCNN loads correctly, runs inference on KITTI val, produces reasonable detections.

### Tasks

1. Install OpenPCDet following its official README. Confirm `pcdet` is importable in your `carbu` environment.

2. Download pretrained PointRCNN weights from the OpenPCDet model zoo (KITTI Car class, BEV AP ~85%).

3. Write a minimal inference test:
```python
# test_inference.py
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import KittiDataset
import torch

# Load config + weights, run on 5 KITTI val frames, print detections
```

4. Confirm `get_proposals()` equivalent: in OpenPCDet's PointRCNN, the `RoIHeadTemplate.forward()` returns `batch_cls_preds` before NMS. Locate this in the source so you can access pre-NMS scores in the attack loop.

### Verification ✓
- `python test_inference.py` produces bounding boxes on ≥ 3 frames with confidence > 0.5
- BEV AP on first 100 val frames is within 5% of reported benchmark

---

## Phase 1 — Gradient Chain Validation (Critical Gate)

**Goal**: Confirm that gradients from PointRCNN's classification loss can flow back to injected point cloud coordinates. This is the go/no-go decision for the white-box design.

### Why this matters
PointRCNN uses PointNet++'s `ball_query` which is a discrete index-selection operation. The indices themselves have no gradient, but the **feature aggregation** (grouped xyz coordinates and max-pooling over gathered features) is differentiable. In practice, the `pointnet2_ops` used by OpenPCDet does support gradient flow through the aggregated features back to input coordinates — but this must be verified empirically, not assumed.

### Tasks

1. Write a focused gradient test:
```python
# test_gradient.py
import torch
from your renderer import render_adversarial_points
from your mesh import Icosphere

mesh = Icosphere()
delta_v = torch.zeros_like(mesh.vertices, requires_grad=True)

# Render ~60 points onto a flat plane at position (4, -2, 0) in LiDAR frame
adv_points = render_adversarial_points(mesh.vertices + delta_v, ...)

# Build a minimal KITTI-format input batch with just these points
# Forward through PointRCNN (no NMS), get max classification score
score = pointrcnn(batch)['batch_cls_preds'].max()
score.backward()

print("delta_v.grad norm:", delta_v.grad.norm().item())  # Must be > 1e-6
```

2. If gradient norm > 1e-6: proceed to Phase 2.

3. **Fallback if gradient is zero**: The ball_query in OpenPCDet may zero out gradients due to the CUDA implementation. In that case, adopt a straight-through estimator: detach the ball_query indices but allow gradients through the gathered feature values. This requires a small monkey-patch of `pointnet2_utils.ball_query`. Document this patch clearly.

### Verification ✓
- `delta_v.grad.norm() > 1e-6` with a meaningful scene (not an all-zero point cloud)
- Gradient is non-uniform across vertices (not a uniform constant, which would indicate gradient leakage)

---

## Phase 2 — Scene Injection Pipeline

**Goal**: For each KITTI val frame, automatically find a valid blank region and inject the rendered adversarial point cloud into the original scene.

### Tasks

**2a. BEV Occupancy Map** (`attack/inject.py`):

Build a BEV occupancy grid per frame. Mark cells occupied by:
- All GT bounding boxes (with 0.5m margin)
- Raw point cloud cells with more than 2 points

```python
def build_bev_occupancy(points, gt_boxes, 
                         x_range=(2, 35), y_range=(-15, 15), 
                         resolution=0.5):
    """Returns a 2D boolean grid: True = occupied."""
```

**2b. Blank Region Sampler**:

```python
def sample_injection_position(occupancy_grid, n_candidates=20):
    """
    Sample a random free cell (x, y) from the occupancy grid.
    Apply a minimum clearance of 1.5m radius around the chosen cell.
    Returns (x, y, z=0) in LiDAR frame.
    Falls back to a fixed fallback position (4, -2, 0) if no free cell found.
    """
```

**2c. Injection**:

```python
def inject_points(scene_points, adv_points, injection_pos):
    """
    Translate adv_points to injection_pos.
    Merge with scene_points.
    Optionally: remove original points inside the adversarial object's AABB
    to avoid physical overlap artifacts.
    Returns merged point cloud.
    """
```

**2d. Record injection metadata per frame**:
```python
{
    "frame_id": "000042",
    "injection_pos": [4.2, -1.8, 0.0],
    "injection_valid": True,   # False if fell back to fixed pos
    "gt_box_count": 3
}
```

### Verification ✓
- On 10 random val frames, visualize occupancy grid + sampled position + injected points (Open3D or matplotlib BEV)
- Confirm injected points do not overlap with any GT bounding box
- Confirm merged point cloud looks physically plausible (no floating objects)

---

## Phase 3 — Model Hook & Reference Feature Precomputation

**Goal**: Hook the RCNN head's penultimate layer in OpenPCDet's PointRCNN; precompute a reference feature vector from real KITTI car instances.

### Tasks

**3a. Identify the hook point** in OpenPCDet's PointRCNN:

In `pcdet/models/roi_heads/pointrcnn_head.py`, the RCNN classification branch is:
```
ROI-pooled features → shared_fc_layers → cls_layers[:-1] → [HOOK HERE] → cls_layers[-1] → logit
```

Register a forward hook on `cls_layers[-2]` (the layer before the final linear).

```python
# models/pointrcnn_wrapper.py
class PointRCNNWrapper:
    def __init__(self, model):
        self.model = model
        self.ref_feature = None
        self._hook = model.roi_head.cls_layers[-2].register_forward_hook(
            self._capture_feature
        )
    
    def _capture_feature(self, module, input, output):
        self.last_feature = output  # shape: (N_rois, D)
    
    def get_proposals_with_features(self, batch_dict):
        """Forward pass, returns (cls_preds, self.last_feature)"""
```

**3b. Precompute reference car features** (`precompute_features.py`):

```python
# For each car instance in KITTI train split:
#   - Crop the point cloud to the GT bounding box (with 0.2m margin)
#   - Run a forward pass with that crop as a "fake scene"
#   - Capture the hook feature
# Average over N=500 car instances to get ref_feature (shape: D,)
# Save to results/ref_car_feature.pt
```

Why average: the paper says "feature vectors generated by a normal car model." Using the mean over many instances makes the reference robust and reduces dependence on a specific pose or distance.

### Verification ✓
- Hook successfully captures a feature tensor with shape `(N_rois, D)` during a normal forward pass
- `ref_car_feature.pt` shape is `(D,)`, values are finite (no NaN/Inf)
- Cosine similarity between features of two different car instances > 0.7 (sanity check that the feature space is meaningful)

---

## Phase 4 — Loss Functions

**Goal**: Implement the four loss terms. Each must be unit-tested independently before combining.

### Loss Term Definitions

**L_cls** — Appearing classification loss (maximize car detection confidence):

The goal is the opposite of the hiding attack. For the top-m RPN proposals near the injection position:

```python
def L_cls(cls_preds, rpn_scores, injection_pos, m=100, gamma=0.9):
    """
    For proposals with IoU > 0.1 with the injection bounding box:
    - Maximize their car classification score
    - Weight by RPN objectness score
    Loss = -sum_i( rpn_score_i * log(car_score_i) )
    Only sum over proposals where car_score < gamma (already-confident ones don't need more push)
    """
```

Note: The injection bounding box is a fixed-size "phantom box" at the injection position, sized as a typical KITTI car (4.0m × 1.6m × 1.5m, yaw = 0 pointing toward sensor).

**L_feat** — Feature adversary (pull adversarial features toward reference car features):

```python
def L_feat(adv_feature, ref_feature, alpha=0.001):
    """
    adv_feature: (N_proposals, D) — captured from hook for proposals near injection
    ref_feature: (D,) — precomputed reference
    Loss = mean over proposals of L2(adv_feature_i, ref_feature)
    """
```

**L_box** — Bounding box orientation control (make detected box face the ego vehicle):

```python
def L_box(predicted_orientations, rpn_orientations, target_orientation, beta=0.001):
    """
    target_orientation: yaw angle pointing toward ego (typically pi or 0 depending on placement side)
    Loss = L2(pred_orientation, target_orientation) for proposals above rpn confidence threshold
    """
```

This term is optional for the digital evaluation (ASR doesn't require correct orientation), but include it for completeness as it reflects the paper's core threat model.

**L_area** — Ground plane stability (physical printing constraint):

```python
def L_area(vertices, bottom_faces, gamma=-0.001):
    """
    Maximize the total area of faces on the z=0 plane (bottom of object).
    This ensures the physical object can stand stably.
    Only vertices with z ≈ 0 (within 0.02m) contribute.
    Loss = gamma * sum_of_bottom_face_areas  (negative because we maximize)
    """
```

**Combined loss**:
```python
L_total = L_cls + alpha * L_feat + beta * L_box + gamma * L_area
```

### Physical Printing Constraints

Applied at every optimizer step (hard constraint, not a loss term):

```python
def apply_physical_constraints(vertices, size_limit=(0.45, 0.45, 0.41)):
    """
    1. Clamp bounding box of vertices to size_limit (in meters)
    2. Ensure at least one face group is approximately co-planar with z=min (ground contact)
    Only x,y coordinates of bottom vertices are modified during optimization.
    """
```

### Verification ✓
- Each loss term returns a scalar tensor with `requires_grad=True`
- With random delta_v, L_cls > 0 and L_area < 0 (signs are correct)
- Manually verify: after 50 steps optimizing only L_cls on a single frame, `score.backward()` shows decreasing L_cls
- Physical constraint: after 200 steps, object bounding box stays within 0.45×0.45×0.41m

---

## Phase 5 — Two-Stage White-box Attack Loop

**Goal**: Full appearing attack on KITTI val set using two-stage optimization.

### Why Two Stages?

From `diagnose_big.png` we learned that single-stage RPN-only attack fails at the **proposal generation** bottleneck:
- RPN per-point foreground scores increase (logit -3.2 → -2.2) ✓
- But zero proposals appear near injection position ✗
- Therefore RCNN output stays at zero confidence ✗

**Root cause analysis** — PointRCNN's proposal pipeline has two non-differentiable gates:

1. `proposal_layer` is decorated with `@torch.no_grad()` → `rois` have no gradient
2. `RoIPointPool3dFunction.backward()` raises `NotImplementedError`
3. `point_cls_scores.detach()` in `roipool3d_gpu`

Our STE patch handles #2 and #3, but **not #1** (NMS-based proposal selection is inherently discrete). The two-stage strategy works around this:

- **Stage 1**: Optimize RPN outputs (foreground scores + box center/size predictions) until proposals naturally appear near the injection position. No STE needed.
- **Stage 2**: Once proposals exist, enable STE and add RCNN classification loss to directly maximize final detection confidence.

### NMS Configuration (from `pointrcnn.yaml`)

```
RPN → Proposal (NMS_CONFIG.TEST):
  NMS_PRE_MAXSIZE:  9000    top-k before NMS
  NMS_POST_MAXSIZE: 100     keep after NMS
  NMS_THRESH:       0.85    IoU overlap threshold
  SCORE_THRESH:     (none)  ← NO score filtering! All points are candidates.

RCNN → Final Detection (POST_PROCESSING):
  SCORE_THRESH:     0.1
  NMS_THRESH:       0.1
```

Key insight: RPN proposal generation does NOT filter by score. It takes the top-9000 points by score, runs NMS with IoU=0.85, and keeps 100 boxes. So the bottleneck is that adversarial points' predicted boxes don't survive NMS (box center/size too far from injection position), not that scores are too low.

### Stage 1 Loss Design

```
L_stage1 = L_cls + β_loc * L_rpn_box_loc + β_size * L_rpn_box_size + α * L_feat + γ * L_area
```

- `L_cls`: CW-style maximize foreground logits at adversarial points
- `L_rpn_box_loc` (NEW): Pull RPN box center predictions toward injection position
- `L_rpn_box_size` (NEW): Match predicted box size to KITTI mean car [3.9, 1.6, 1.56]
- `L_feat`: Pull backbone features toward reference car features
- `L_area`: Physical stability

### Stage 2 Loss Design

```
L_stage2 = w_rcnn * L_rcnn + w_rpn * L_cls + β_loc * L_loc + α * L_feat + γ * L_area
```

- `L_rcnn` (PRIMARY): Maximize RCNN classification logits for proposals near injection
- `L_cls` (AUXILIARY): Maintain RPN foreground scores to keep proposals alive
- STE enabled: gradients flow through ROI pooling via `_RoIPointPool3dSTE`

### Auto-Transition Logic

Stage 1 → Stage 2 when proposals consistently appear near injection:
- Check every 50 steps
- Need ≥1 proposal within 3m of injection position
- Must pass 3 consecutive checks

### Usage

```bash
# Full two-stage attack
python two_stage_attack.py --device cuda:0

# Quick single-frame diagnostic (200+100 iters)
python two_stage_attack.py --device cuda:0 --single_frame

# Warm-start Stage 2 from existing Stage 1 checkpoint
python two_stage_attack.py --device cuda:0 --warm_start results/adv_mesh_whitebox_stage1_final.pth
```

### Verification ✓
- Stage 1: Proposals appear near injection position (n_proposals_near > 0)
- Stage 2: RCNN confidence at injection position exceeds 0.3
- Multi-frame: Loss curve shows convergence, ASR > 80% on val set

---

## Phase 6 — Black-box Attack (CMA-ES)

**Goal**: Generate adversarial objects without access to PointRCNN gradients. Reuse the existing CMA-ES implementation with a new fitness function. Target models: PointPillar and PV-RCNN.

### Reuse Strategy

The CMA-ES structure in `lidar_attack/attack/blackbox.py` only needs the fitness function replaced. The mesh parameterization, physical constraints, and injection pipeline are identical to the white-box attack.

### New Fitness Function

```python
def fitness(delta_v_flat, mesh, injection_pos, scene_points, model, config):
    """
    CW-style appearing attack objective (to be minimized):
    
    f(x) = max( max_{i != 'car'} Z_i(x) - Z_car(x), -kappa )
    
    where Z are logits from the target model's classification head,
    x is the injected point cloud.
    
    kappa = 0 (default confidence margin)
    
    Total fitness = f(x) + c * ||delta_v||_2 + w1*L_lap + w2*L_edge + w3*L_normal
    (geometry regularizers same as CVPR 2020 black-box, already in codebase)
    """
```

### CMA-ES Configuration

```python
cma_config = {
    "sigma0": 0.05,
    "popsize": 160,
    "maxiter": 1000,
    "tolfun": 1e-6,       # stop if fitness change < 1e-6 for 10 gens
}
```

CMA-ES is strictly better than the paper's simple genetic algorithm: it adapts the covariance structure of the search distribution, converges faster, and requires far fewer hand-tuned hyperparameters.

### Target Models for Black-box

Both are available in OpenPCDet with pretrained KITTI weights:
- `PointPillar` (used by Baidu Apollo 6.0 — the paper's original black-box target)
- `PV-RCNN` (stronger baseline, good for transferability evaluation)

The mesh optimized against one model can be evaluated against the other to measure transferability — this replicates Table 2 from the paper at low additional cost.

### Verification ✓
- CMA-ES fitness curve decreases over 1000 iterations
- After optimization, PointPillar detects a car at the injection position with score > 0.3
- Wall-clock time per optimization run < 2 hours on RTX 3090

---

## Phase 7 — Evaluation & Defense

**Goal**: Report ASR and Recall-IoU curves for both white-box and black-box attacks. Evaluate existing defenses and the paper's proposed density-based defense.

### ASR Definition for Appearing Attack

A frame is counted as **attack success** if and only if:
1. Before attack: no detection within 1.5m of injection position (IoU < 0.1 with any output box)
2. After attack: at least one detection within 1.5m of injection position with confidence > 0.3

```python
def compute_asr(dataset, adv_mesh, model, injection_positions):
    """
    For each frame:
    - Run model on original scene → check condition 1
    - Run model on injected scene → check condition 2
    - Count successes
    Returns: ASR (float), per-frame results (list)
    """
```

### Recall-IoU Curve (equivalent for Appearing Attack)

For a range of IoU thresholds [0.1, 0.2, ..., 0.7], compute the fraction of injected frames where the model produces a detection with IoU > threshold with the phantom injection box.

This is the direct analog of the Recall-IoU curve from CVPR 2020, adapted to the appearing attack.

### Defense Evaluation

Implement and evaluate three defenses:

**Defense 1: kNN Outlier Removal** (from existing literature, evaluated in paper):
```python
def knn_outlier_removal(points, k=5, alpha=0.1):
    """Remove points where mean kNN distance > mu + alpha * sigma."""
```
Expected result: minimal impact on ASR (paper shows ~2% reduction).

**Defense 2: Random Gaussian Noise**:
```python
def gaussian_noise_defense(points, sigma2=0.01):
    """Add N(0, sigma2) noise to all point coordinates."""
```
Expected result: ASR drops ~25% at sigma2=0.01 but benign detection also degrades.

**Defense 3: Density-based SVM Detector** (paper's proposed defense):
```python
def density_defense(predicted_boxes, points, lidar_pos, r=0.35):
    """
    For each predicted bounding box:
    1. Find the point nearest to the LiDAR within the box
    2. Count points within sphere of radius r around that point
    3. Feed count to a pre-trained SVM to classify: real vehicle vs. adversarial object
    
    SVM training: use real KITTI car instances (positive) and 
    adversarial object scans (negative) from the attack results.
    """
```
Expected result: ~98% accuracy at separating real vehicles from adversarial objects.

### Expected Results Summary

| Method | Model | Sim ASR | Notes |
|--------|-------|---------|-------|
| White-box | PointRCNN | ~85-88% | With all loss terms |
| Black-box (CMA-ES) | PointPillar | ~70-84% | Paper reports 70-85% |
| Black-box (CMA-ES) | PV-RCNN | ~75-80% | Paper reports 77% |
| White-box + kNN defense | PointRCNN | ~80-85% | Minimal reduction |
| White-box + Gaussian noise σ²=0.01 | PointRCNN | ~55-65% | ~25% reduction |
| Density SVM defense | — | ~1-5% ASR | Near-perfect detection |

### Verification ✓
- White-box ASR on KITTI val ≥ 80%
- Black-box ASR on PointPillar ≥ 70%
- Density defense accuracy ≥ 95%
- All result plots saved to `results/`

---

## What to Reuse vs. Build New — Quick Reference

| Component | Action | Source |
|-----------|--------|--------|
| `attack/mesh.py` | **Reuse as-is** | `lidar_attack/attack/mesh.py` |
| `attack/renderer.py` | **Reuse, minor edit** | Update ray sampling origin from rooftop to ground-level injection pos |
| `attack/reparameterize.py` | **Reuse, update params** | Change `b` to `(0.45, 0.45, 0.41)`, `c` to `(0.1, 0.1, 0.0)` |
| `attack/blackbox.py` (CMA-ES) | **Reuse, new fitness fn** | Replace `hiding_fitness` with `appearing_fitness` |
| `utils/kitti_utils.py` | **Reuse as-is** | `lidar_attack/utils/kitti_utils.py` |
| `utils/bev_iou.py` | **Reuse as-is** | `lidar_attack/utils/bev_iou.py` |
| `evaluation/visualize.py` | **Reuse as-is** | `lidar_attack/evaluation/visualize.py` |
| `evaluation/metrics.py` | **Reuse, update ASR def** | Flip condition: appearing not hiding |
| `models/voxelizer.py` | **Do NOT use** | PointRCNN doesn't voxelize |
| `models/pixor_net.py` | **Do NOT use** | Replaced by OpenPCDet PointRCNN |
| `attack/rooftop.py` | **Do NOT use** | Replaced by `inject.py` |
| `attack/loss.py` | **Rewrite from scratch** | Completely different loss structure |
| `attack/whitebox.py` | **Rewrite from scratch** | Appearing attack, different loop structure |
| `models/pointrcnn_wrapper.py` | **Build new** | OpenPCDet hook interface |
| `attack/inject.py` | **Build new** | BEV occupancy + blank region sampler |
| `precompute_features.py` | **Build new** | Reference car feature extraction |

---

## Dependency Notes

No new pip packages required beyond what `lidar_attack` already uses, assuming OpenPCDet is installed. Confirm:

```bash
python -c "from pcdet.models import build_network; print('OpenPCDet OK')"
python -c "import cma; print('CMA-ES OK')"   # pip install cma if missing
```
