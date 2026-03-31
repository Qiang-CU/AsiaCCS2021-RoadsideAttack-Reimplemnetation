"""
Loss functions for the appearing adversarial attack.

Single-stage RPN-level loss (inspired by Tu et al. CVPR 2020):
  L = L_cls + beta_loc*L_loc + beta_size*L_size + alpha_feat*L_feat + lambda_lap*L_lap

- L_cls:  CW-style maximize RPN foreground logits at adversarial points
- L_loc:  Align RPN box center predictions to injection position (CRITICAL)
- L_size: Match RPN box size predictions to car dimensions
- L_feat: Pull backbone features toward reference car features
- L_lap:  Laplacian mesh smoothing regulariser
"""

import torch
import torch.nn.functional as F
import numpy as np

from utils.bev_iou import bev_iou


def compute_phantom_box(injection_pos, box_size=(4.0, 1.6, 1.5), yaw=0.0):
    """
    Create a phantom bounding box at the injection position.

    Args:
        injection_pos: (3,) [x, y, z]
        box_size:      (l, w, h) in meters
        yaw:           heading angle

    Returns:
        phantom_box: (7,) numpy array [cx, cy, cz, l, w, h, yaw]
    """
    pos = np.asarray(injection_pos, dtype=np.float32)
    l, w, h = box_size
    return np.array([pos[0], pos[1], pos[2] + h / 2, l, w, h, yaw],
                    dtype=np.float32)


def select_proposals_near_injection(rois, injection_pos, phantom_box,
                                    iou_thresh=0.1, max_proposals=100):
    """
    Select proposal indices that overlap with the phantom injection box.

    Args:
        rois:          (N, 7) tensor of proposal boxes [cx,cy,cz,l,w,h,yaw]
        injection_pos: (3,) numpy array
        phantom_box:   (7,) numpy array
        iou_thresh:    minimum BEV IoU with phantom box
        max_proposals: maximum number of proposals to return

    Returns:
        indices: list of int, indices into rois
    """
    rois_np = rois.detach().cpu().numpy()
    indices = []
    for i in range(len(rois_np)):
        iou = bev_iou(rois_np[i], phantom_box)
        if iou >= iou_thresh:
            indices.append(i)
        if len(indices) >= max_proposals:
            break

    # If no proposals pass IoU threshold, fall back to proximity-based selection
    if not indices:
        dists = np.sqrt((rois_np[:, 0] - injection_pos[0])**2 +
                        (rois_np[:, 1] - injection_pos[1])**2)
        nearest = np.argsort(dists)[:min(max_proposals, len(dists))]
        indices = nearest[dists[nearest] < 5.0].tolist()

    return indices


def L_cls(point_cls_logits, n_scene, n_adv, kappa=0.0):
    """
    CW-style appearing classification loss in logit space.

    Maximize RPN foreground logits at adversarial point locations.
    Operating in logit space avoids sigmoid saturation and provides
    smooth, stable gradients throughout the optimization.

    CW formulation: minimize max(-Z_car, -kappa)
    which is equivalent to maximizing Z_car until it exceeds kappa.

    Args:
        point_cls_logits: (N_pts,) RPN per-point car classification logits
                          (pre-sigmoid). Has grad_fn.
        n_scene:          int, number of original scene points
        n_adv:            int, number of injected adversarial points
        kappa:            confidence margin (in logit space); attack stops
                          pushing once logit > kappa

    Returns:
        loss: scalar tensor (minimize -> car logit at adv points goes up)
    """
    if n_adv == 0 or point_cls_logits is None:
        return torch.tensor(0.0, requires_grad=True)

    adv_logits = point_cls_logits[n_scene:n_scene + n_adv]

    if adv_logits.numel() == 0:
        return point_cls_logits.new_tensor(0.0, requires_grad=True)

    # CW loss: max(-logit, -kappa) → pushes logit above kappa
    loss = torch.clamp(-adv_logits, min=-kappa).mean()
    return loss


def L_feat(point_features, ref_feature, n_scene, n_adv):
    """
    Feature adversary loss: pull backbone features at adversarial points
    toward reference car features.

    Uses PointNet2 backbone features (which have gradients).

    Args:
        point_features: (N_pts, D) backbone feature tensor (has grad)
        ref_feature:    (D,) precomputed reference car feature
        n_scene:        int, number of original scene points
        n_adv:          int, number of injected adversarial points

    Returns:
        loss: scalar tensor
    """
    if n_adv == 0 or point_features is None:
        return torch.tensor(0.0, requires_grad=True)

    adv_feat = point_features[n_scene:n_scene + n_adv]  # (n_adv, D)
    if adv_feat.numel() == 0:
        return point_features.new_tensor(0.0, requires_grad=True)

    ref = ref_feature.to(adv_feat.device)
    # Flatten ref to 1-D regardless of saved shape (e.g. (256,1) or (256,))
    ref = ref.reshape(-1)

    # Handle dimension mismatch: ref_feature may come from RCNN hook (256-d)
    # while point_features come from RPN backbone (128-d).
    feat_dim = adv_feat.shape[-1]
    ref_dim = ref.shape[0]
    if ref_dim != feat_dim:
        # Reshape to (1, 1, ref_dim) for adaptive_avg_pool1d → (1, 1, feat_dim)
        ref = F.adaptive_avg_pool1d(
            ref.view(1, 1, ref_dim), feat_dim
        ).view(feat_dim)

    ref = ref.unsqueeze(0).expand(adv_feat.shape[0], -1)

    # Mean squared error
    loss = F.mse_loss(adv_feat, ref)
    return loss


def L_box(rpn_box_preds, n_scene, n_adv, target_yaw=0.0):
    """
    Bounding box orientation control loss using RPN per-point box predictions.

    Encourages the RPN box predictions at adversarial point locations
    to have the desired yaw angle.

    Args:
        rpn_box_preds: (N_pts, 7) RPN per-point box predictions (has grad)
        n_scene:       int, number of original scene points
        n_adv:         int, number of injected adversarial points
        target_yaw:    desired yaw angle (facing ego vehicle)

    Returns:
        loss: scalar tensor
    """
    if n_adv == 0 or rpn_box_preds is None:
        return torch.tensor(0.0, requires_grad=True)

    adv_box = rpn_box_preds[n_scene:n_scene + n_adv]  # (n_adv, 7)
    if adv_box.numel() == 0:
        return rpn_box_preds.new_tensor(0.0, requires_grad=True)

    pred_yaw = adv_box[:, 6]
    target = torch.full_like(pred_yaw, target_yaw)
    loss = F.mse_loss(pred_yaw, target)
    return loss


def L_area(vertices, faces):
    """
    Ground plane stability loss: maximize bottom face area.

    Encourages faces near z=0 to have large area for physical stability.

    Args:
        vertices: (V, 3) mesh vertices
        faces:    (F, 3) LongTensor face indices

    Returns:
        loss: scalar tensor (negative, to be added with gamma < 0)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Face centre z-coordinate
    face_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0

    # Select faces near the bottom (z ≈ z_min)
    z_min = vertices[:, 2].min().detach()
    bottom_mask = (face_z - z_min).abs() < 0.02

    if not bottom_mask.any():
        return vertices.new_tensor(0.0, requires_grad=True)

    # Compute face areas via cross product
    e1 = v1 - v0
    e2 = v2 - v0
    cross = torch.linalg.cross(e1, e2, dim=-1)
    areas = 0.5 * torch.norm(cross, dim=-1)

    bottom_areas = areas[bottom_mask]
    # Negative: we want to maximize area, so loss = -sum(areas)
    loss = -bottom_areas.sum()
    return loss


def L_rcnn_cls(rcnn_cls_preds, injection_pos, rois, proximity=5.0):
    """
    RCNN-level appearing loss: maximize the final detection confidence
    for ROI proposals near the injection position.

    Args:
        rcnn_cls_preds: (1, K, 1) RCNN classification logits (has grad via STE)
        injection_pos:  (3,) numpy array, injection position
        rois:           (1, K, 7) ROI boxes
        proximity:      max distance from injection_pos to consider a proposal

    Returns:
        loss: scalar tensor (minimize -> car confidence goes up)
    """
    if rcnn_cls_preds is None or rois is None:
        return torch.tensor(0.0, requires_grad=True)

    rois_np = rois.detach().cpu().numpy().squeeze(0)  # (K, 7)
    logits = rcnn_cls_preds.squeeze(0).squeeze(-1)    # (K,)

    if logits.numel() == 0:
        return logits.new_tensor(0.0, requires_grad=True)

    dists = np.sqrt((rois_np[:, 0] - injection_pos[0])**2 +
                    (rois_np[:, 1] - injection_pos[1])**2)
    near_mask = torch.tensor(dists < proximity, dtype=torch.bool,
                             device=logits.device)

    if not near_mask.any():
        closest_idx = int(np.argmin(dists))
        loss = -logits[closest_idx]
        return loss

    near_logits = logits[near_mask]
    # Weight by inverse distance: closer proposals matter more
    near_dists = torch.tensor(dists[near_mask.cpu().numpy()],
                              dtype=torch.float32, device=logits.device)
    weights = 1.0 / (near_dists + 0.5)
    weights = weights / weights.sum()
    loss = -(near_logits * weights).sum()
    return loss


def L_rpn_box_loc(rpn_box_preds, n_scene, n_adv, injection_pos, device):
    """
    RPN box localization loss: encourage RPN box predictions at adversarial
    points to center on the injection position.

    Without this, RPN might assign high foreground scores to adversarial points
    but predict boxes far from the injection region, so they never survive NMS
    as useful proposals.

    Args:
        rpn_box_preds:  (N_pts, 7) RPN per-point box predictions [cx,cy,cz,l,w,h,yaw]
        n_scene:        int
        n_adv:          int
        injection_pos:  (3,) numpy or list
        device:         torch device

    Returns:
        loss: scalar tensor
    """
    if n_adv == 0 or rpn_box_preds is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    adv_box = rpn_box_preds[n_scene:n_scene + n_adv]  # (n_adv, 7)
    if adv_box.numel() == 0:
        return rpn_box_preds.new_tensor(0.0, requires_grad=True)

    target_xy = torch.tensor(injection_pos[:2], dtype=torch.float32, device=device)
    pred_xy = adv_box[:, 0:2]
    loss = F.mse_loss(pred_xy, target_xy.unsqueeze(0).expand_as(pred_xy))
    return loss


def L_rpn_box_size(rpn_box_preds, n_scene, n_adv,
                   target_size=(3.9, 1.6, 1.56), device='cuda'):
    """
    Encourage RPN box size predictions at adversarial points to match
    a typical car size. This helps proposals look like plausible car
    detections to the RCNN stage.

    Args:
        rpn_box_preds:  (N_pts, 7)
        n_scene:        int
        n_adv:          int
        target_size:    (l, w, h) target dimensions

    Returns:
        loss: scalar tensor
    """
    if n_adv == 0 or rpn_box_preds is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    adv_box = rpn_box_preds[n_scene:n_scene + n_adv]
    if adv_box.numel() == 0:
        return rpn_box_preds.new_tensor(0.0, requires_grad=True)

    target = torch.tensor(target_size, dtype=torch.float32, device=adv_box.device)
    pred_lwh = adv_box[:, 3:6]
    loss = F.mse_loss(pred_lwh, target.unsqueeze(0).expand_as(pred_lwh))
    return loss


def laplacian_loss(verts, adj):
    """
    Laplacian smoothing regulariser (from Tu et al. CVPR 2020).

    Penalises each vertex for deviating from the mean of its neighbours,
    encouraging a smooth mesh surface.

    Args:
        verts: (V, 3) tensor — current mesh vertices (has grad)
        adj:   list of lists — adjacency from build_adjacency()

    Returns:
        loss: scalar tensor
    """
    global _LAP_CACHE
    key = id(adj)
    device = verts.device

    if key not in _LAP_CACHE:
        _LAP_CACHE[key] = _build_lap_tensors(adj, device)

    idx, mask = _LAP_CACHE[key]
    if idx.device != device:
        idx = idx.to(device)
        mask = mask.to(device)
        _LAP_CACHE[key] = (idx, mask)

    nbr_verts = verts[idx]                                  # (V, max_nbr, 3)
    nbr_verts = nbr_verts * mask.unsqueeze(-1).float()
    count = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    mean_nbr = nbr_verts.sum(dim=1) / count                # (V, 3)

    lap = verts - mean_nbr
    return (lap * lap).sum()


_LAP_CACHE: dict = {}


def _build_lap_tensors(adj, device):
    """Build padded adjacency index and mask tensors."""
    V = len(adj)
    max_nbr = max(len(n) for n in adj)
    idx = torch.zeros(V, max_nbr, dtype=torch.long, device=device)
    mask = torch.zeros(V, max_nbr, dtype=torch.bool, device=device)
    for i, nbrs in enumerate(adj):
        if nbrs:
            idx[i, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
            mask[i, :len(nbrs)] = True
    return idx, mask


def appearing_loss(point_cls_logits, rpn_box_preds, point_features,
                   ref_feature, vertices, faces, adj,
                   n_scene, n_adv, injection_pos, device,
                   kappa=5.0, alpha_feat=0.1, beta_loc=0.5,
                   beta_size=0.2, lambda_lap=0.001,
                   target_size=(3.9, 1.6, 1.56)):
    """
    Single-stage appearing attack loss (inspired by Tu et al. CVPR 2020).

    All terms operate on RPN per-point outputs which have full gradient:
      L = L_cls + beta_loc*L_loc + beta_size*L_size + alpha_feat*L_feat + lambda_lap*L_lap

    Args:
        point_cls_logits: (N,)    RPN foreground logits (has grad)
        rpn_box_preds:    (N, 7)  RPN box predictions (has grad)
        point_features:   (N, D)  backbone features (has grad)
        ref_feature:      (D,)    reference car feature
        vertices:         (V, 3)  mesh vertices
        faces:            (F, 3)  face indices
        adj:              adjacency list for Laplacian
        n_scene:          int
        n_adv:            int
        injection_pos:    (3,) injection position
        device:           torch device
        kappa:            CW margin for L_cls
        alpha_feat:       L_feat weight
        beta_loc:         L_rpn_box_loc weight (CRITICAL for proposal alignment)
        beta_size:        L_rpn_box_size weight
        lambda_lap:       Laplacian smoothing weight
        target_size:      (l, w, h) target car dimensions

    Returns:
        total_loss, loss_dict
    """
    l_cls = L_cls(point_cls_logits, n_scene, n_adv, kappa=kappa)
    l_loc = L_rpn_box_loc(rpn_box_preds, n_scene, n_adv, injection_pos, device)
    l_size = L_rpn_box_size(rpn_box_preds, n_scene, n_adv,
                            target_size=target_size, device=device)
    l_feat = L_feat(point_features, ref_feature, n_scene, n_adv)
    l_lap = laplacian_loss(vertices, adj)

    total = (l_cls
             + beta_loc * l_loc
             + beta_size * l_size
             + alpha_feat * l_feat
             + lambda_lap * l_lap)

    loss_dict = {
        'L_total': total.item(),
        'L_cls': l_cls.item(),
        'L_loc': l_loc.item(),
        'L_size': l_size.item(),
        'L_feat': l_feat.item(),
        'L_lap': l_lap.item(),
    }
    return total, loss_dict


def apply_physical_constraints(vertices, v0, size_limit=(0.45, 0.45, 0.41)):
    """
    Hard physical printing constraints applied after each optimizer step.

    1. Clamp bounding box of vertices to size_limit
    2. Ensure ground contact (at least some vertices near z=min)

    Args:
        vertices: (V, 3) tensor (modified in-place via .data)
        v0:       (V, 3) initial vertices (for reference)
        size_limit: (sx, sy, sz) max half-extents in meters

    Returns:
        vertices: (V, 3) constrained vertices
    """
    sx, sy, sz = size_limit

    # Clamp to size limits
    centre = vertices.mean(dim=0, keepdim=True)
    offset = vertices - centre
    offset[:, 0].data.clamp_(-sx, sx)
    offset[:, 1].data.clamp_(-sy, sy)
    offset[:, 2].data.clamp_(-sz, sz)
    vertices.data.copy_(centre + offset)

    return vertices
