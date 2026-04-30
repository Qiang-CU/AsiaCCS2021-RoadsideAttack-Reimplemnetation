"""
Loss functions for AsiaCCS 2021 appearing attack.

White-box loss (paper Eq. 10):
  L = L_cls + α*L_feat + β*L_box + γ*L_area

where L_cls combines RPN objectness + RCNN target/background logits,
L_feat matches intermediate features to a real car reference,
L_box constrains predicted box orientation/size to a target vehicle,
L_area maximizes bottom face area for physical stability.

L_laplacian / L_edge / L_normal are used only in black-box genetic fitness,
not in white-box gradient optimization.
"""

import torch
import torch.nn.functional as F


# ── RCNN-level losses (primary signal, via forced ROI + STE) ────────────────

def L_rcnn_cls(rcnn_cls_preds, kappa=0.0):
    """
    CW-style loss on RCNN classification logits.
    Push the forced ROI's classification toward "Car" (logit > kappa).
    """
    if rcnn_cls_preds is None:
        return torch.tensor(0.0, requires_grad=True)

    logits = rcnn_cls_preds.view(-1)
    if logits.numel() == 0:
        return logits.new_tensor(0.0, requires_grad=True)

    return torch.clamp(kappa - logits, min=0.0).mean()


def L_rcnn_feat(rcnn_features, ref_rcnn_feature):
    """
    RCNN feature matching: make adversarial ROI's penultimate features
    resemble a real car's RCNN features.
    """
    if rcnn_features is None or ref_rcnn_feature is None:
        return torch.tensor(0.0, requires_grad=True)

    if rcnn_features.dim() >= 2:
        feat = rcnn_features.mean(dim=0)
    else:
        feat = rcnn_features

    ref = ref_rcnn_feature.to(feat.device).squeeze()

    if feat.shape != ref.shape:
        min_dim = min(feat.numel(), ref.numel())
        feat = feat.view(-1)[:min_dim]
        ref = ref.view(-1)[:min_dim]

    return F.mse_loss(feat, ref)


# ── RPN-level losses (auxiliary signal) ─────────────────────────────────────

def L_rpn_cls(point_cls_logits, n_scene, n_adv, kappa=0.0, topk=10):
    """
    CW-style on adversarial points' RPN foreground logits.
    Uses top-K logits: only push the K highest logits above kappa,
    since detection only needs a few high-scoring foreground points.
    """
    if point_cls_logits is None or n_adv == 0:
        return torch.tensor(0.0, requires_grad=True)

    n_total = point_cls_logits.shape[0]
    if n_scene + n_adv > n_total:
        n_scene = max(0, n_total - n_adv)

    adv_logits = point_cls_logits[n_scene:n_scene + n_adv]
    if adv_logits.numel() == 0:
        return adv_logits.new_tensor(0.0, requires_grad=True)

    k = min(topk, adv_logits.shape[0])
    top_logits = adv_logits.topk(k, largest=True).values
    return torch.clamp(kappa - top_logits, min=0.0).mean()


def L_feat_backbone(point_features, ref_backbone_feature, n_scene, n_adv):
    """Backbone feature matching for adversarial points."""
    if point_features is None or ref_backbone_feature is None or n_adv == 0:
        return torch.tensor(0.0, requires_grad=True)

    n_total = point_features.shape[0]
    if n_scene + n_adv > n_total:
        n_scene = max(0, n_total - n_adv)

    adv_feats = point_features[n_scene:n_scene + n_adv]
    if adv_feats.numel() == 0:
        return adv_feats.new_tensor(0.0, requires_grad=True)

    adv_mean = adv_feats.mean(dim=0)
    ref = ref_backbone_feature.to(adv_mean.device)

    if adv_mean.shape != ref.shape:
        min_dim = min(adv_mean.shape[0], ref.shape[0])
        adv_mean = adv_mean[:min_dim]
        ref = ref[:min_dim]

    return F.mse_loss(adv_mean, ref)


# ── Box matching loss (paper Eq. 10: L_box) ────────────────────────────────

def L_box(rcnn_box_preds, ref_orientation, ref_box_size):
    """
    Paper L_box: push predicted box orientation and size toward the target car.

    d(φ_r(x), φ_r(x_t)) + d(Z_r(x), Z_r(x_t))

    Args:
        rcnn_box_preds: (1, K, 7) RCNN refined box predictions [x,y,z,dx,dy,dz,ry]
        ref_orientation: scalar or (1,) — mean RPN orientation of real car
        ref_box_size: (3,) — mean RPN box size [dx,dy,dz] of real car
    """
    if rcnn_box_preds is None:
        return torch.tensor(0.0, requires_grad=True)

    preds = rcnn_box_preds.view(-1, 7)
    if preds.shape[0] == 0:
        return preds.new_tensor(0.0, requires_grad=True)

    pred_size = preds[:, 3:6]
    pred_ry = preds[:, 6]

    ref_sz = ref_box_size.to(preds.device).view(1, 3)
    l_size = F.mse_loss(pred_size, ref_sz.expand_as(pred_size))

    if isinstance(ref_orientation, (int, float)):
        ref_ry = preds.new_tensor(ref_orientation)
    else:
        ref_ry = ref_orientation.to(preds.device).view(-1)

    l_orient = (1.0 - torch.cos(pred_ry - ref_ry)).mean()

    return l_size + l_orient


# ── Regularization losses ──────────────────────────────────────────────────

def L_area(vertices, faces):
    """Paper Eq. 9: maximize bottom face area for physical stability."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    z_min = vertices[:, 2].min().detach()
    bottom_mask = (face_z - z_min).abs() < 0.02

    if not bottom_mask.any():
        return vertices.new_tensor(0.0, requires_grad=True)

    e1 = v1[bottom_mask] - v0[bottom_mask]
    e2 = v2[bottom_mask] - v0[bottom_mask]
    cross = torch.linalg.cross(e1, e2, dim=-1)
    areas = torch.norm(cross, dim=-1)

    return areas.sum()


_LAP_CACHE: dict = {}


def _build_lap_tensors(adj, device):
    V = len(adj)
    max_nbr = max(len(n) for n in adj)
    idx = torch.zeros(V, max_nbr, dtype=torch.long, device=device)
    mask = torch.zeros(V, max_nbr, dtype=torch.bool, device=device)
    for i, nbrs in enumerate(adj):
        if nbrs:
            idx[i, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
            mask[i, :len(nbrs)] = True
    return idx, mask


def L_laplacian(verts, adj):
    """Laplacian smoothing: penalizes non-uniform vertex spacing."""
    key = id(adj)
    device = verts.device

    if key not in _LAP_CACHE:
        _LAP_CACHE[key] = _build_lap_tensors(adj, device)

    idx, mask = _LAP_CACHE[key]
    if idx.device != device:
        idx = idx.to(device)
        mask = mask.to(device)
        _LAP_CACHE[key] = (idx, mask)

    nbr_verts = verts[idx]
    nbr_verts = nbr_verts * mask.unsqueeze(-1).float()
    count = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    mean_nbr = nbr_verts.sum(dim=1) / count
    lap = verts - mean_nbr
    return (lap * lap).sum()


# ── Combined loss ──────────────────────────────────────────────────────────

def appearing_loss(rcnn_cls_preds, rcnn_features, ref_rcnn_feature,
                   rcnn_box_preds, ref_orientation, ref_box_size,
                   point_cls_logits, point_features, ref_backbone_feature,
                   vertices, faces,
                   n_scene, n_adv,
                   alpha_rpn=0.1, alpha_feat=0.1, beta_box=0.1,
                   gamma_area=-0.01,
                   kappa_rcnn=0.0, kappa_rpn=0.0):
    """
    Paper Eq. 10: L = L_cls + α*L_feat + β*L_box + γ*L_area

    L_cls = L_rcnn_cls (RCNN target/bg logits) + α_rpn * L_rpn_cls (RPN objectness)
    L_feat = L_rcnn_feat (RCNN penultimate features) + L_feat_backbone (backbone features)
    L_box = orientation + size matching to target car
    L_area = maximize bottom face area (physical stability)
    """
    l_rcnn = L_rcnn_cls(rcnn_cls_preds, kappa=kappa_rcnn)
    l_rpn = L_rpn_cls(point_cls_logits, n_scene, n_adv, kappa=kappa_rpn)
    l_rcnn_f = L_rcnn_feat(rcnn_features, ref_rcnn_feature)
    l_backbone_f = L_feat_backbone(point_features, ref_backbone_feature,
                                   n_scene, n_adv)
    l_box = L_box(rcnn_box_preds, ref_orientation, ref_box_size)
    l_area = L_area(vertices, faces)

    total = (l_rcnn
             + alpha_rpn * l_rpn
             + alpha_feat * (l_rcnn_f + l_backbone_f)
             + beta_box * l_box
             + gamma_area * l_area)

    loss_dict = {
        'L_total': total.item(),
        'L_rcnn_cls': l_rcnn.item(),
        'L_rpn_cls': l_rpn.item(),
        'L_rcnn_feat': l_rcnn_f.item(),
        'L_backbone_feat': l_backbone_f.item(),
        'L_box': l_box.item(),
        'L_area': l_area.item(),
    }
    return total, loss_dict


def apply_physical_constraints(vertices, v0, size_limit=(0.225, 0.225, 0.205)):
    """Paper Eq. 7: clamp mesh size if axis exceeds threshold."""
    sx, sy, sz = size_limit
    centre = vertices.mean(dim=0, keepdim=True)
    offset = vertices - centre
    offset[:, 0].data.clamp_(-sx, sx)
    offset[:, 1].data.clamp_(-sy, sy)
    offset[:, 2].data.clamp_(-sz, sz)
    vertices.data.copy_(centre + offset)
    return vertices
