"""
Physical realizability constraints for adversarial point cloud optimization.

Implements the AAAI 2020 (Tsai et al.) / AsiaCCS 2021 (Yang et al.) approach:
  - Chamfer distance: limit perturbation from initial shape
  - kNN smoothing:    enforce uniform point spacing (printable surface)
  - Normal projection: prevent points from moving inward (non-physical surface)

All losses are differentiable and designed for use inside a PyTorch
gradient-based optimization loop.
"""

import torch
import torch.nn.functional as F


def chamfer_distance(pts, pts_ref):
    """
    Bidirectional Chamfer distance.

    CD(P, Q) = mean_{p in P} min_{q in Q} ||p-q||^2
             + mean_{q in Q} min_{p in P} ||q-p||^2

    Args:
        pts:     (N, 3) current adversarial points
        pts_ref: (M, 3) reference (initial) points

    Returns:
        loss: scalar tensor
    """
    diff = pts.unsqueeze(1) - pts_ref.unsqueeze(0)  # (N, M, 3)
    d2 = (diff * diff).sum(-1)                       # (N, M)
    loss_p2r = d2.min(dim=1).values.mean()
    loss_r2p = d2.min(dim=0).values.mean()
    return (loss_p2r + loss_r2p) / 2.0


def knn_smoothing_loss(pts, k=10):
    """
    kNN smoothing loss: encourage uniform spacing among k nearest neighbors.

    Penalizes variance of kNN distances so that no point is isolated or
    clumped.  This is critical for producing printable surfaces.

    Args:
        pts: (N, 3) tensor
        k:   number of nearest neighbors

    Returns:
        loss: scalar tensor
    """
    N = pts.shape[0]
    if N < k + 1:
        return pts.new_tensor(0.0)

    diff = pts.unsqueeze(0) - pts.unsqueeze(1)  # (N, N, 3)
    d2 = (diff * diff).sum(-1)                   # (N, N)
    d2 = d2 + torch.eye(N, device=pts.device) * 1e6

    topk_d2, _ = d2.topk(k, dim=1, largest=False)  # (N, k)
    topk_d = topk_d2.sqrt()

    mean_d = topk_d.mean(dim=1, keepdim=True)       # (N, 1)
    var_loss = ((topk_d - mean_d) ** 2).mean()

    return var_loss


def estimate_normals_knn(pts, k=10):
    """
    Estimate outward surface normals via local PCA on k nearest neighbors.

    The normal at each point is the eigenvector corresponding to the
    smallest eigenvalue of the local covariance matrix.  Normals are
    oriented to point away from the point cloud centroid.

    Args:
        pts: (N, 3) tensor (detached / no grad needed)
        k:   number of neighbors for local PCA

    Returns:
        normals: (N, 3) unit normals (detached)
    """
    N = pts.shape[0]
    with torch.no_grad():
        diff = pts.unsqueeze(0) - pts.unsqueeze(1)   # (N, N, 3)
        d2 = (diff * diff).sum(-1)
        d2 = d2 + torch.eye(N, device=pts.device) * 1e6
        _, knn_idx = d2.topk(k, dim=1, largest=False)  # (N, k)

        neighbors = pts[knn_idx]                       # (N, k, 3)
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)

        cov = torch.bmm(centered.transpose(1, 2), centered) / k  # (N,3,3)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        normals = eigenvectors[:, :, 0]                # smallest eigenvalue

        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms

        centroid = pts.mean(0)
        outward = pts - centroid
        flip = (normals * outward).sum(-1) < 0
        normals[flip] *= -1

    return normals


def normal_projection_loss(pts, pts_init, normals_init):
    """
    Penalize perturbation components that push points inward.

    For each point, the perturbation delta = pts - pts_init is projected
    onto the estimated outward normal.  Inward movement (negative dot
    product with normal) is penalized.

    Args:
        pts:           (N, 3) current points (has grad)
        pts_init:      (N, 3) initial points (detached)
        normals_init:  (N, 3) outward normals estimated from pts_init

    Returns:
        loss: scalar tensor
    """
    delta = pts - pts_init
    inward_component = -(delta * normals_init).sum(-1)  # positive = inward
    penalty = F.relu(inward_component)
    return penalty.mean()
