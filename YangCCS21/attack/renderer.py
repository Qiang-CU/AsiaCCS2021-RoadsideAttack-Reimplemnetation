"""
Differentiable LiDAR renderer using Möller-Trumbore ray-triangle intersection.

Gradient flows through mesh vertices → hit positions → adversarial point cloud.
"""

import math
import torch

from attack.mesh import get_aabb


# ---------------------------------------------------------------------------
# Möller-Trumbore ray-mesh intersection
# ---------------------------------------------------------------------------

def ray_mesh_intersect(rays_o, rays_d, vertices, faces):
    """
    Differentiable ray-mesh intersection (Möller-Trumbore).

    Args:
        rays_o:   (N, 3) ray origins
        rays_d:   (N, 3) unit direction vectors
        vertices: (V, 3) mesh vertices  (requires_grad=True for whitebox attack)
        faces:    (F, 3) LongTensor face indices

    Returns:
        t_min:    (N,) distance to nearest hit face (inf if no hit)
        hit_mask: (N,) bool – True if ray hits any face

    Memory: intermediate tensors are (N, F, 3); for N=5000, F=320 ≈ 18 MB
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0                # (F, 3)
    e2 = v2 - v0

    N = rays_o.shape[0]
    F = faces.shape[0]

    # Expand to (N, F, 3)
    d   = rays_d.unsqueeze(1).expand(N, F, 3)
    e1_ = e1.unsqueeze(0).expand(N, F, 3)
    e2_ = e2.unsqueeze(0).expand(N, F, 3)

    pvec    = torch.linalg.cross(d, e2_, dim=-1)               # (N, F, 3)
    det     = (e1_ * pvec).sum(-1)                              # (N, F)
    inv_det = 1.0 / (det + 1e-10)

    tvec    = rays_o.unsqueeze(1) - v0.unsqueeze(0)             # (N, F, 3)
    u       = (tvec * pvec).sum(-1) * inv_det                   # (N, F)

    qvec    = torch.linalg.cross(tvec, e1_, dim=-1)             # (N, F, 3)
    v       = (d * qvec).sum(-1) * inv_det                      # (N, F)
    t       = (e2_ * qvec).sum(-1) * inv_det                    # (N, F)

    hit = (
        (det.abs() > 1e-8) &
        (t > 1e-6) &
        (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    )

    t_masked = torch.where(hit, t, t.new_full((), float('inf')))
    t_min, _ = t_masked.min(dim=1)                              # (N,)
    hit_mask = t_min < float('inf')
    return t_min, hit_mask


# ---------------------------------------------------------------------------
# Velodyne HDL-64E ray sampling
# ---------------------------------------------------------------------------

def sample_rays_toward_mesh(sensor_pos, aabb_min, aabb_max,
                             n_elevation=64,
                             elev_min_deg=-24.9, elev_max_deg=2.0,
                             h_step_deg=0.08,
                             margin_deg=2.0,
                             device='cpu'):
    """
    Sample Velodyne HDL-64E rays whose directions point toward the mesh AABB.

    Args:
        sensor_pos:    (3,) tensor – LiDAR origin (usually zeros)
        aabb_min:      (3,) tensor
        aabb_max:      (3,) tensor
        n_elevation:   number of elevation rings (64)
        elev_min_deg:  lowest elevation angle (deg)
        elev_max_deg:  highest elevation angle (deg)
        h_step_deg:    horizontal angular resolution (deg)
        margin_deg:    extra margin around AABB angular extent
        device:        torch device string

    Returns:
        rays_o: (M, 3) – all equal to sensor_pos
        rays_d: (M, 3) – unit direction vectors
    """
    margin = math.radians(margin_deg)
    sp = sensor_pos.to(device)

    # 8 corners of AABB
    lo, hi = aabb_min.to(device), aabb_max.to(device)
    corners = torch.stack([
        torch.stack([lo[0], lo[1], lo[2]]),
        torch.stack([hi[0], lo[1], lo[2]]),
        torch.stack([lo[0], hi[1], lo[2]]),
        torch.stack([hi[0], hi[1], lo[2]]),
        torch.stack([lo[0], lo[1], hi[2]]),
        torch.stack([hi[0], lo[1], hi[2]]),
        torch.stack([lo[0], hi[1], hi[2]]),
        torch.stack([hi[0], hi[1], hi[2]]),
    ], dim=0)  # (8, 3)

    dirs = corners - sp.unsqueeze(0)
    az   = torch.atan2(dirs[:, 1], dirs[:, 0])
    el   = torch.atan2(dirs[:, 2],
                        torch.norm(dirs[:, :2], dim=1).clamp(min=1e-6))

    az_min, az_max = az.min().item() - margin, az.max().item() + margin
    el_min, el_max = el.min().item() - margin, el.max().item() + margin

    # Elevation channels
    all_el = torch.linspace(math.radians(elev_min_deg),
                             math.radians(elev_max_deg),
                             n_elevation, device=device)
    sel_el = all_el[(all_el >= el_min) & (all_el <= el_max)]
    if sel_el.numel() == 0:
        # Fallback: use all elevation channels
        sel_el = all_el

    # Horizontal angles
    h_step = math.radians(h_step_deg)
    n_h = max(int((az_max - az_min) / h_step) + 1, 1)
    sel_az = torch.linspace(az_min, az_max, n_h, device=device)

    elev, azim = torch.meshgrid(sel_el, sel_az, indexing='ij')
    elev = elev.reshape(-1)
    azim = azim.reshape(-1)

    rays_d = torch.stack([
        torch.cos(elev) * torch.cos(azim),
        torch.cos(elev) * torch.sin(azim),
        torch.sin(elev),
    ], dim=-1)   # (M, 3)

    rays_o = sp.unsqueeze(0).expand_as(rays_d)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Full rendering pipeline
# ---------------------------------------------------------------------------

def render_adversarial_points(vertices_world, faces, sensor_pos,
                               n_elevation=64,
                               elev_min_deg=-24.9, elev_max_deg=2.0,
                               h_step_deg=0.08,
                               margin_deg=2.0,
                               ray_batch_size=8192):
    """
    Render adversarial point cloud from a mesh placed in world coordinates.

    Args:
        vertices_world: (V, 3) tensor, requires_grad=True for whitebox attack
        faces:          (F, 3) LongTensor
        sensor_pos:     (3,) tensor – LiDAR origin
        ray_batch_size: int – process rays in chunks to limit peak memory

    Returns:
        adv_pts: (M, 3) tensor of adversarial hit points (differentiable w.r.t. vertices_world)
    """
    device = vertices_world.device
    sensor_pos = sensor_pos.to(device)
    faces = faces.to(device)

    with torch.no_grad():
        aabb_min, aabb_max = vertices_world.detach().min(dim=0).values, \
                              vertices_world.detach().max(dim=0).values

    rays_o, rays_d = sample_rays_toward_mesh(
        sensor_pos, aabb_min, aabb_max,
        n_elevation=n_elevation,
        elev_min_deg=elev_min_deg, elev_max_deg=elev_max_deg,
        h_step_deg=h_step_deg, margin_deg=margin_deg,
        device=device,
    )

    # Process rays in batches to manage peak memory
    all_pts = []
    N = rays_o.shape[0]
    for start in range(0, N, ray_batch_size):
        end = min(start + ray_batch_size, N)
        ro_b = rays_o[start:end]
        rd_b = rays_d[start:end]

        t_min, hit = ray_mesh_intersect(ro_b, rd_b, vertices_world, faces)
        if hit.any():
            pts = ro_b[hit] + t_min[hit].unsqueeze(-1) * rd_b[hit]
            all_pts.append(pts)

    if not all_pts:
        return vertices_world.new_zeros((0, 3))

    return torch.cat(all_pts, dim=0)   # (M, 3)
