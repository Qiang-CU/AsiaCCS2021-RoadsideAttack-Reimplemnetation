"""
Vertex reparameterization for the adversarial mesh.

Maps unconstrained optimisation variables (delta_v, t_tilde) to bounded
world-frame vertices, preventing self-intersection and large deformations.

Formula (per vertex i):
    v_i = R @ (b ⊙ sign(v0_i) ⊙ σ(|v0_i| + Δv_i)) + c ⊙ tanh(t̃)

- b = (1.5, 0.8, 0.7) m  — shape half-extent bound (relaxed for digital validation)
- c = (0.3, 0.3, 0.0) m  — translation bound (no z-shift)
- sign locks each vertex to its original octant → prevents self-intersection
- σ  maps amplitude to (0, b)
- tanh maps translation to (−c, c)
"""

import numpy as np
import torch


def reparameterize(v0, delta_v, t_tilde, R, b, c):
    """
    Args:
        v0:      (V, 3) initial icosphere vertices (unit sphere, frozen)
        delta_v: (V, 3) optimisable shape perturbation
        t_tilde: (3,)   optimisable translation in unconstrained space
        R:       (3, 3) rotation matrix (vehicle yaw, frozen)
        b:       (3,)   shape bound
        c:       (3,)   translation bound

    Returns:
        verts_local: (V, 3) deformed vertices in vehicle-local frame (centred at rooftop)
    """
    # Shape: sign-locked, sigmoid-bounded
    verts = b * v0.sign() * torch.sigmoid(v0.abs() + delta_v)   # (V, 3)

    # Rotate into vehicle heading frame
    verts = (R @ verts.T).T                                       # (V, 3)

    # Translation
    translation = c * torch.tanh(t_tilde)                        # (3,)
    verts = verts + translation                                   # (V, 3)

    return verts


def place_on_rooftop(verts_local, rooftop_world):
    """
    Translate local-frame vertices to world (LiDAR) coordinates.

    Args:
        verts_local:   (V, 3) in vehicle-local centred frame
        rooftop_world: (3,)   rooftop centre in LiDAR frame

    Returns:
        verts_world: (V, 3)
    """
    if not isinstance(rooftop_world, torch.Tensor):
        rooftop_world = torch.tensor(np.asarray(rooftop_world), dtype=torch.float32,
                                     device=verts_local.device)
    return verts_local + rooftop_world.to(verts_local.device)


def place_at_ground(verts_local, injection_pos):
    """
    Translate local-frame vertices to a ground-level injection position.

    Unlike place_on_rooftop, this places the object at ground level (z=0)
    for the appearing attack.

    Args:
        verts_local:   (V, 3) in local centred frame
        injection_pos: (3,)   injection position in LiDAR frame [x, y, z≈0]

    Returns:
        verts_world: (V, 3)
    """
    if not isinstance(injection_pos, torch.Tensor):
        injection_pos = torch.tensor(np.asarray(injection_pos), dtype=torch.float32,
                                     device=verts_local.device)
    return verts_local + injection_pos.to(verts_local.device)
