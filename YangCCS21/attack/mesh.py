"""
Icosphere mesh creation and adjacency utilities.

trimesh.creation.icosphere(subdivisions=2) → 162 vertices, 320 faces
"""

import numpy as np
import torch
import trimesh


def create_icosphere(subdivisions=2, radius=1.0):
    """
    Create an icosphere and return tensors ready for optimisation.

    Returns:
        v0:   (V, 3) float32 tensor – initial normalised vertices (requires_grad=False)
        faces:(F, 3) int64  tensor – face index triplets
        adj:  dict mapping vertex index → list of neighbour indices (for Laplacian)
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    v0 = torch.tensor(mesh.vertices, dtype=torch.float32)  # (162, 3)
    faces = torch.tensor(mesh.faces, dtype=torch.long)      # (320, 3)
    adj = build_adjacency(faces, len(v0))
    return v0, faces, adj


def build_adjacency(faces, n_verts):
    """
    Build adjacency list from face index tensor.

    Args:
        faces: (F, 3) LongTensor
        n_verts: int
    Returns:
        adj: list of length n_verts, each element is a list of neighbour indices
    """
    adj = [[] for _ in range(n_verts)]
    faces_np = faces.cpu().numpy()
    for f in faces_np:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        if j not in adj[i]: adj[i].append(j)
        if k not in adj[i]: adj[i].append(k)
        if i not in adj[j]: adj[j].append(i)
        if k not in adj[j]: adj[j].append(k)
        if i not in adj[k]: adj[k].append(i)
        if j not in adj[k]: adj[k].append(j)
    return adj


def get_aabb(vertices):
    """
    Compute axis-aligned bounding box of a vertex tensor.

    Args:
        vertices: (V, 3)
    Returns:
        aabb_min: (3,)
        aabb_max: (3,)
    """
    return vertices.min(dim=0).values, vertices.max(dim=0).values


def rotation_matrix_z(yaw):
    """
    3×3 rotation matrix around z-axis for given yaw (radians).

    Args:
        yaw: float or scalar tensor
    Returns:
        R: (3, 3) float32 tensor
    """
    if isinstance(yaw, torch.Tensor):
        c, s = torch.cos(yaw), torch.sin(yaw)
        R = torch.stack([
            torch.stack([ c, -s, c.new_zeros(())], dim=0),
            torch.stack([ s,  c, c.new_zeros(())], dim=0),
            torch.stack([c.new_zeros(()), c.new_zeros(()), c.new_ones(())], dim=0),
        ], dim=0)
    else:
        c, s = np.cos(yaw), np.sin(yaw)
        R = torch.tensor([
            [ c, -s, 0.],
            [ s,  c, 0.],
            [0., 0., 1.],
        ], dtype=torch.float32)
    return R
