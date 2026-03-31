"""
Export adversarial mesh / point-opt checkpoints to viewable formats.

Outputs:
  1. .obj file   — viewable in MeshLab, Blender, Windows 3D Viewer, etc.
  2. .ply file   — point cloud (point-opt only)
  3. .html file  — standalone three.js viewer, open in any browser

Usage:
    # Export white-box mesh
    python export_mesh.py --ckpt results/adv_mesh_whitebox_final.pth --out-dir results/mesh_export

    # Export point-opt adversarial points → mesh reconstruction
    python export_mesh.py --pointopt results/adv_points_blackbox_final.pth --out-dir results/mesh_export

    # Export both original icosphere and deformed mesh
    python export_mesh.py --ckpt results/adv_mesh_whitebox_final.pth --out-dir results/mesh_export --show-original
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from attack.mesh import create_icosphere
from attack.reparameterize import reparameterize


def load_mesh_from_ckpt(ckpt_path, device='cpu'):
    """Load and reconstruct mesh vertices from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    delta_v = ckpt['delta_v']
    t_tilde = ckpt['t_tilde']
    v0 = ckpt['v0']
    faces = ckpt['faces']

    # Reconstruct deformed mesh using reparameterization
    # Use identity rotation (no car-specific placement)
    R = torch.eye(3, dtype=torch.float32)
    # Load b/c from checkpoint if saved, otherwise use YangCCS21 defaults
    b = ckpt.get('b', torch.tensor([0.45, 0.45, 0.41], dtype=torch.float32))
    c = ckpt.get('c', torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32))
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)

    with torch.no_grad():
        verts_deformed = reparameterize(v0, delta_v, t_tilde, R, b, c)

    return {
        'v0': v0.numpy(),
        'verts': verts_deformed.numpy(),
        'faces': faces.numpy(),
        'delta_v': delta_v.numpy(),
        't_tilde': t_tilde.numpy(),
        'method': ckpt.get('method', 'whitebox'),
    }


def export_obj(verts, faces, path, name='adversarial_mesh'):
    """Export mesh as Wavefront .obj file."""
    with open(path, 'w') as f:
        f.write(f'# {name}\n')
        f.write(f'# Vertices: {len(verts)}, Faces: {len(faces)}\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    print(f'  OBJ saved: {path}')


def load_pointopt_ckpt(ckpt_path, device='cpu'):
    """Load adversarial points from point-opt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pts = ckpt['adv_points'].numpy()
    history = ckpt.get('history', {})
    return pts, history


def export_ply(points, path):
    """Export point cloud as PLY file."""
    n = len(points)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for p in points:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    print(f'  PLY saved: {path} ({n} points)')


def reconstruct_mesh_from_points(points, method='convex_hull'):
    """
    Reconstruct a watertight triangle mesh from point cloud.

    Methods:
      - convex_hull (default): scipy ConvexHull, always produces closed mesh
      - alpha: Open3D alpha shape, may have holes for sparse points
      - poisson: Open3D Screened Poisson, closed watertight mesh (best for printing)
      - ball_pivoting: Open3D ball pivoting, usually open for sparse data

    Returns (vertices, faces) as numpy arrays.
    """
    if method == 'convex_hull':
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        verts = points.copy()
        faces = hull.simplices.copy()
        print(f'  ConvexHull: {len(verts)} verts, {len(faces)} faces (watertight)')
        return verts, faces

    try:
        import open3d as o3d
    except ImportError:
        print('  Open3D not available, falling back to ConvexHull')
        return reconstruct_mesh_from_points(points, method='convex_hull')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    if method == 'ball_pivoting':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * f for f in [0.5, 1.0, 1.5, 2.0, 3.0]]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif method == 'poisson':
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        densities = np.asarray(densities)
        thresh = np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(densities < thresh)
    elif method == 'alpha':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=avg_dist * 3.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    is_watertight = mesh.is_watertight()
    status = 'watertight' if is_watertight else 'open'
    print(f'  Open3D {method}: {len(verts)} verts, {len(faces)} faces ({status})')

    if not is_watertight and len(faces) > 0:
        print(f'  WARNING: mesh is not watertight, consider using --mesh-method convex_hull')

    return verts, faces


def poisson_reconstruct_and_clean(points, depth=8, min_component_faces=1024):
    """
    Screened Poisson reconstruction with small component removal.

    Follows AAAI 2020 pipeline:
      1. Estimate normals
      2. Screened Poisson surface reconstruction
      3. Remove connected components with < min_component_faces faces

    Args:
        points:              (N, 3) numpy array
        depth:               Poisson octree depth (higher = finer detail)
        min_component_faces: remove components smaller than this

    Returns:
        mesh:  Open3D TriangleMesh (watertight, cleaned)
        verts: (V, 3) numpy array
        faces: (F, 3) numpy array
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    thresh = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < thresh)

    n_faces_before = len(mesh.triangles)

    triangle_clusters, cluster_n_tri, cluster_area = (
        mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_tri = np.asarray(cluster_n_tri)

    small_cluster_mask = cluster_n_tri[triangle_clusters] < min_component_faces
    mesh.remove_triangles_by_mask(small_cluster_mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    n_removed = n_faces_before - len(mesh.triangles)
    n_clusters_removed = (cluster_n_tri < min_component_faces).sum()
    print(f'  Poisson reconstruction: depth={depth}')
    print(f'  Before cleanup: {n_faces_before} faces')
    print(f'  Removed {n_clusters_removed} small components ({n_removed} faces)')
    print(f'  After cleanup: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces')
    print(f'  Watertight: {mesh.is_watertight()}')

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return mesh, verts, faces


def sample_points_from_mesh(verts, faces, n_points, method='uniform'):
    """
    Sample points from a triangle mesh surface.

    Args:
        verts:    (V, 3) numpy array
        faces:    (F, 3) numpy array (int indices)
        n_points: number of points to sample
        method:   'uniform' or 'poisson_disk'

    Returns:
        sampled_points: (n_points, 3) numpy array
    """
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        if method == 'poisson_disk':
            pcd = mesh.sample_points_poisson_disk(n_points)
        else:
            pcd = mesh.sample_points_uniformly(n_points)
        return np.asarray(pcd.points)
    except ImportError:
        pass

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()

    chosen = np.random.choice(len(faces), size=n_points, p=probs)
    r1 = np.sqrt(np.random.rand(n_points, 1))
    r2 = np.random.rand(n_points, 1)
    pts = (1 - r1) * v0[chosen] + r1 * (1 - r2) * v1[chosen] + r1 * r2 * v2[chosen]
    return pts


def closest_sample_points_from_mesh(adv_points, verts, faces):
    """
    AAAI 2020 "Closest" sampling: for each adversarial point, find the
    closest point on the reconstructed mesh surface.

    This preserves adversarial positioning far better than uniform sampling.

    Args:
        adv_points: (N, 3) numpy array — original adversarial points
        verts:      (V, 3) numpy array — mesh vertices
        faces:      (F, 3) numpy array — mesh face indices

    Returns:
        closest_pts: (N, 3) numpy array — points on mesh surface
        distances:   (N,)   numpy array — distance from each adv point to surface
    """
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

        query = o3d.core.Tensor(adv_points.astype(np.float32),
                                dtype=o3d.core.Dtype.Float32)
        result = scene.compute_closest_points(query)
        closest_pts = result['points'].numpy()
        distances = np.linalg.norm(closest_pts - adv_points, axis=1)

        print(f'  Closest sampling: {len(adv_points)} points')
        print(f'  Mean dist to surface: {distances.mean():.4f} m')
        print(f'  Max  dist to surface: {distances.max():.4f} m')
        return closest_pts, distances

    except (ImportError, AttributeError):
        print('  Open3D RaycastingScene unavailable, using triangle projection fallback')
        return _closest_sample_fallback(adv_points, verts, faces)


def _closest_sample_fallback(adv_points, verts, faces):
    """Brute-force closest-point-on-triangle for each adversarial point."""
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    closest_pts = np.zeros_like(adv_points)
    distances = np.full(len(adv_points), np.inf)

    for i, p in enumerate(adv_points):
        edge0 = v1 - v0
        edge1 = v2 - v0
        v0p = p[None, :] - v0

        d00 = np.sum(edge0 * edge0, axis=1)
        d01 = np.sum(edge0 * edge1, axis=1)
        d11 = np.sum(edge1 * edge1, axis=1)
        d20 = np.sum(v0p * edge0, axis=1)
        d21 = np.sum(v0p * edge1, axis=1)
        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom

        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        uv_sum = u + v
        mask = uv_sum > 1
        u[mask] = u[mask] / uv_sum[mask]
        v[mask] = v[mask] / uv_sum[mask]

        proj = v0 + u[:, None] * edge0 + v[:, None] * edge1
        dists = np.linalg.norm(proj - p[None, :], axis=1)
        best = np.argmin(dists)
        closest_pts[i] = proj[best]
        distances[i] = dists[best]

    print(f'  Closest sampling (fallback): {len(adv_points)} points')
    print(f'  Mean dist to surface: {distances.mean():.4f} m')
    print(f'  Max  dist to surface: {distances.max():.4f} m')
    return closest_pts, distances


def export_pointopt_html(points, verts, faces, path,
                         title='Point-Opt Adversarial Object'):
    """Export HTML viewer for point-opt: shows both point cloud and reconstructed mesh."""
    pts_json = json.dumps(points.tolist())
    verts_json = json.dumps(verts.tolist())
    faces_json = json.dumps(faces.tolist())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
  #info {{
    position: absolute; top: 10px; left: 10px; z-index: 10;
    color: #e0e0e0; font-size: 13px; background: rgba(0,0,0,0.6);
    padding: 12px 16px; border-radius: 8px; max-width: 350px;
  }}
  #info h3 {{ margin-bottom: 6px; color: #ff7043; }}
  #info p {{ margin: 2px 0; line-height: 1.4; }}
  .btn {{
    display: inline-block; margin: 4px 2px; padding: 6px 14px;
    background: #2d2d44; color: #ff7043; border: 1px solid #ff7043;
    border-radius: 4px; cursor: pointer; font-size: 12px;
  }}
  .btn:hover {{ background: #ff7043; color: #1a1a2e; }}
  .btn.active {{ background: #ff7043; color: #1a1a2e; }}
</style>
</head>
<body>
<div id="info">
  <h3>Point-Opt Adversarial Object</h3>
  <p>Points: {len(points)} | Mesh: {len(verts)} verts, {len(faces)} faces</p>
  <p>Bbox: [{points.min(0)[0]:.2f}, {points.min(0)[1]:.2f}, {points.min(0)[2]:.2f}]
     → [{points.max(0)[0]:.2f}, {points.max(0)[1]:.2f}, {points.max(0)[2]:.2f}]</p>
  <div style="margin-top:8px">
    <span class="btn active" onclick="toggle('points')">Points</span>
    <span class="btn" onclick="toggle('mesh')">Mesh</span>
    <span class="btn" onclick="toggle('both')">Both</span>
    <span class="btn" onclick="toggle('wireframe')">Wireframe</span>
  </div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const ptsData = {pts_json};
const vertsData = {verts_json};
const facesData = {faces_json};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
camera.position.set(2, 1.5, 2);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0x404060, 1.5));
const dl = new THREE.DirectionalLight(0xffffff, 1.2);
dl.position.set(3, 5, 4); scene.add(dl);

const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
grid.rotation.x = Math.PI / 2; scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

// Point cloud
const ptGeom = new THREE.BufferGeometry();
const ptPos = new Float32Array(ptsData.length * 3);
for (let i = 0; i < ptsData.length; i++) {{
  ptPos[i*3] = ptsData[i][0]; ptPos[i*3+1] = ptsData[i][1]; ptPos[i*3+2] = ptsData[i][2];
}}
ptGeom.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));
const ptMat = new THREE.PointsMaterial({{ color: 0xff7043, size: 0.03, sizeAttenuation: true }});
const ptCloud = new THREE.Points(ptGeom, ptMat);
scene.add(ptCloud);

// Reconstructed mesh
const meshGeom = new THREE.BufferGeometry();
const meshPos = new Float32Array(vertsData.length * 3);
for (let i = 0; i < vertsData.length; i++) {{
  meshPos[i*3] = vertsData[i][0]; meshPos[i*3+1] = vertsData[i][1]; meshPos[i*3+2] = vertsData[i][2];
}}
meshGeom.setAttribute('position', new THREE.BufferAttribute(meshPos, 3));
const idx = [];
for (const f of facesData) idx.push(f[0], f[1], f[2]);
meshGeom.setIndex(idx);
meshGeom.computeVertexNormals();

const meshMat = new THREE.MeshPhongMaterial({{
  color: 0x4fc3f7, specular: 0x222222, shininess: 30,
  transparent: true, opacity: 0.6, side: THREE.DoubleSide,
}});
const meshObj = new THREE.Mesh(meshGeom, meshMat);
meshObj.visible = false;
scene.add(meshObj);

const wireObj = new THREE.LineSegments(
  new THREE.WireframeGeometry(meshGeom),
  new THREE.LineBasicMaterial({{ color: 0x4fc3f7, linewidth: 1 }})
);
wireObj.visible = false;
scene.add(wireObj);

window.toggle = function(mode) {{
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  ptCloud.visible = (mode === 'points' || mode === 'both');
  meshObj.visible = (mode === 'mesh' || mode === 'both');
  wireObj.visible = (mode === 'wireframe');
}};

(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();

addEventListener('resize', () => {{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
</script>
</body>
</html>"""

    with open(path, 'w') as f:
        f.write(html)
    print(f'  HTML viewer saved: {path}')


def export_pointopt(ckpt_path, out_dir, mesh_method='ball_pivoting'):
    """Full export pipeline for point-opt checkpoint."""
    print(f'Loading point-opt checkpoint: {ckpt_path}')
    points, history = load_pointopt_ckpt(ckpt_path)
    name = os.path.splitext(os.path.basename(ckpt_path))[0]

    print(f'  Points: {len(points)}')
    print(f'  Bbox: [{points.min(0)}] → [{points.max(0)}]')
    extent = points.max(0) - points.min(0)
    print(f'  Size: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m')

    # 1. Export PLY point cloud
    ply_path = os.path.join(out_dir, f'{name}.ply')
    export_ply(points, ply_path)

    # 2. Reconstruct mesh
    print(f'  Reconstructing mesh ({mesh_method})...')
    verts, faces = reconstruct_mesh_from_points(points, method=mesh_method)

    # 3. Export OBJ
    if len(faces) > 0:
        obj_path = os.path.join(out_dir, f'{name}.obj')
        export_obj(verts, faces, obj_path, f'{name} (reconstructed)')
    else:
        print('  WARNING: mesh reconstruction produced 0 faces, skipping OBJ')

    # 4. Export HTML viewer
    html_path = os.path.join(out_dir, f'{name}.html')
    export_pointopt_html(points, verts, faces, html_path)

    return points, verts, faces


def export_html_viewer(mesh_data, path, title='Adversarial Mesh Viewer'):
    """Export standalone HTML file with three.js interactive 3D viewer."""
    verts = mesh_data['verts']
    faces = mesh_data['faces']
    v0 = mesh_data['v0']
    method = mesh_data['method']

    # Convert to JSON-serializable lists
    verts_json = json.dumps(verts.tolist())
    faces_json = json.dumps(faces.tolist())
    v0_json = json.dumps(v0.tolist())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
  #info {{
    position: absolute; top: 10px; left: 10px; z-index: 10;
    color: #e0e0e0; font-size: 13px; background: rgba(0,0,0,0.6);
    padding: 12px 16px; border-radius: 8px; max-width: 320px;
  }}
  #info h3 {{ margin-bottom: 6px; color: #4fc3f7; }}
  #info p {{ margin: 2px 0; line-height: 1.4; }}
  #controls {{
    position: absolute; bottom: 10px; left: 10px; z-index: 10;
    color: #aaa; font-size: 12px; background: rgba(0,0,0,0.5);
    padding: 8px 12px; border-radius: 6px;
  }}
  .btn {{
    display: inline-block; margin: 4px 2px; padding: 6px 14px;
    background: #2d2d44; color: #4fc3f7; border: 1px solid #4fc3f7;
    border-radius: 4px; cursor: pointer; font-size: 12px;
  }}
  .btn:hover {{ background: #4fc3f7; color: #1a1a2e; }}
  .btn.active {{ background: #4fc3f7; color: #1a1a2e; }}
</style>
</head>
<body>
<div id="info">
  <h3>Adversarial Mesh ({method})</h3>
  <p>Vertices: {len(verts)} | Faces: {len(faces)}</p>
  <p>delta_v norm: {np.linalg.norm(mesh_data['delta_v']):.3f}</p>
  <p>t_tilde: [{', '.join(f'{x:.3f}' for x in mesh_data['t_tilde'])}]</p>
  <div style="margin-top:8px">
    <span class="btn active" onclick="toggleMesh('deformed')">Deformed</span>
    <span class="btn" onclick="toggleMesh('original')">Original</span>
    <span class="btn" onclick="toggleMesh('both')">Both</span>
    <span class="btn" onclick="toggleWireframe()">Wireframe</span>
  </div>
</div>
<div id="controls">
  Drag to rotate | Scroll to zoom | Right-drag to pan
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const vertsData = {verts_json};
const facesData = {faces_json};
const v0Data = {v0_json};

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(1.5, 1.0, 1.5);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Lights
scene.add(new THREE.AmbientLight(0x404060, 1.5));
const dl1 = new THREE.DirectionalLight(0xffffff, 1.2);
dl1.position.set(3, 5, 4);
scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0x4fc3f7, 0.5);
dl2.position.set(-2, -3, 1);
scene.add(dl2);

// Grid + axes
const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

// Build geometry helper
function buildGeom(verts) {{
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {{
    positions[i*3]   = verts[i][0];
    positions[i*3+1] = verts[i][1];
    positions[i*3+2] = verts[i][2];
  }}
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const indices = [];
  for (const f of facesData) {{
    indices.push(f[0], f[1], f[2]);
  }}
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}}

// Deformed mesh (red-orange)
const geomDef = buildGeom(vertsData);
const matDef = new THREE.MeshPhongMaterial({{
  color: 0xff4444, specular: 0x444444, shininess: 30,
  transparent: true, opacity: 0.85, side: THREE.DoubleSide,
}});
const meshDef = new THREE.Mesh(geomDef, matDef);
scene.add(meshDef);

const wireDef = new THREE.LineSegments(
  new THREE.WireframeGeometry(geomDef),
  new THREE.LineBasicMaterial({{ color: 0xff8888, linewidth: 1 }})
);
wireDef.visible = false;
scene.add(wireDef);

// Original mesh (cyan, semi-transparent)
const geomOrig = buildGeom(v0Data);
const matOrig = new THREE.MeshPhongMaterial({{
  color: 0x00bcd4, specular: 0x222222, shininess: 20,
  transparent: true, opacity: 0.3, side: THREE.DoubleSide,
}});
const meshOrig = new THREE.Mesh(geomOrig, matOrig);
meshOrig.visible = false;
scene.add(meshOrig);

const wireOrig = new THREE.LineSegments(
  new THREE.WireframeGeometry(geomOrig),
  new THREE.LineBasicMaterial({{ color: 0x00bcd4, linewidth: 1, transparent: true, opacity: 0.4 }})
);
wireOrig.visible = false;
scene.add(wireOrig);

// State
let showWireframe = false;
let currentMode = 'deformed';

window.toggleMesh = function(mode) {{
  currentMode = mode;
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');

  meshDef.visible = (mode === 'deformed' || mode === 'both');
  meshOrig.visible = (mode === 'original' || mode === 'both');
  wireDef.visible = showWireframe && meshDef.visible;
  wireOrig.visible = showWireframe && meshOrig.visible;
  if (mode === 'both') {{
    matDef.opacity = 0.7;
    matOrig.opacity = 0.3;
  }} else {{
    matDef.opacity = 0.85;
    matOrig.opacity = 0.6;
  }}
}};

window.toggleWireframe = function() {{
  showWireframe = !showWireframe;
  wireDef.visible = showWireframe && meshDef.visible;
  wireOrig.visible = showWireframe && meshOrig.visible;
  const btn = document.querySelectorAll('.btn')[3];
  btn.classList.toggle('active', showWireframe);
}};

// Animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    with open(path, 'w') as f:
        f.write(html)
    print(f'  HTML viewer saved: {path}')


def main():
    p = argparse.ArgumentParser(description='Export adversarial mesh / point-opt to OBJ + HTML')
    p.add_argument('--ckpt', default=None,
                   help='Path to adversarial MESH checkpoint')
    p.add_argument('--pointopt', default=None,
                   help='Path to point-opt checkpoint (adv_points_pointopt_*.pth)')
    p.add_argument('--out-dir', default='results/mesh_export')
    p.add_argument('--show-original', action='store_true',
                   help='Also export original icosphere as separate OBJ')
    p.add_argument('--mesh-method', default='convex_hull',
                   choices=['convex_hull', 'ball_pivoting', 'poisson', 'alpha'],
                   help='Mesh reconstruction method (convex_hull guarantees watertight)')
    args = p.parse_args()

    if args.pointopt is None and args.ckpt is None:
        args.ckpt = 'results/adv_mesh_whitebox_final.pth'

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pointopt:
        export_pointopt(args.pointopt, args.out_dir,
                        mesh_method=args.mesh_method)
    else:
        print(f'Loading {args.ckpt}...')
        mesh = load_mesh_from_ckpt(args.ckpt)

        name = os.path.splitext(os.path.basename(args.ckpt))[0]
        print(f'  Method: {mesh["method"]}')
        print(f'  Vertices: {len(mesh["verts"])}, Faces: {len(mesh["faces"])}')
        print(f'  delta_v norm: {np.linalg.norm(mesh["delta_v"]):.3f}')

        obj_path = os.path.join(args.out_dir, f'{name}_deformed.obj')
        export_obj(mesh['verts'], mesh['faces'], obj_path, f'{name} (deformed)')

        if args.show_original:
            obj_orig = os.path.join(args.out_dir, f'{name}_original.obj')
            export_obj(mesh['v0'], mesh['faces'], obj_orig,
                       f'{name} (original icosphere)')

        html_path = os.path.join(args.out_dir, f'{name}_viewer.html')
        export_html_viewer(mesh, html_path, f'Adversarial Mesh — {name}')

    print(f'\nDone! Files in {args.out_dir}/')
    print(f'  - Open the .html file in a browser for interactive 3D viewing')
    print(f'  - Open the .obj / .ply files in MeshLab/Blender for editing')


if __name__ == '__main__':
    main()
