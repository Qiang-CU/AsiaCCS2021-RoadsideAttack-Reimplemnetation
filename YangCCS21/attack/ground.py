"""
Deterministic ground point generation via LiDAR ray-plane intersection.

For a flat ground at z=ground_z (in LiDAR frame, LiDAR at origin),
each downward-pointing LiDAR ray hits the plane at a deterministic point.
This eliminates random noise in ground points that was causing
detection instability across evaluations.
"""

import numpy as np
import functools

_CACHE = {}


def generate_ground_raytrace(ground_z=-0.75,
                              n_elevation=16,
                              elev_min_deg=-15.0,
                              elev_max_deg=15.0,
                              h_step_deg=0.5,
                              x_range=(0.5, 40.0),
                              y_range=(-20.0, 20.0)):
    """
    Generate ground points by intersecting VLP-16 rays with a flat plane.

    Args:
        ground_z: ground plane z-coordinate in LiDAR frame (negative = below)
        n_elevation: number of LiDAR channels
        h_step_deg: azimuth step in degrees (coarser than 0.2 for fewer points)
        x_range: valid x range for ground points (forward direction)
        y_range: valid y range for ground points

    Returns:
        ground_pts: (N, 4) float32 array [x, y, z, intensity]
    """
    cache_key = (ground_z, n_elevation, elev_min_deg, elev_max_deg,
                 h_step_deg, x_range, y_range)
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    elevations = np.linspace(np.radians(elev_min_deg),
                              np.radians(elev_max_deg), n_elevation)

    # Only downward-pointing channels hit the ground
    down_mask = elevations < -np.radians(0.3)
    down_elevs = elevations[down_mask]

    if len(down_elevs) == 0 or ground_z >= 0:
        result = np.zeros((0, 4), dtype=np.float32)
        _CACHE[cache_key] = result
        return result.copy()

    # Forward-facing azimuths: -90° to +90° (x > 0 quadrant)
    n_az = int(180.0 / h_step_deg) + 1
    azimuths = np.linspace(-np.pi / 2, np.pi / 2, n_az)

    pts = []
    for el in down_elevs:
        cos_el = np.cos(el)
        sin_el = np.sin(el)  # negative
        t = ground_z / sin_el  # positive (ground_z < 0, sin_el < 0)

        for az in azimuths:
            x = t * cos_el * np.cos(az)
            y = t * cos_el * np.sin(az)

            if x_range[0] < x < x_range[1] and y_range[0] < y < y_range[1]:
                pts.append([x, y, ground_z, 0.2])

    if not pts:
        result = np.zeros((0, 4), dtype=np.float32)
    else:
        result = np.array(pts, dtype=np.float32)

    _CACHE[cache_key] = result
    return result.copy()
