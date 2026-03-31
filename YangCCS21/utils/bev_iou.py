"""BEV rotated box IoU using Shapely polygons."""

import numpy as np
from shapely.geometry import Polygon


def bbox_to_bev_polygon(bbox):
    """
    Convert a 7-DoF bbox to a Shapely Polygon in BEV.

    Args:
        bbox: array-like [cx, cy, cz, l, w, h, yaw]
    Returns:
        shapely.geometry.Polygon
    """
    cx, cy, _, l, w, _, yaw = bbox
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # Four corners in local frame (l along x, w along y)
    corners_local = np.array([
        [ l / 2,  w / 2],
        [-l / 2,  w / 2],
        [-l / 2, -w / 2],
        [ l / 2, -w / 2],
    ])

    R = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]])
    corners = (R @ corners_local.T).T + np.array([cx, cy])
    return Polygon(corners)


def bev_iou(bbox1, bbox2):
    """
    Compute BEV IoU between two 7-DoF bboxes.

    Args:
        bbox1, bbox2: array-like [cx, cy, cz, l, w, h, yaw]
    Returns:
        float IoU in [0, 1]
    """
    p1 = bbox_to_bev_polygon(bbox1)
    p2 = bbox_to_bev_polygon(bbox2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / (union + 1e-10)


def bev_iou_matrix(bboxes1, bboxes2):
    """
    Compute pairwise BEV IoU matrix.

    Args:
        bboxes1: (M, 7)
        bboxes2: (N, 7)
    Returns:
        iou_matrix: (M, N) numpy float32
    """
    M, N = len(bboxes1), len(bboxes2)
    iou = np.zeros((M, N), dtype=np.float32)
    polys2 = [bbox_to_bev_polygon(bboxes2[j]) for j in range(N)]
    for i in range(M):
        p1 = bbox_to_bev_polygon(bboxes1[i])
        if not p1.is_valid:
            continue
        for j, p2 in enumerate(polys2):
            if not p2.is_valid:
                continue
            inter = p1.intersection(p2).area
            union = p1.union(p2).area
            iou[i, j] = inter / (union + 1e-10)
    return iou


def nms_bev(bboxes, scores, iou_thresh=0.1):
    """
    Fast NMS for BEV boxes using axis-aligned bounding-box approximation.

    Converts rotated boxes to their axis-aligned envelopes and runs
    vectorised greedy NMS.  ~100× faster than Shapely-based NMS.

    Args:
        bboxes: (N, 7)  [cx, cy, cz, l, w, h, yaw]
        scores: (N,)
        iou_thresh: float
    Returns:
        keep: list of indices to keep
    """
    if len(bboxes) == 0:
        return []

    cx, cy = bboxes[:, 0], bboxes[:, 1]
    l, w, yaw = bboxes[:, 3], bboxes[:, 4], bboxes[:, 6]

    # Axis-aligned half-extents of the rotated rectangle
    cos_y = np.abs(np.cos(yaw))
    sin_y = np.abs(np.sin(yaw))
    half_x = (l * cos_y + w * sin_y) / 2
    half_y = (l * sin_y + w * cos_y) / 2

    x1 = cx - half_x
    y1 = cy - half_y
    x2 = cx + half_x
    y2 = cy + half_y
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1].copy()
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def nms_bev_shapely(bboxes, scores, iou_thresh=0.1):
    """
    Non-maximum suppression for BEV rotated boxes (Shapely, slow).

    Kept for exact rotated-box NMS when precision matters more than speed.

    Args:
        bboxes: (N, 7)
        scores: (N,)
        iou_thresh: float
    Returns:
        keep: list of indices to keep
    """
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(bboxes), dtype=bool)

    polys = [bbox_to_bev_polygon(bboxes[i]) for i in range(len(bboxes))]

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        p_cur = polys[idx]
        if not p_cur.is_valid:
            continue
        for other in order:
            if suppressed[other] or other == idx:
                continue
            p_other = polys[other]
            if not p_other.is_valid:
                continue
            inter = p_cur.intersection(p_other).area
            union = p_cur.union(p_other).area
            if inter / (union + 1e-10) > iou_thresh:
                suppressed[other] = True

    return keep
