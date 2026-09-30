"""Rigid transforms, box corners, projection, BEV IoU, and points-in-box tests.

Box convention everywhere in this project: (x, y, z, l, w, h, yaw) in the Boreas LiDAR frame
(x forward, y left, z up), z at the box *centre*, l along the heading, yaw about +z.
"""
import numpy as np
from shapely.geometry import Polygon


def inv_tf(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_tf(T, pts):
    """pts (N,3+) -> transformed xyz (N,3)."""
    return pts[:, :3] @ T[:3, :3].T + T[:3, 3]


def relative_tf(T_enu_a, T_enu_b):
    """T_a_b: maps points in frame b (at its timestamp) into frame a (at its timestamp)."""
    return inv_tf(T_enu_a) @ T_enu_b


def bev_corners(boxes):
    """boxes (N,7) -> (N,4,2) BEV corners, counter-clockwise."""
    boxes = np.atleast_2d(boxes)
    x, y, l, w, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    dx = np.array([0.5, -0.5, -0.5, 0.5])
    dy = np.array([0.5, 0.5, -0.5, -0.5])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    px = x[:, None] + c * (dx * l[:, None]) - s * (dy * w[:, None])
    py = y[:, None] + s * (dx * l[:, None]) + c * (dy * w[:, None])
    return np.stack([px, py], axis=-1)


def box_corners_3d(boxes):
    """boxes (N,7) -> (N,8,3)."""
    boxes = np.atleast_2d(boxes)
    bev = bev_corners(boxes)
    zb = (boxes[:, 2] - boxes[:, 5] / 2)[:, None]
    zt = (boxes[:, 2] + boxes[:, 5] / 2)[:, None]
    bottom = np.concatenate([bev, np.repeat(zb, 4, 1)[..., None]], -1)
    top = np.concatenate([bev, np.repeat(zt, 4, 1)[..., None]], -1)
    return np.concatenate([bottom, top], axis=1)


def bev_iou_matrix(a, b):
    """Rotated BEV IoU between (N,7) and (M,7)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    out = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return out
    pa = [Polygon(c) for c in bev_corners(a)]
    pb = [Polygon(c) for c in bev_corners(b)]
    ca, cb = a[:, :2], b[:, :2]
    ra = 0.5 * np.hypot(a[:, 3], a[:, 4])
    rb = 0.5 * np.hypot(b[:, 3], b[:, 4])
    d = np.linalg.norm(ca[:, None] - cb[None], axis=-1)
    for i, j in zip(*np.nonzero(d < ra[:, None] + rb[None])):
        inter = pa[i].intersection(pb[j]).area
        if inter > 0:
            out[i, j] = inter / (pa[i].area + pb[j].area - inter)
    return out


def rotated_nms_bev(boxes, scores, iou_thr, max_det=None):
    order = np.argsort(-scores)
    keep = []
    polys = [Polygon(c) for c in bev_corners(boxes)] if len(boxes) else []
    done = np.zeros(len(boxes), bool)
    for idx in order:
        if done[idx]:
            continue
        keep.append(idx)
        done[idx] = True
        if max_det and len(keep) >= max_det:
            break
        pi = polys[idx]
        for j in order:
            if done[j]:
                continue
            if np.hypot(*(boxes[idx, :2] - boxes[j, :2])) > 0.5 * (np.hypot(*boxes[idx, 3:5]) + np.hypot(*boxes[j, 3:5])):
                continue
            inter = pi.intersection(polys[j]).area
            if inter > 0 and inter / (pi.area + polys[j].area - inter) > iou_thr:
                done[j] = True
    return np.array(keep, dtype=np.int64)


def points_in_boxes(points, boxes, enlarge=1.0):
    """Returns (M, N) bool mask: point j inside box i (boxes scaled by `enlarge`)."""
    boxes = np.atleast_2d(boxes)
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros((len(boxes), len(points)), bool)
    rel = points[None, :, :3] - boxes[:, None, :3]
    c, s = np.cos(-boxes[:, 6])[:, None], np.sin(-boxes[:, 6])[:, None]
    lx = c * rel[..., 0] - s * rel[..., 1]
    ly = s * rel[..., 0] + c * rel[..., 1]
    return (np.abs(lx) <= boxes[:, 3:4] * enlarge / 2) & (np.abs(ly) <= boxes[:, 4:5] * enlarge / 2) & \
           (np.abs(rel[..., 2]) <= boxes[:, 5:6] * enlarge / 2)


def project_to_image(pts_cam, P, width, height):
    """pts_cam (N,3) in camera frame -> (N,2) pixels and a validity mask (in front and inside)."""
    z = pts_cam[:, 2]
    uvw = pts_cam @ P[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    valid = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid


def box_image_rect(boxes, T_cam_lidar, P, width, height):
    """Project 3D boxes to clipped 2D rectangles. Returns (N,4) [u0,v0,u1,v1], visible-fraction (N,)."""
    boxes = np.atleast_2d(boxes)
    rects = np.zeros((len(boxes), 4))
    frac = np.zeros(len(boxes))
    if len(boxes) == 0:
        return rects, frac
    corners = box_corners_3d(boxes).reshape(-1, 3)
    cam = apply_tf(T_cam_lidar, corners).reshape(len(boxes), 8, 3)
    for i in range(len(boxes)):
        if np.all(cam[i, :, 2] < 0.5):
            continue
        c = cam[i][cam[i, :, 2] >= 0.5]
        uv = (c @ P[:3, :3].T)
        uv = uv[:, :2] / uv[:, 2:3]
        u0, v0 = uv.min(0)
        u1, v1 = uv.max(0)
        full = max((u1 - u0) * (v1 - v0), 1e-6)
        cu0, cv0 = np.clip([u0, v0], 0, [width - 1, height - 1])
        cu1, cv1 = np.clip([u1, v1], 0, [width - 1, height - 1])
        clipped = max(cu1 - cu0, 0) * max(cv1 - cv0, 0)
        rects[i] = [cu0, cv0, cu1, cv1]
        frac[i] = clipped / full
    return rects, frac


def in_roi(xy, roi):
    x, y = xy[:, 0], xy[:, 1]
    az = np.degrees(np.arctan2(y, np.maximum(x, 1e-6)))
    return (x >= roi["x_min"]) & (x <= roi["x_max"]) & (np.abs(y) <= roi["y_abs_max"]) & \
           (np.abs(az) <= roi["half_fov_deg"])


def polar(xy):
    return np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
