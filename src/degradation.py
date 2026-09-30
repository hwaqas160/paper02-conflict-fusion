"""Per-object and per-frame degradation features ("probes").

Probes are evaluated on the data the frozen detectors actually saw (i.e. after weather
degradation), at *query boxes*. At inference the query boxes are the union of every modality's
detections, so a camera probe can be read at a location only LiDAR or radar reported — that is
what lets the arbitration head judge "should the camera have seen this?". Probes at GT boxes are
cached too, but are used only for analysis (harm characterisation), never as model inputs.
"""
import cv2
import numpy as np

from .geometry import apply_tf, box_image_rect, points_in_boxes

LIDAR_PROBES = ["l_log_pts", "l_density", "l_mean_int", "l_frac_low_int", "l_z_std", "l_ring_ratio"]
CAMERA_PROBES = ["c_in_image", "c_vis_frac", "c_log_area", "c_lum", "c_contrast", "c_log_lap",
                 "c_dark", "c_edge", "c_sat"]
RADAR_PROBES = ["r_max_pow", "r_mean_pow", "r_bg_pow", "r_snr", "r_log_hits"]
GEOM_PROBES = ["g_range", "g_azimuth_abs"]
PROBE_NAMES = GEOM_PROBES + LIDAR_PROBES + CAMERA_PROBES + RADAR_PROBES

FRAME_FEATURES = ["f_l_log_pts", "f_l_near_clutter", "f_l_mean_int", "f_l_log_far_pts",
                  "f_c_lum", "f_c_contrast", "f_c_log_lap", "f_c_dark", "f_c_sat",
                  "f_r_noise", "f_r_log_hits", "f_r_hit_pow"]


class ImageStats:
    """Integral images for O(1) crop statistics on the degraded image."""

    def __init__(self, img_rgb):
        img = img_rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        dark = img.min(axis=2)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.h, self.w = gray.shape
        self.I = {k: cv2.integral(v.astype(np.float64)) for k, v in {
            "g": gray, "g2": gray * gray, "lap": lap, "lap2": lap * lap, "edge": np.hypot(gx, gy),
            "dark": dark, "sat": hsv[..., 1]}.items()}
        self.global_feats = [float(gray.mean()), float(gray.std() / max(gray.mean(), 1e-3)),
                             float(np.log(lap.var() + 1e-8)), float(np.percentile(dark, 5)),
                             float(hsv[..., 1].mean())]

    def _mean(self, key, u0, v0, u1, v1):
        I = self.I[key]
        s = I[v1, u1] - I[v0, u1] - I[v1, u0] + I[v0, u0]
        return s / max((u1 - u0) * (v1 - v0), 1)

    def crop(self, rect):
        u0, v0, u1, v1 = [int(round(x)) for x in rect]
        u0, u1 = np.clip([u0, u1], 0, self.w)
        v0, v1 = np.clip([v0, v1], 0, self.h)
        if u1 - u0 < 2 or v1 - v0 < 2:
            return None
        m = self._mean("g", u0, v0, u1, v1)
        var = max(self._mean("g2", u0, v0, u1, v1) - m * m, 0)
        lap_var = max(self._mean("lap2", u0, v0, u1, v1) - self._mean("lap", u0, v0, u1, v1) ** 2, 0)
        return [m, np.sqrt(var) / max(m, 1e-3), np.log(lap_var + 1e-8), self._mean("dark", u0, v0, u1, v1),
                self._mean("edge", u0, v0, u1, v1), self._mean("sat", u0, v0, u1, v1)]


def lidar_frame_features(points, z_ground):
    r = np.linalg.norm(points[:, :2], axis=1)
    above = points[:, 2] > z_ground + 0.3
    near = r < 5.0
    return [float(np.log1p(len(points))), float(np.mean(near & above)) if len(points) else 0.0,
            float(points[:, 3].mean()) / 255.0 if len(points) else 0.0, float(np.log1p(np.sum(r > 40)))]


def compute_probes(query_boxes, points, img_stats, T_cam_lidar, P, radar_map, radar_grid, radar_hits_occ,
                   n_rings=128):
    """query_boxes (N,7) -> (N, len(PROBE_NAMES)) float32. NaN = probe undefined."""
    N = len(query_boxes)
    out = np.full((N, len(PROBE_NAMES)), np.nan, np.float32)
    if N == 0:
        return out
    col = {k: i for i, k in enumerate(PROBE_NAMES)}
    rng = np.hypot(query_boxes[:, 0], query_boxes[:, 1])
    out[:, col["g_range"]] = rng
    out[:, col["g_azimuth_abs"]] = np.abs(np.arctan2(query_boxes[:, 1], query_boxes[:, 0]))

    # ---- LiDAR: restrict the sweep to a corridor around the query boxes first (speed)
    lo = query_boxes[:, :2].min(0) - 6
    hi = query_boxes[:, :2].max(0) + 6
    sel = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
    pts = points[sel]
    inside = points_in_boxes(pts, query_boxes, enlarge=1.25)
    for i in range(N):
        m = inside[i]
        n = int(m.sum())
        l, h = query_boxes[i, 3], query_boxes[i, 5]
        out[i, col["l_log_pts"]] = np.log1p(n)
        out[i, col["l_density"]] = np.log1p(n * max(rng[i], 1.0) ** 2 / max(l * h, 0.3) / 100.0)
        if n:
            it = pts[m, 3]
            out[i, col["l_mean_int"]] = it.mean() / 255.0
            out[i, col["l_frac_low_int"]] = np.mean(it < 8)
            out[i, col["l_z_std"]] = pts[m, 2].std()
            out[i, col["l_ring_ratio"]] = len(np.unique(pts[m, 4])) / n_rings
        else:
            out[i, [col["l_mean_int"], col["l_frac_low_int"], col["l_z_std"], col["l_ring_ratio"]]] = [0, 1, 0, 0]

    # ---- Camera
    rects, vis = box_image_rect(query_boxes, T_cam_lidar, P, img_stats.w, img_stats.h)
    for i in range(N):
        stats = img_stats.crop(rects[i]) if vis[i] > 0 else None
        out[i, col["c_in_image"]] = float(stats is not None)
        out[i, col["c_vis_frac"]] = vis[i]
        if stats is None:
            continue
        u0, v0, u1, v1 = rects[i]
        out[i, col["c_log_area"]] = np.log1p((u1 - u0) * (v1 - v0))
        out[i, [col["c_lum"], col["c_contrast"], col["c_log_lap"], col["c_dark"], col["c_edge"], col["c_sat"]]] = stats

    # ---- Radar: BEV power map already in the LiDAR frame
    X0, Y0, res = radar_grid.x_min, radar_grid.y_min, radar_grid.res
    for i in range(N):
        x, y = query_boxes[i, :2]
        half = 0.5 * max(query_boxes[i, 3], query_boxes[i, 4]) + 0.5
        ix0, ix1 = int((x - half - X0) / res), int((x + half - X0) / res) + 1
        iy0, iy1 = int((y - half - Y0) / res), int((y + half - Y0) / res) + 1
        if ix1 <= 0 or iy1 <= 0 or ix0 >= radar_grid.nx or iy0 >= radar_grid.ny:
            continue
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        patch = radar_map[ix0:ix1, iy0:iy1]
        ring = radar_map[max(ix0 - 8, 0):ix1 + 8, max(iy0 - 8, 0):iy1 + 8]
        if patch.size == 0:
            continue
        bg = float(np.median(ring))
        out[i, col["r_max_pow"]] = patch.max()
        out[i, col["r_mean_pow"]] = patch.mean()
        out[i, col["r_bg_pow"]] = bg
        out[i, col["r_snr"]] = patch.max() - bg
        out[i, col["r_log_hits"]] = np.log1p(radar_hits_occ[ix0:ix1, iy0:iy1].sum())
    return out
