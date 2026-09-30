"""Training-free radar detector for the Navtech CIR304-H scanning radar.

No pretrained object detector exists for Navtech polar scans (PLAN.md §13 risk), so the radar
stream is a classical, parameter-free-at-inference pipeline — this is still a *frozen*
off-the-shelf detector in the sense the paper needs (nothing is fitted to Boreas labels):

  1. 1-D cell-averaging CFAR along range for every azimuth (Richards, "Fundamentals of Radar
     Signal Processing", ch. 7), in the normalised log-power domain Boreas stores.
  2. CFAR hits -> Cartesian radar frame -> LiDAR frame at the LiDAR timestamp via Applanix poses.
  3. BEV occupancy grid, connected components, PCA box per component.
  4. Score: monotone function of hit count and mean power; label from extent.

Boreas' Navtech provides no Doppler. The "Doppler consistency" cue of PLAN.md is replaced by
*temporal persistence*: the fraction of a cluster's cells also occupied in the previous scan
after ego-motion compensation (static structure ~1, moving objects lower).
"""
import numpy as np
from scipy import ndimage

from ..geometry import apply_tf, inv_tf


def ca_cfar(polar, guard, train, offset, min_power, max_bin):
    """Returns (A, R) bool hits and (A, R) local noise estimate."""
    p = polar[:, :max_bin].astype(np.float64)
    A, R = p.shape
    c = np.concatenate([np.zeros((A, 1)), np.cumsum(p, axis=1)], axis=1)

    def window_sum(lo, hi):
        lo = np.clip(lo, 0, R)
        hi = np.clip(hi, 0, R)
        return c[:, hi] - c[:, lo], (hi - lo)

    idx = np.arange(R)
    s_left, n_left = window_sum(idx - guard - train, idx - guard)
    s_right, n_right = window_sum(idx + guard + 1, idx + guard + 1 + train)
    noise = (s_left + s_right) / np.maximum(n_left + n_right, 1)
    hits = (p > noise + offset) & (p > min_power)
    return hits, noise


def radar_points(radar, cfg, T_lidar_radar, z_ground):
    res = radar["resolution"]
    max_bin = min(int(cfg["max_range_m"] / res), radar["polar"].shape[1])
    hits, noise = ca_cfar(radar["polar"], cfg["cfar_guard"], cfg["cfar_train"], cfg["cfar_offset"],
                          cfg["min_power"], max_bin)
    a_idx, r_idx = np.nonzero(hits)
    rng = (r_idx + 0.5) * res
    az = radar["azimuths"][a_idx]
    pts_r = np.stack([rng * np.cos(az), rng * np.sin(az), np.zeros_like(rng)], axis=1)
    pts_l = apply_tf(T_lidar_radar, pts_r)
    pts_l[:, 2] = z_ground + 0.5
    power = radar["polar"][a_idx, r_idx]
    snr = power - noise[a_idx, r_idx]
    return pts_l, power, snr, noise


class BEVGrid:
    def __init__(self, x_min, x_max, y_min, y_max, res):
        self.x_min, self.y_min, self.res = x_min, y_min, res
        self.nx = int(np.ceil((x_max - x_min) / res))
        self.ny = int(np.ceil((y_max - y_min) / res))

    def cell(self, xy):
        ix = np.floor((xy[:, 0] - self.x_min) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y_min) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def centres(self):
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")


def power_map_lidar_frame(radar, T_lidar_radar, grid, z_ground):
    """Sample the polar scan at every BEV cell centre (LiDAR frame) -> (nx, ny) power in [0,1]."""
    X, Y = grid.centres()
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_ground + 0.5)], axis=1)
    pr = apply_tf(inv_tf(T_lidar_radar), pts)
    r = np.hypot(pr[:, 0], pr[:, 1])
    th = np.mod(np.arctan2(pr[:, 1], pr[:, 0]), 2 * np.pi)
    az = np.mod(radar["azimuths"], 2 * np.pi)
    order = np.argsort(az)
    az_sorted = az[order]
    row = np.interp(th, az_sorted, np.arange(len(az_sorted)), period=None)
    row = np.clip(np.round(row).astype(int), 0, len(az) - 1)
    col = np.clip((r / radar["resolution"] - 0.5).round().astype(int), 0, radar["polar"].shape[1] - 1)
    vals = radar["polar"][order[row], col]
    vals[r > radar["polar"].shape[1] * radar["resolution"]] = 0
    return vals.reshape(X.shape).astype(np.float32)


class RadarDetector:
    def __init__(self, cfg, roi, z_ground):
        self.cfg, self.z_ground = cfg, z_ground
        pad = 5.0
        self.grid = BEVGrid(roi["x_min"] - pad, roi["x_max"] + pad, -roi["y_abs_max"] - pad, roi["y_abs_max"] + pad,
                            cfg["grid_res_m"])

    def _occupancy(self, pts):
        occ = np.zeros((self.grid.nx, self.grid.ny), np.int32)
        ix, iy, ok = self.grid.cell(pts)
        np.add.at(occ, (ix[ok], iy[ok]), 1)
        return occ

    def __call__(self, radar, T_lidar_radar, radar_prev=None, T_lidar_radar_prev=None):
        pts, power, snr, noise = radar_points(radar, self.cfg, T_lidar_radar, self.z_ground)
        ix, iy, ok = self.grid.cell(pts)
        pts, power, snr, ix, iy = pts[ok], power[ok], snr[ok], ix[ok], iy[ok]
        occ = self._occupancy(pts)
        labels, n_comp = ndimage.label(ndimage.binary_dilation(occ > 0, structure=np.ones((3, 3))),
                                       structure=np.ones((3, 3)))
        prev_occ = None
        if radar_prev is not None:
            pp, *_ = radar_points(radar_prev, self.cfg, T_lidar_radar_prev, self.z_ground)
            prev_occ = ndimage.binary_dilation(self._occupancy(pp) > 0, structure=np.ones((3, 3)))
        comp_of_pt = labels[ix, iy]
        boxes, scores, labs, feats = [], [], [], []
        order = np.argsort(comp_of_pt)
        comp_sorted = comp_of_pt[order]
        bounds = np.searchsorted(comp_sorted, np.arange(1, n_comp + 2))
        for k in range(1, n_comp + 1):
            sel = order[bounds[k - 1]:bounds[k]]
            if len(sel) < self.cfg["min_points"]:
                continue
            xy = pts[sel, :2]
            w = power[sel] + 1e-3
            centre = (xy * w[:, None]).sum(0) / w.sum()
            cov = np.cov((xy - centre).T) if len(sel) > 2 else np.eye(2) * 0.05
            evals, evecs = np.linalg.eigh(cov + 1e-4 * np.eye(2))
            length = float(np.clip(4 * np.sqrt(evals[1]), 0.5, None))
            width = float(np.clip(4 * np.sqrt(evals[0]), 0.4, None))
            if length > self.cfg["max_extent_m"] or width > 0.6 * self.cfg["max_extent_m"]:
                continue
            yaw = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
            cells = np.unique(np.stack([ix[sel], iy[sel]], 1), axis=0)
            persistence = float(prev_occ[cells[:, 0], cells[:, 1]].mean()) if prev_occ is not None else np.nan
            mean_p, peak_p = float(power[sel].mean()), float(power[sel].max())
            score = (1 - np.exp(-len(sel) / 12.0)) / (1 + np.exp(-(mean_p - 0.35) / 0.06))
            cls = 0 if length >= 1.5 else 1
            boxes.append([centre[0], centre[1], self.z_ground + 0.8, max(length, 1.0), max(width, 0.6), 1.6, yaw])
            scores.append(score)
            labs.append(cls)
            feats.append([len(sel), mean_p, peak_p, float(snr[sel].mean()), persistence, length, width])
        n = len(boxes)
        return {"boxes": np.asarray(boxes, float).reshape(n, 7), "scores": np.asarray(scores, float),
                "labels": np.asarray(labs, int), "feats": np.asarray(feats, float).reshape(n, 7),
                "feat_names": ["n_hits", "mean_power", "peak_power", "mean_snr", "persistence", "extent_l", "extent_w"],
                "occupancy": occ,
                "frame": {"n_hits_roi": int(len(pts)), "noise_floor_median": float(np.median(noise)),
                          "hit_mean_power": float(power.mean()) if len(power) else 0.0}}
