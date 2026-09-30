"""Training-free detector for 4D radar point clouds (MAN TruckScenes), same pipeline and output schema as the
Navtech CA-CFAR detector (radar_cfar.RadarDetector) so the router sees identically defined radar inputs:

  points (ref frame, [x, y, z, v_radial, rcs]) -> power = RCS normalised to [0, 1]
  -> BEV occupancy -> 8-connected components -> PCA box per component
  -> score = hit-count term x power term, with a Doppler bonus for moving clusters.

All constants are fixed a priori (2026-09-23, Registration 2), never tuned on TruckScenes results. The radar
already provides detections as points (CFAR is done on-sensor), so no CFAR stage is needed.
"""
import numpy as np
from scipy import ndimage

from .radar_cfar import BEVGrid

RCS_LO, RCS_HI = -20.0, 30.0   # dBsm range mapped to power [0, 1]
MOVING_MPS = 0.5


def rcs_power(rcs):
    return np.clip((rcs - RCS_LO) / (RCS_HI - RCS_LO), 0.0, 1.0)


class RadarPoints4D:
    def __init__(self, cfg, roi, z_ground):
        self.cfg, self.z_ground = cfg, z_ground
        pad = 5.0
        self.grid = BEVGrid(roi["x_min"] - pad, roi["x_max"] + pad, -roi["y_abs_max"] - pad, roi["y_abs_max"] + pad,
                            cfg["grid_res_m"])

    def _occ(self, pts):
        occ = np.zeros((self.grid.nx, self.grid.ny), np.int32)
        ix, iy, ok = self.grid.cell(pts)
        np.add.at(occ, (ix[ok], iy[ok]), 1)
        return occ

    def power_map(self, radar):
        """BEV max-power raster (for the shared radar probes)."""
        pm = np.zeros((self.grid.nx, self.grid.ny), np.float32)
        if len(radar):
            ix, iy, ok = self.grid.cell(radar[:, :2])
            np.maximum.at(pm, (ix[ok], iy[ok]), rcs_power(radar[ok, 4]).astype(np.float32))
        return pm

    def __call__(self, radar, radar_prev=None):
        z_ok = np.abs(radar[:, 2] - (self.z_ground + 1.0)) < 3.0 if len(radar) else np.zeros(0, bool)
        pts = radar[z_ok]
        ix, iy, ok = self.grid.cell(pts)
        pts, ix, iy = pts[ok], ix[ok], iy[ok]
        power = rcs_power(pts[:, 4])
        occ = self._occ(pts)
        noise = np.median(power) if len(power) else 0.0
        snr = power - noise
        labels, n_comp = ndimage.label(ndimage.binary_dilation(occ > 0, structure=np.ones((3, 3))),
                                       structure=np.ones((3, 3)))
        prev_occ = None
        if radar_prev is not None and len(radar_prev):
            prev_occ = ndimage.binary_dilation(self._occ(radar_prev) > 0, structure=np.ones((3, 3)))
        comp = labels[ix, iy] if len(pts) else np.zeros(0, int)
        order = np.argsort(comp)
        cs = comp[order]
        bounds = np.searchsorted(cs, np.arange(1, n_comp + 2))
        boxes, scores, labs, feats = [], [], [], []
        for k in range(1, n_comp + 1):
            sel = order[bounds[k - 1]:bounds[k]]
            if len(sel) < 2:
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
            moving = float(np.median(np.abs(pts[sel, 3])) > MOVING_MPS)
            score = (1 - np.exp(-len(sel) / 6.0)) / (1 + np.exp(-(mean_p - 0.35) / 0.08))
            score = min(1.0, score * (1.0 + 0.5 * moving))
            boxes.append([centre[0], centre[1], self.z_ground + 0.8, max(length, 1.0), max(width, 0.6), 1.6, yaw])
            scores.append(score)
            labs.append(0 if length >= 1.5 or moving else 1)
            feats.append([len(sel), mean_p, peak_p, float(snr[sel].mean()), persistence, length, width])
        n = len(boxes)
        return {"boxes": np.asarray(boxes, float).reshape(n, 7), "scores": np.asarray(scores, float),
                "labels": np.asarray(labs, int), "feats": np.asarray(feats, float).reshape(n, 7), "occupancy": occ,
                "power_map": self.power_map(pts),
                "frame": {"n_hits_roi": int(len(pts)), "noise_floor_median": float(noise),
                          "hit_mean_power": float(power.mean()) if len(power) else 0.0}}
