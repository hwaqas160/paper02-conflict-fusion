"""Synthetic detector cache in the exact on-disk format of cache_detections.py.

FOR TESTS ONLY — exercises association, fusion, harm, conformal and every experiment end to end
without Boreas. Nothing produced from this module may be reported as a result.

Ground truth is constructed so the paper's premise holds by design (camera degrades with weather
AND range, LiDAR with fog/snow, radar barely), letting tests assert the pipeline can detect it.
"""
import zlib
from pathlib import Path

import numpy as np

from .degradation import FRAME_FEATURES, PROBE_NAMES

QUALITY = {  # (camera, lidar, radar) quality in [0,1]
    "clear0": (1.0, 1.0, 0.8), "fog1": (0.6, 0.9, 0.8), "fog2": (0.35, 0.8, 0.8), "fog3": (0.15, 0.6, 0.78),
    "rain1": (0.85, 0.95, 0.78), "rain2": (0.7, 0.9, 0.75), "rain3": (0.5, 0.85, 0.7),
    "snow1": (0.8, 0.85, 0.78), "snow2": (0.55, 0.7, 0.75), "snow3": (0.3, 0.5, 0.7), "night2": (0.3, 1.0, 0.8),
}


def _probe_row(rng, r, q, blob):
    p = dict.fromkeys(PROBE_NAMES, 0.0)
    qc, ql, qr = q
    qc_local = qc * (0.2 if blob else 1.0)
    p.update(g_range=r, g_azimuth_abs=rng.uniform(0, 0.6), l_log_pts=np.log1p(max(0, 400 * ql / (1 + r / 10) + rng.normal(0, 5))),
             l_density=ql + rng.normal(0, 0.05), l_mean_int=0.2 * ql, l_frac_low_int=1 - ql, l_z_std=0.4, l_ring_ratio=0.1,
             c_in_image=1.0, c_vis_frac=1.0, c_log_area=8 - np.log1p(r), c_lum=0.4 + 0.4 * (1 - qc_local),
             c_contrast=0.5 * qc_local + rng.normal(0, 0.03), c_log_lap=-6 + 3 * qc_local, c_dark=0.4 * (1 - qc_local),
             c_edge=0.3 * qc_local, c_sat=0.3 * qc_local, r_max_pow=0.5 * qr + rng.normal(0, 0.05), r_mean_pow=0.3,
             r_bg_pow=0.1, r_snr=0.4 * qr, r_log_hits=2.0)
    return np.array([p[k] for k in PROBE_NAMES], np.float32)


def make_synthetic_cache(root, n_frames=60, tags=None, seed=0, n_scenes=12):
    root = Path(root)
    tags = tags or list(QUALITY)
    rng0 = np.random.default_rng(seed)
    scenes = rng0.integers(0, n_scenes, n_frames)
    split_of = {s: ("train" if s < n_scenes * 0.45 else "cal" if s < n_scenes * 0.65 else "test") for s in range(n_scenes)}
    base = []
    for f in range(n_frames):
        n = int(rng0.integers(3, 9))
        r = rng0.uniform(6, 65, n)
        az = rng0.uniform(-0.6, 0.6, n)
        base.append(np.stack([r * np.cos(az), r * np.sin(az)], 1))
    for tag in tags:
        q = QUALITY[tag]
        (root / tag).mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            rng = np.random.default_rng(zlib.crc32(f"{seed}|{tag}|{f}".encode()))
            gxy = base[f]
            ng = len(gxy)
            gr = np.hypot(gxy[:, 0], gxy[:, 1])
            blob = rng.random() < (0.5 if tag.startswith("snow") and tag != "snow1" else 0.0)
            blob_side = rng.choice([-1, 1])
            dets = {}
            for m, name in enumerate(("camera", "lidar", "radar")):
                qm = q[m]
                boxes, scores, labels, sig = [], [], [], []
                for i in range(ng):
                    local_q = qm
                    if name == "camera":
                        local_q *= np.exp(-gr[i] / (80 * max(q[0], 0.2)))
                        if blob and np.sign(gxy[i, 1]) == blob_side:
                            local_q *= 0.2
                    if name == "lidar":
                        local_q *= np.exp(-gr[i] / (150 * q[1]))
                    if rng.random() < 0.3 + 0.65 * local_q:
                        if name == "camera":
                            s_r = 0.02 * gr[i] ** 1.5 / max(q[0], 0.3) ** 0.5
                            rr = gr[i] + rng.normal(0, s_r)
                            az = np.arctan2(gxy[i, 1], gxy[i, 0]) + rng.normal(0, 0.005)
                            xy = np.array([rr * np.cos(az), rr * np.sin(az)])
                            sig.append(max(s_r, 0.2))
                        else:
                            xy = gxy[i] + rng.normal(0, 0.25 if name == "lidar" else 0.8, 2)
                        boxes.append([xy[0], xy[1], -1.5, 4.5, 1.9, 1.6, 0])
                        scores.append(np.clip(0.35 + 0.6 * local_q + rng.normal(0, 0.1), 0.06, 0.99))
                        labels.append(0)
                n_fp = rng.poisson({"camera": 1 + 6 * (1 - q[0]), "lidar": 1 + 4 * (1 - q[1]), "radar": 8}[name])
                for _ in range(n_fp):
                    r, az = rng.uniform(5, 65), rng.uniform(-0.6, 0.6)
                    boxes.append([r * np.cos(az), r * np.sin(az), -1.5, 4.5, 1.9, 1.6, 0])
                    hi = name == "camera" and (1 - q[0]) > 0.4
                    scores.append(np.clip(rng.uniform(0.3, 0.9) if hi else rng.uniform(0.05, 0.5), 0.06, 0.99))
                    labels.append(0)
                    if name == "camera":
                        sig.append(0.02 * r ** 1.5)
                b = np.asarray(boxes, np.float32).reshape(-1, 7)
                dets[name] = (b, np.asarray(scores, np.float32), np.asarray(labels, np.int8), np.asarray(sig, np.float32))
            arrays = {"frame_feats": np.array([12 * q[1], 0.01 * (1 - q[1]), 0.05, 10 * q[1], 0.3 + 0.3 * (1 - q[0]), 0.9 * q[0],
                                               -3 - 2 * (1 - q[0]), 0.03 + 0.2 * (1 - q[0]), 0.2, 0.04, 8.6, 0.29], np.float32)
                      + rng.normal(0, 0.02, len(FRAME_FEATURES)).astype(np.float32),
                      "gt_boxes": np.concatenate([gxy, np.tile([[-1.5, 4.5, 1.9, 1.6, 0]], (ng, 1))], 1),
                      "gt_cls": np.zeros(ng, np.int64), "gt_npts": np.full(ng, 100), "gt_dontcare": np.zeros((0, 7)),
                      "gt_probes": np.zeros((ng, len(PROBE_NAMES)), np.float32), "gt_in_roi": np.ones(ng, bool)}
            for name, (b, s, l, sig) in dets.items():
                arrays[f"{name}_boxes"], arrays[f"{name}_scores"], arrays[f"{name}_labels"] = b, s, l
                arrays[f"{name}_probes"] = np.stack([_probe_row(rng, float(np.hypot(*x[:2])), q,
                                                                blob and np.sign(x[1]) == blob_side) for x in b]) \
                    if len(b) else np.zeros((0, len(PROBE_NAMES)), np.float32)
            nc = len(dets["camera"][0])
            arrays["camera_range_ground"] = np.hypot(dets["camera"][0][:, 0], dets["camera"][0][:, 1])
            arrays["camera_range_size"] = arrays["camera_range_ground"] * 1.05
            arrays["camera_range_sigma"] = dets["camera"][3]
            arrays["camera_truncated"] = np.zeros(nc, np.float32)
            arrays["camera_boxes2d"] = np.tile([[100, 100, 200, 180]], (nc, 1)).astype(np.float32)
            arrays["radar_feats"] = np.tile([[8, 0.4, 0.6, 0.2, 0.5, 3.0, 1.5]], (len(dets["radar"][0]), 1)).astype(np.float32)
            cond = tag.rstrip("0123456789")
            meta = {"frame_id": str(1000 + f), "scene": int(scenes[f]), "split": split_of[int(scenes[f])],
                    "condition": cond, "severity": int(tag[len(cond):]), "timings": {}}
            np.savez_compressed(root / tag / f"{1000 + f}.npz", meta=np.array(repr(meta)), **arrays)
    return root
