"""Cross-modal association of cached detections into per-object *candidates*.

A candidate holds at most one detection per modality. Association is greedy on a normalised cost:
  * LiDAR–radar:  BEV centre distance (gate 2.5 m);
  * camera–X:     azimuth difference (camera bearing is accurate) and range difference scaled by the
                  camera's own monocular range sigma (camera range is not).
Each candidate gets
  * presence mask, detector scores / logits, member positions, camera range sigma, radar cluster feats,
  * degradation probes read at an *anchor* member (LiDAR > radar > camera: best-localised first),
  * cross-sensor conflict features (pairwise disagreements, normalised),
  * frame-level features, and GT label / GT position (training & evaluation only).
Output: one table per (split, condition) in results/candidates/, flat arrays + frame index.
"""
import argparse
from pathlib import Path

import numpy as np

from .common import MODALITIES, get_logger, iter_conditions, load_config, resolve, condition_tag, save_json
from .degradation import FRAME_FEATURES, PROBE_NAMES
from .geometry import in_roi, wrap_angle

M_CAM, M_LID, M_RAD = 0, 1, 2
LIDAR_SIGMA_M, RADAR_SIGMA_M = 0.35, 0.9
DET_FEATURES = ["cam_log_sigma", "cam_truncated", "cam_range_disagree", "cam_log_h2d",
                "rad_log_hits", "rad_mean_pow", "rad_peak_pow", "rad_snr", "rad_persistence", "rad_extent_l",
                "lid_log_volume"]
CONFLICT_FEATURES = ["k_d_lr", "k_az_cl", "k_rng_cl", "k_az_cr", "k_rng_cr", "k_score_std", "k_n_present",
                     "k_cls_disagree"]


def load_sample(path):
    d = np.load(path, allow_pickle=False)
    meta = eval(str(d["meta"]))  # repr of a plain dict written by cache_detections
    return meta, {k: d[k] for k in d.files if k != "meta"}


def associate(dets, cfg):
    """dets: dict modality -> dict(xy (n,2), sigma_r (n,)). Returns list of [cam, lid, rad] indices (-1 = none)."""
    a = cfg["association"]
    groups = []
    lid_xy, rad_xy, cam_xy = dets["lidar"]["xy"], dets["radar"]["xy"], dets["camera"]["xy"]
    # 1) LiDAR–radar by BEV distance
    pairs = []
    if len(lid_xy) and len(rad_xy):
        d = np.linalg.norm(lid_xy[:, None] - rad_xy[None], axis=-1)
        for i, j in zip(*np.nonzero(d <= a["lidar_radar_gate_m"])):
            pairs.append((d[i, j] / a["lidar_radar_gate_m"], i, j))
    used_l, used_r = set(), set()
    for _, i, j in sorted(pairs):
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        groups.append([-1, i, j])
    groups += [[-1, i, -1] for i in range(len(lid_xy)) if i not in used_l]
    groups += [[-1, -1, j] for j in range(len(rad_xy)) if j not in used_r]
    # 2) camera to groups by bearing + range normalised by camera sigma
    if len(cam_xy):
        r_c = np.hypot(cam_xy[:, 0], cam_xy[:, 1])
        az_c = np.arctan2(cam_xy[:, 1], cam_xy[:, 0])
        sig = dets["camera"]["sigma_r"]
        pairs = []
        for g, (_, li, ri) in enumerate(groups):
            gxy = lid_xy[li] if li >= 0 else rad_xy[ri]
            r_g, az_g = np.hypot(*gxy), np.arctan2(gxy[1], gxy[0])
            daz = np.abs(wrap_angle(az_c - az_g))
            drng = np.abs(r_c - r_g)
            gate_az = np.radians(a["camera_azimuth_gate_deg"]) + 1.0 / np.maximum(r_g, 1.0)  # object angular size
            gate_r = a["camera_range_gate_rel"] * r_g + a["camera_range_gate_abs"]
            ok = (daz <= gate_az) & (drng <= np.maximum(gate_r, 2 * sig))
            for c in np.nonzero(ok)[0]:
                cost = daz[c] / gate_az + drng[c] / max(gate_r, 2 * sig[c]) + (0.0 if li >= 0 else 0.25)
                pairs.append((cost, c, g))
        used_c, used_g = set(), set()
        for _, c, g in sorted(pairs):
            if c in used_c or g in used_g:
                continue
            used_c.add(c)
            used_g.add(g)
            groups[g][0] = c
        groups += [[c, -1, -1] for c in range(len(cam_xy)) if c not in used_c]
    return np.asarray(groups, dtype=np.int64).reshape(-1, 3)


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_sample_candidates(meta, s, cfg):
    roi = cfg["roi"]
    dets, keep = {}, {}
    for m in MODALITIES:
        b = s[f"{m}_boxes"].astype(np.float64)
        k = in_roi(b, roi) if len(b) else np.zeros(0, bool)
        # classes evaluated here: Car / Pedestrian (Cyclist too rare, see config)
        k &= np.isin(s[f"{m}_labels"], [0, 1])
        keep[m] = np.nonzero(k)[0]
        dets[m] = {"xy": b[keep[m], :2], "sigma_r": np.zeros(len(keep[m]))}
    dets["camera"]["sigma_r"] = s["camera_range_sigma"][keep["camera"]].astype(np.float64)
    groups = associate(dets, cfg)
    n = len(groups)
    present = groups >= 0
    score = np.zeros((n, 3), np.float32)
    xy = np.full((n, 3, 2), np.nan, np.float32)
    labels = np.full((n, 3), -1, np.int64)
    sigma_r = np.full((n, 3), np.nan, np.float32)
    probes = np.zeros((n, len(PROBE_NAMES)), np.float32)
    det_feats = np.zeros((n, len(DET_FEATURES)), np.float32)
    lidar_box = np.zeros((n, 7), np.float32)
    for mi, m in enumerate(MODALITIES):
        idx = groups[:, mi]
        has = idx >= 0
        orig = keep[m][idx[has]]
        score[has, mi] = s[f"{m}_scores"][orig]
        xy[has, mi] = s[f"{m}_boxes"][orig, :2]
        labels[has, mi] = s[f"{m}_labels"][orig]
    # anchor: LiDAR > radar > camera
    anchor_m = np.where(present[:, M_LID], M_LID, np.where(present[:, M_RAD], M_RAD, M_CAM))
    for mi, m in enumerate(MODALITIES):
        sel = anchor_m == mi
        if sel.any():
            probes[sel] = s[f"{m}_probes"][keep[m][groups[sel, mi]]]
    has = present[:, M_CAM]
    if has.any():
        o = keep["camera"][groups[has, M_CAM]]
        sig = s["camera_range_sigma"][o]
        sigma_r[has, M_CAM] = sig
        det_feats[has, 0] = np.log(np.maximum(sig, 0.05))
        det_feats[has, 1] = s["camera_truncated"][o]
        rg, rs = s["camera_range_ground"][o], s["camera_range_size"][o]
        det_feats[has, 2] = np.where(np.isfinite(rg), np.abs(np.log(np.maximum(rg, 1) / np.maximum(rs, 1))), 1.0)
        b2 = s["camera_boxes2d"][o]
        det_feats[has, 3] = np.log(np.maximum(b2[:, 3] - b2[:, 1], 1))
    has = present[:, M_RAD]
    if has.any():
        f = s["radar_feats"][keep["radar"][groups[has, M_RAD]]]
        det_feats[has, 4] = np.log1p(f[:, 0])
        det_feats[has, 5:8] = f[:, 1:4]
        det_feats[has, 8] = np.nan_to_num(f[:, 4], nan=0.5)
        det_feats[has, 9] = f[:, 5]
    has = present[:, M_LID]
    sigma_r[has, M_LID] = LIDAR_SIGMA_M
    sigma_r[present[:, M_RAD], M_RAD] = RADAR_SIGMA_M
    if has.any():
        lb = s["lidar_boxes"][keep["lidar"][groups[has, M_LID]]]
        lidar_box[has] = lb
        det_feats[has, 10] = np.log(np.maximum(lb[:, 3] * lb[:, 4] * lb[:, 5], 0.05))

    # candidate class: LiDAR label, else camera, else radar (radar label is an extent heuristic)
    cls = np.where(labels[:, M_LID] >= 0, labels[:, M_LID], np.where(labels[:, M_CAM] >= 0, labels[:, M_CAM], labels[:, M_RAD]))

    # conflict features
    conflict = np.zeros((n, len(CONFLICT_FEATURES)), np.float32)
    def pair(a_, b_):
        both = present[:, a_] & present[:, b_]
        ra, rb = np.hypot(xy[:, a_, 0], xy[:, a_, 1]), np.hypot(xy[:, b_, 0], xy[:, b_, 1])
        daz = np.abs(wrap_angle(np.arctan2(xy[:, a_, 1], xy[:, a_, 0]) - np.arctan2(xy[:, b_, 1], xy[:, b_, 0])))
        return both, np.nan_to_num(ra - rb), np.nan_to_num(daz), np.nan_to_num(np.linalg.norm(xy[:, a_] - xy[:, b_], axis=1))
    both, _, _, d = pair(M_LID, M_RAD)
    conflict[:, 0] = np.where(both, d / RADAR_SIGMA_M, 0)
    for k, other in ((1, M_LID), (3, M_RAD)):
        both, drng, daz, _ = pair(M_CAM, other)
        conflict[:, k] = np.where(both, np.degrees(daz), 0)
        conflict[:, k + 1] = np.where(both, np.abs(drng) / np.maximum(np.nan_to_num(sigma_r[:, M_CAM], nan=1.0), 0.3), 0)
    sc = np.where(present, score, np.nan)
    with np.errstate(invalid="ignore"):
        conflict[:, 5] = np.nan_to_num(np.nanstd(sc, axis=1), nan=0.0)
    conflict[:, 6] = present.sum(1)
    lab_present = np.where(present, labels, -1)
    conflict[:, 7] = np.array([len(set(r[r >= 0])) > 1 for r in lab_present], np.float32)

    # in field of view (sensor could have seen it): camera from the image probe; LiDAR/radar ROI ~ always
    fov = np.ones((n, 3), np.float32)
    fov[:, M_CAM] = np.nan_to_num(probes[:, PROBE_NAMES.index("c_in_image")], nan=0.0)

    # GT labels: per class, greedy by max member score, best member distance
    gt_boxes, gt_cls = s["gt_boxes"], s["gt_cls"]
    groi = s["gt_in_roi"].astype(bool) if len(gt_boxes) else np.zeros(0, bool)
    y = np.zeros(n, np.float32)
    gt_xy = np.full((n, 2), np.nan, np.float32)
    gt_npts = np.zeros(n, np.float32)
    ignore = np.zeros(n, bool)
    dontcare = s["gt_dontcare"]
    best_member_d = np.full(n, np.inf)
    thr_of = {0: cfg["matching"]["center_dist_m"]["Car"], 1: cfg["matching"]["center_dist_m"]["Pedestrian"]}
    order = np.argsort(-np.nanmax(np.where(present, score, -1), axis=1))
    for c, thr in thr_of.items():
        gmask = (gt_cls == c) & groi
        g_xy = gt_boxes[gmask, :2] if len(gt_boxes) else np.zeros((0, 2))
        g_n = s["gt_npts"][gmask] if len(gt_boxes) else np.zeros(0)
        taken = np.zeros(len(g_xy), bool)
        for i in order:
            if cls[i] != c:
                continue
            if len(g_xy):
                dm = np.linalg.norm(xy[i][present[i]][:, None] - g_xy[None], axis=-1).min(0)
                dm[taken] = np.inf
                j = int(np.argmin(dm))
                if dm[j] <= thr:
                    taken[j] = True
                    y[i], gt_xy[i], gt_npts[i], best_member_d[i] = 1, g_xy[j], g_n[j], dm[j]
                    continue
            if len(dontcare):
                if np.linalg.norm(xy[i][present[i]][:, None] - dontcare[None, :, :2], axis=-1).min() <= thr:
                    ignore[i] = True
    frame_feats = np.repeat(s["frame_feats"][None], n, 0)
    return {"present": present, "score": score, "xy": xy, "sigma_r": sigma_r, "cls": cls.astype(np.int8),
            "probes": probes, "det_feats": det_feats, "conflict": conflict, "fov": fov, "frame_feats": frame_feats,
            "lidar_box": lidar_box, "y": y, "gt_xy": gt_xy, "gt_npts": gt_npts, "ignore": ignore}


def build_table(cache_dir, cfg, frame_filter=None):
    rows, gts, frames = [], [], []
    for path in sorted(Path(cache_dir).glob("*.npz")):
        meta, s = load_sample(path)
        if frame_filter and meta["frame_id"] not in frame_filter:
            continue
        c = build_sample_candidates(meta, s, cfg)
        fi = len(frames)
        c["frame"] = np.full(len(c["y"]), fi, np.int64)
        rows.append(c)
        groi = s["gt_in_roi"].astype(bool) if len(s["gt_boxes"]) else np.zeros(0, bool)
        gts.append({"xy": s["gt_boxes"][groi, :2], "cls": s["gt_cls"][groi], "npts": s["gt_npts"][groi],
                    "dontcare_xy": s["gt_dontcare"][:, :2]})
        frames.append({"frame_id": meta["frame_id"], "scene": meta["scene"], "split": meta["split"],
                       "condition": meta["condition"], "severity": meta["severity"]})
    if not rows:
        return None
    table = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return {"cand": table, "gts": gts, "frames": frames}


def save_table(tbl, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"cand__{k}": v for k, v in tbl["cand"].items()}
    lengths = np.array([len(g["xy"]) for g in tbl["gts"]])
    flat["gt_xy"] = np.concatenate([g["xy"] for g in tbl["gts"]]) if lengths.sum() else np.zeros((0, 2))
    flat["gt_cls"] = np.concatenate([g["cls"] for g in tbl["gts"]]) if lengths.sum() else np.zeros(0)
    flat["gt_npts"] = np.concatenate([g["npts"] for g in tbl["gts"]]) if lengths.sum() else np.zeros(0)
    flat["gt_len"] = lengths
    dcl = np.array([len(g["dontcare_xy"]) for g in tbl["gts"]])
    flat["dc_xy"] = np.concatenate([g["dontcare_xy"] for g in tbl["gts"]]) if dcl.sum() else np.zeros((0, 2))
    flat["dc_len"] = dcl
    flat["frames"] = np.array(repr(tbl["frames"]))
    np.savez_compressed(path, **flat)


def load_table(path):
    d = np.load(path)
    cand = {k[len("cand__"):]: d[k] for k in d.files if k.startswith("cand__")}
    splits = np.cumsum(d["gt_len"])[:-1]
    dsplits = np.cumsum(d["dc_len"])[:-1]
    xy, cl, npt = np.split(d["gt_xy"], splits), np.split(d["gt_cls"], splits), np.split(d["gt_npts"], splits)
    dc = np.split(d["dc_xy"], dsplits)
    gts = [{"xy": a, "cls": b, "npts": c, "dontcare_xy": e} for a, b, c, e in zip(xy, cl, npt, dc)]
    return {"cand": cand, "gts": gts, "frames": eval(str(d["frames"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="boreas")
    ap.add_argument("--manifest", default=None,
                    help="frame list to restrict to (default: <cache>/manifest.json); e.g. manifest_frozen1654.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("candidates")
    root = resolve(cfg, "cache_root") / args.dataset
    out_root = resolve(cfg, "candidates_root") / args.dataset
    summary = {}
    tags = [condition_tag(c, s) for c, s in iter_conditions(cfg)] if args.dataset == "boreas" else \
        sorted(p.name for p in root.iterdir() if p.is_dir())
    # Only frames on the cache manifest's frozen list, so every condition covers the SAME frames.
    frame_filter = None
    manifest = Path(args.manifest) if args.manifest else root / "manifest.json"
    if not manifest.is_absolute() and args.manifest:
        manifest = root / manifest
    if manifest.exists():
        from .common import load_json
        ids = load_json(manifest).get("frame_ids")
        frame_filter = set(ids) if ids else None
        if frame_filter:
            log.info(f"restricting to {len(frame_filter)} manifest frames")
    for tag in tags:
        if not (root / tag).exists():
            continue
        tbl = build_table(root / tag, cfg, frame_filter)
        if tbl is None:
            continue
        save_table(tbl, out_root / f"{tag}.npz")
        c = tbl["cand"]
        summary[tag] = {"frames": len(tbl["frames"]), "candidates": int(len(c["y"])), "positives": int(c["y"].sum()),
                        "gt_objects": int(sum(len(g["xy"]) for g in tbl["gts"]))}
        log.info(f"{tag}: {summary[tag]}")
    save_json(summary, out_root / "summary.json")


if __name__ == "__main__":
    main()
