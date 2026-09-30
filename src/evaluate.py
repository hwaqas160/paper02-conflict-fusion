"""Turning fused candidate predictions into detections, decisions and metrics."""
import numpy as np

from .metrics import ap_from_matches, match_frame, mean_ap_center


def class_thr(cfg, class_id):
    return cfg["matching"]["center_dist_m"]["Car" if class_id == 0 else "Pedestrian"]


def frame_detections(tbl, pred, class_id, nms_m=1.0, score_floor=1e-3):
    """Per-frame (xy, score, cand_index) after centre-distance NMS."""
    c = tbl["cand"]
    out = [(np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)) for _ in tbl["frames"]]
    sel = np.nonzero((c["cls"] == class_id) & (pred["score"] > score_floor))[0]
    if len(sel) == 0:
        return out
    order = sel[np.argsort(c["frame"][sel], kind="stable")]
    frames = c["frame"][order]
    bounds = np.searchsorted(frames, np.arange(len(tbl["frames"]) + 1))
    for f in range(len(tbl["frames"])):
        idx = order[bounds[f]:bounds[f + 1]]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-pred["score"][idx])]
        xy = pred["xy"][idx]
        keep = []
        for k in range(len(idx)):
            if all(np.linalg.norm(xy[k] - xy[j]) > nms_m for j in keep):
                keep.append(k)
        keep = np.asarray(keep)
        out[f] = (xy[keep], pred["score"][idx][keep], idx[keep])
    return out


def gt_lists(tbl, class_id):
    gts = [g["xy"][g["cls"] == class_id] for g in tbl["gts"]]
    dcs = [np.concatenate([g["dontcare_xy"], g["xy"][(g["cls"] != class_id) & (g["cls"] < 0)]]) for g in tbl["gts"]]
    return gts, dcs


def eval_map(tbl, pred, cfg, class_id=0, frames=None):
    dets = frame_detections(tbl, pred, class_id, cfg["association"]["nms_center_m"])
    gts, dcs = gt_lists(tbl, class_id)
    idx = range(len(dets)) if frames is None else frames
    preds = [(dets[i][0], dets[i][1]) for i in idx]
    mAP, aps = mean_ap_center(preds, [gts[i] for i in idx], cfg["matching"]["ap_thresholds_m"], [dcs[i] for i in idx])
    return mAP, aps


def per_frame_units(tbl, pred, cfg, class_id=0):
    """Units for the frame-cluster bootstrap. Matching is per frame and does not change under resampling,
    so each unit stores, per distance threshold, (kept scores, TP flags, #GT)."""
    dets = frame_detections(tbl, pred, class_id, cfg["association"]["nms_center_m"])
    gts, dcs = gt_lists(tbl, class_id)
    units = []
    for d, g, dc in zip(dets, gts, dcs):
        per_thr = []
        for thr in cfg["matching"]["ap_thresholds_m"]:
            tp, ign, _ = match_frame(d[0].reshape(-1, 2), d[1], g.reshape(-1, 2), thr, dc)
            per_thr.append((d[1][~ign], tp[~ign], len(g)))
        units.append(per_thr)
    return units


def map_of_units(units, cfg):
    aps = []
    for k in range(len(cfg["matching"]["ap_thresholds_m"])):
        s = np.concatenate([u[k][0] for u in units]) if units else np.zeros(0)
        tp = np.concatenate([u[k][1] for u in units]) if units else np.zeros(0, bool)
        n_gt = sum(u[k][2] for u in units)
        aps.append(ap_from_matches(s, tp, n_gt))
    aps = [a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else float("nan")


def decisions(tbl, pred, tau, cfg):
    """Correctness of each candidate's fused decision at threshold tau.
    correct = (GT-backed AND score >= tau AND fused centre within radius) OR (no GT AND score < tau)."""
    c = tbl["cand"]
    thr = np.where(c["cls"] == 0, cfg["matching"]["center_dist_m"]["Car"], cfg["matching"]["center_dist_m"]["Pedestrian"])
    has_gt = c["y"] > 0
    dist = np.linalg.norm(pred["xy"] - np.nan_to_num(c["gt_xy"], nan=1e6), axis=1)
    on = pred["score"] >= tau
    correct = np.where(has_gt, on & (dist <= thr), ~on)
    return correct & ~c["ignore"] | c["ignore"]


def choose_tau_f1(tbl, pred, cfg, class_id=0):
    c = tbl["cand"]
    m = (c["cls"] == class_id) & ~c["ignore"]
    thr = class_thr(cfg, class_id)
    dist = np.linalg.norm(pred["xy"][m] - np.nan_to_num(c["gt_xy"][m], nan=1e6), axis=1)
    good = (c["y"][m] > 0) & (dist <= thr)
    n_gt = sum(int((g["cls"] == class_id).sum()) for g in tbl["gts"])
    s = pred["score"][m]
    order = np.argsort(-s)
    tp = np.cumsum(good[order])
    fp = np.cumsum(~good[order])
    f1 = 2 * tp / np.maximum(tp + fp + n_gt, 1)
    k = int(np.argmax(f1))
    return float(s[order][k]), float(f1[k])
