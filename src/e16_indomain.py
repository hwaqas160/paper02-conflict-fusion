"""E16 (exploratory, not registered): a fully in-domain TruckScenes sensor suite -- camera and radar as well as LiDAR.

E8-C replaced only the LiDAR detector with a cross-fitted in-domain one and found the per-object advantage vanished,
leaving open whether it returns when camera and radar are also matched to the sensors. Training a camera or radar
network is not possible on the shared 5 GB GPU, so the in-domain camera and radar here are LEARNED HEADS on the frozen
detectors' outputs, fitted on TruckScenes labels:
  camera  range correction (monocular lift error), range sigma, and a re-scored detection confidence;
  radar   a re-scored cluster confidence from the cluster features.
Cross-fitting follows ts_detector.py: the head applied to a scene is fitted only on the OTHER detector fold, so no head
has seen the labels of the scenes it scores. Probes stay as cached (computed at the original boxes).

    python -m src.e16_indomain fit            # fit heads, report held-out quality, write results/ts_detector/cache_full
    python -m src.e16_indomain candidates     # candidate tables from the full in-domain cache
    python -m src.e16_indomain router         # five-fold router comparison (same protocol as E8-C)
"""
import json
import sys
from pathlib import Path

import numpy as np

from .common import load_config, resolve

CAR = 0
CAM_MATCH_M, RAD_MATCH_M = 4.0, 2.0


def roots(cfg):
    base = resolve(cfg, "results_root") / "ts_detector"
    return base / "cache" / "truckscenes", base / "cache_full" / "truckscenes", base / "candidates_full" / "truckscenes"


def _match(xy, gt_xy):
    if len(gt_xy) == 0 or len(xy) == 0:
        return np.full(len(xy), np.inf), np.zeros(len(xy), int)
    d = np.linalg.norm(xy[:, None] - gt_xy[None], axis=-1)
    return d.min(1), d.argmin(1)


def cam_feats(s):
    b, b2 = s["camera_boxes"].astype(np.float64), s["camera_boxes2d"].astype(np.float64)
    r = np.hypot(b[:, 0], b[:, 1])
    w, h = b2[:, 2] - b2[:, 0], b2[:, 3] - b2[:, 1]
    return np.column_stack([r, np.arctan2(b[:, 1], b[:, 0]), s["camera_range_ground"], s["camera_range_size"],
                            s["camera_range_sigma"], s["camera_truncated"], s["camera_scores"], b2, w, h,
                            w / np.maximum(h, 1.0), s["camera_labels"]])


def rad_feats(s):
    b = s["radar_boxes"].astype(np.float64)
    return np.column_stack([s["radar_feats"], s["radar_scores"], b[:, 0], b[:, 1], np.hypot(b[:, 0], b[:, 1]), b[:, 3], b[:, 4]])


def load_all(src, fold_of):
    rows = {"cam": [], "rad": []}
    files = sorted(src.glob("*/*.npz"))
    for i, p in enumerate(files):
        z = np.load(p, allow_pickle=False)
        scene = eval(str(z["meta"]))["scene"]
        s = {k: z[k] for k in z.files if k != "meta"}
        gt = s["gt_boxes"][(s["gt_cls"] == CAR) & s["gt_in_roi"].astype(bool)][:, :2] if len(s["gt_boxes"]) else np.zeros((0, 2))
        f = fold_of[scene]
        n = len(s["camera_scores"])
        if n:
            d, j = _match(s["camera_boxes"][:, :2].astype(float), gt)
            rg = np.hypot(gt[j, 0], gt[j, 1]) if len(gt) else np.zeros(n)
            rows["cam"].append((np.full(n, i), np.full(n, f), cam_feats(s), d, rg - np.hypot(*s["camera_boxes"][:, :2].astype(float).T)))
        n = len(s["radar_scores"])
        if n:
            d, _ = _match(s["radar_boxes"][:, :2].astype(float), gt)
            rows["rad"].append((np.full(n, i), np.full(n, f), rad_feats(s), d))
        if i % 4000 == 0:
            print(f"load {i}/{len(files)}", flush=True)
    cat = lambda k, j: np.concatenate([r[j] for r in rows[k]])
    cam = {"frame": cat("cam", 0), "fold": cat("cam", 1), "X": np.concatenate([r[2] for r in rows["cam"]]), "d": cat("cam", 3), "dr": cat("cam", 4)}
    rad = {"frame": cat("rad", 0), "fold": cat("rad", 1), "X": np.concatenate([r[2] for r in rows["rad"]]), "d": cat("rad", 3)}
    return files, cam, rad


def fit_predict(cam, rad):
    from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    out = {"cam_dr": np.zeros(len(cam["d"])), "cam_sig": np.zeros(len(cam["d"])), "cam_score": np.zeros(len(cam["d"])),
           "rad_score": np.zeros(len(rad["d"]))}
    q = {}
    for k in (0, 1):  # head k is fitted on detector fold k and applied to the other fold
        tr_c, te_c = cam["fold"] == k, cam["fold"] != k
        pos = tr_c & (cam["d"] < CAM_MATCH_M)
        reg = HR(max_iter=200, learning_rate=0.08, random_state=0).fit(cam["X"][pos], cam["dr"][pos])
        res = np.abs(cam["dr"][pos] - reg.predict(cam["X"][pos]))
        sig = HR(max_iter=100, learning_rate=0.08, random_state=0).fit(cam["X"][pos], res)
        clf = HC(max_iter=200, learning_rate=0.08, random_state=0).fit(cam["X"][tr_c], (cam["d"][tr_c] < CAM_MATCH_M).astype(int))
        out["cam_dr"][te_c] = reg.predict(cam["X"][te_c])
        out["cam_sig"][te_c] = np.maximum(sig.predict(cam["X"][te_c]), 0.3)
        out["cam_score"][te_c] = clf.predict_proba(cam["X"][te_c])[:, 1]
        tr_r, te_r = np.nonzero(rad["fold"] == k)[0], rad["fold"] != k
        y = (rad["d"][tr_r] < RAD_MATCH_M).astype(int)
        neg = tr_r[y == 0]
        keep = np.concatenate([tr_r[y == 1], rng.choice(neg, min(len(neg), 250000), replace=False)])
        rclf = HC(max_iter=200, learning_rate=0.08, random_state=0).fit(rad["X"][keep], (rad["d"][keep] < RAD_MATCH_M).astype(int))
        out["rad_score"][te_r] = rclf.predict_proba(rad["X"][te_r])[:, 1]
    m = cam["d"] < CAM_MATCH_M
    q["camera_median_abs_range_error_m"] = {"original": float(np.median(np.abs(cam["dr"][m]))),
                                            "learned": float(np.median(np.abs(cam["dr"][m] - out["cam_dr"][m])))}
    q["camera_matched_auroc"] = {"original": float(roc_auc_score(m, cam["X"][:, 6])), "learned": float(roc_auc_score(m, out["cam_score"]))}
    mr = rad["d"] < RAD_MATCH_M
    q["radar_auroc"] = {"original": float(roc_auc_score(mr, rad["X"][:, 7])), "learned": float(roc_auc_score(mr, out["rad_score"]))}
    return out, q


def write_cache(files, cam, rad, out, dst):
    ci, ri = np.cumsum(np.r_[0, np.bincount(cam["frame"], minlength=len(files))]), np.cumsum(np.r_[0, np.bincount(rad["frame"], minlength=len(files))])
    for i, p in enumerate(files):
        z = np.load(p, allow_pickle=False)
        s = {k: z[k] for k in z.files}
        a, b = ci[i], ci[i + 1]
        if b > a:
            bx = s["camera_boxes"].astype(np.float64)
            r0 = np.hypot(bx[:, 0], bx[:, 1])
            r1 = np.maximum(r0 + out["cam_dr"][a:b], 1.0)
            bx[:, 0], bx[:, 1] = bx[:, 0] * r1 / r0, bx[:, 1] * r1 / r0
            s["camera_boxes"] = bx.astype(np.float32)
            s["camera_range_sigma"] = out["cam_sig"][a:b].astype(np.float32)
            s["camera_scores"] = out["cam_score"][a:b].astype(np.float32)
        a, b = ri[i], ri[i + 1]
        if b > a:
            s["radar_scores"] = out["rad_score"][a:b].astype(np.float32)
        t = dst / p.parent.name / p.name
        t.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(t, **s)
        if i % 4000 == 0:
            print(f"write {i}/{len(files)}", flush=True)


def main():
    cmd = sys.argv[1]
    cfg = load_config()
    src, dst, cand = roots(cfg)
    if cmd == "fit":
        from .ts_detector import load_prep
        fold_of = load_prep(cfg)["fold"]
        files, cam, rad = load_all(src, fold_of)
        out, q = fit_predict(cam, rad)
        print(json.dumps(q, indent=2))
        out_dir = resolve(cfg, "results_root") / "r4"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "e16_head_quality.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
        write_cache(files, cam, rad, out, dst)
        (dst / "manifest.json").write_text((src / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    elif cmd == "candidates":
        from .candidates import build_table, save_table
        for tag in sorted(p.name for p in dst.iterdir() if p.is_dir()):
            tbl = build_table(dst / tag, cfg)
            if tbl is not None:
                save_table(tbl, cand / f"{tag}.npz")
                print(tag, len(tbl["frames"]), flush=True)
    else:
        from .e8c_router import run
        r = run(n_boot=10000, n_boot2=1000, cand_name="candidates_full", out_name="e16_fully_indomain_router.json", router_dir="router_full")
        c = r["primary_object_vs_frame_cls"]
        print({k: round(v, 3) for k, v in r["oof_map"].items()})
        print("primary", f"{c['diff']:+.4f}", [round(x, 4) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}",
              f"folds ahead {r['folds_object_ahead_of_frame_cls']}/5")


if __name__ == "__main__":
    main()
