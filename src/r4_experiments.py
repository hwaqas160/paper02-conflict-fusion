"""Registration 4 helpers: an information-equal per-frame arm.

The per-object router sees every per-candidate input (probes, detector features, conflict features,
confidences, frame features). A per-frame arm that sees only frame-level features tests a different
information pool, so the decision-unit claim is confounded with input. `frame_pooled` removes that confound:
it receives the same per-candidate pool, pooled over each frame's candidates (mean, std, 10th and 90th
percentile per input), plus the frame-level features. Every candidate in a frame therefore receives identical
inputs and identical fusion parameters, so the decision is per frame by construction.

Nothing in arbitrate.py is edited. The arm is registered at run time, so cached models for existing arms keep
their fingerprints.
"""
import numpy as np

from . import arbitrate as A

POOL_STATS = ("mean", "std", "p10", "p90")


def pooled_inputs(tbl):
    c = tbl["cand"]
    X = A.object_inputs(c, A.ALL_BLOCKS)
    frames = c["frame"]
    n_cand, d = X.shape
    ff = A._frame_feat_matrix(tbl)
    out = np.zeros((n_cand, len(POOL_STATS) * d + ff.shape[1]), np.float32)
    if n_cand == 0:
        return out
    order = np.argsort(frames, kind="stable")
    sorted_frames = frames[order]
    bounds = np.flatnonzero(np.diff(sorted_frames)) + 1
    for rows in np.split(order, bounds):
        f = frames[rows[0]]
        block = X[rows]
        stats = np.concatenate([
            np.nanmean(block, 0), np.nanstd(block, 0),
            np.nanpercentile(block, 10, axis=0), np.nanpercentile(block, 90, axis=0),
        ]) if len(rows) > 1 else np.concatenate([block[0], np.zeros(d), block[0], block[0]])
        out[rows] = np.concatenate([np.nan_to_num(stats), ff[f]])
    return out


_ORIGINAL_ARM_INPUTS = A.arm_inputs


def _arm_inputs(arm_blocks, tbl, weather_clf=None):
    if "frame_pooled" in arm_blocks:
        return pooled_inputs(tbl)
    return _ORIGINAL_ARM_INPUTS(arm_blocks, tbl, weather_clf)


def register():
    """Make the arm available to the existing fitting code. Idempotent."""
    A.ARM_BLOCKS["frame_pooled"] = ["frame_pooled"]
    A.arm_inputs = _arm_inputs


def run_r4_1(cfg, n_boot=10000):
    """R4-1 (Boreas): information-equal per-frame arm vs per-object arm, adverse pooled Car mAP.

    Registered in notes/registration4.md. Primary test: one-sided scene-cluster bootstrap (whole scenes resampled).
    Refuted if the mean difference is <= 0 or that p is >= 0.05. The frame-cluster p is reported as secondary.
    """
    import json
    from .run_experiments import Ctx
    from .evaluate import map_of_units, per_frame_units
    from .metrics import paired_bootstrap_pvalue
    from .common import resolve
    from .supplementary import scene_paired, scene_units

    register()
    ctx = Ctx(cfg)
    obj = ctx.arm("object")
    pooled = ctx.arm("frame_pooled")
    tt = ctx.test_table(tags=ctx.adverse_tags)
    u_obj = per_frame_units(tt, obj.predict(tt), cfg)
    u_pool = per_frame_units(tt, pooled.predict(tt), cfg)
    scene = scene_paired(scene_units(tt, u_obj), scene_units(tt, u_pool), cfg, n=n_boot)
    # secondary, not registered: frame-level resampling at 10,000 is too slow over ~50k frames; 1,000 used
    frame_diff, frame_p = paired_bootstrap_pvalue(u_obj, u_pool, lambda us: map_of_units(us, cfg), n=1000)
    res = {
        "registration": "R4-1",
        "pool": "ADVERSE_POOLED",
        "n_boot": n_boot,
        "map_object": map_of_units(u_obj, cfg),
        "map_frame_pooled": map_of_units(u_pool, cfg),
        "primary_scene_cluster": scene,
        "secondary_frame_cluster": {"diff": frame_diff, "p_one_sided": frame_p, "n_boot": 1000},
        "refuted": bool(scene["diff"] <= 0 or scene["p_one_sided"] >= 0.05),
        "params": {"object": obj.n_params(), "frame_pooled": pooled.n_params()},
    }
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r4_1_decision_unit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_r4_2(k=5, n_boot=10000):
    """R4-2 (TruckScenes, 5-fold scene CV stratified by weather): information-equal per-frame arm vs per-object arm.

    Registered in notes/registration4.md. Refuted if the pooled out-of-fold scene-cluster difference is <= 0,
    or the per-object arm is ahead in fewer than 4 of the 5 folds.
    """
    import json
    from .r2a_truckscenes import ts_ctx, ts_pools
    from .r2_experiments import fold_ctx, scene_folds
    from .evaluate import map_of_units, per_frame_units, eval_map
    from .supplementary import scene_paired, scene_units
    from .common import resolve

    register()
    ctx = ts_ctx()
    cfg = ctx.cfg
    folds = scene_folds(ctx, k, seed=0, stratify=True)
    oof_obj, oof_pool, per_fold = [], [], []
    for i, fs in enumerate(folds):
        d = fold_ctx(ctx, fs, f"r4_ts_fold{i}")
        arms = {"object": d.arm("object"), "frame_pooled": d.arm("frame_pooled")}
        tt = d.test_table(tags=ctx.adverse_tags)
        maps = {}
        for a, m in arms.items():
            pr = m.predict(tt)
            maps[a] = eval_map(tt, pr, cfg)[0]
        u_obj = scene_units(tt, per_frame_units(tt, arms["object"].predict(tt), cfg))
        u_pool = scene_units(tt, per_frame_units(tt, arms["frame_pooled"].predict(tt), cfg))
        oof_obj.extend(u_obj)
        oof_pool.extend(u_pool)
        per_fold.append({"n_test_scenes": len(fs), "map": maps, "object_minus_pooled": maps["object"] - maps["frame_pooled"]})
        ctx.log.info(f"[r4-2] fold {i}: {maps}")
    pooled = scene_paired(oof_obj, oof_pool, cfg, n=n_boot)
    wins = sum(f["object_minus_pooled"] > 0 for f in per_fold)
    res = {
        "registration": "R4-2",
        "pool": "ADVERSE_POOLED, out-of-fold",
        "n_boot": n_boot,
        "oof_map_object": map_of_units(oof_obj, cfg),
        "oof_map_frame_pooled": map_of_units(oof_pool, cfg),
        "primary_scene_cluster": pooled,
        "folds_object_ahead": int(wins),
        "per_fold": per_fold,
        "refuted": bool(pooled["diff"] <= 0 or wins < 4),
    }
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r4_2_truckscenes_decision_unit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_r4_3():
    """R4-3: the label-free support check vs three standard out-of-distribution scores.

    Registered in notes/registration4.md. Calibration = the same "mild" calibration-split frames the support
    check itself uses (clear0 + severity-1 + night, Section r2e). Positive class = test-split severe frames
    (tags ending "3"); negative class = test-split clear0 frames. Metric: AUROC for flagging severe frames.
    Refuted if the support check's AUROC is more than 0.02 below the best baseline.
    """
    import json
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    from .arbitrate import _frame_feat_matrix
    from .common import load_config, resolve
    from .metrics import safe_auroc
    from .r2_experiments import frame_degradation
    from .run_experiments import Ctx

    cfg = load_config()
    ctx = Ctx(cfg)
    mild = ["clear0"] + [t for t in ctx.tags if t.endswith("1")] + [t for t in ctx.tags if t.startswith("night")]
    cal_m = ctx.sub((ctx.split == "cal") & np.isin(ctx.tag, mild))
    severe_tags = [t for t in ctx.tags if t.endswith("3")]
    test_pos = ctx.sub((ctx.split == "test") & np.isin(ctx.tag, severe_tags))
    test_neg = ctx.sub((ctx.split == "test") & (ctx.tag == "clear0"))

    cal_ff = _frame_feat_matrix(cal_m)
    pos_ff = _frame_feat_matrix(test_pos)
    neg_ff = _frame_feat_matrix(test_neg)
    X_test = np.concatenate([pos_ff, neg_ff])
    y_test = np.concatenate([np.ones(len(pos_ff)), np.zeros(len(neg_ff))])
    score_existing = np.concatenate([frame_degradation(test_pos), frame_degradation(test_neg)])

    mu, sd = cal_ff.mean(0), cal_ff.std(0) + 1e-6
    Z_cal, Z_test = (cal_ff - mu) / sd, (X_test - mu) / sd

    cov = np.cov(Z_cal, rowvar=False) + 1e-3 * np.eye(Z_cal.shape[1])
    inv = np.linalg.pinv(cov)
    diff = Z_test - Z_cal.mean(0)
    maha = np.einsum("ij,jk,ik->i", diff, inv, diff)

    knn = NearestNeighbors(n_neighbors=10).fit(Z_cal)
    knn_dist, _ = knn.kneighbors(Z_test)
    knn_score = knn_dist.mean(1)

    clf = LogisticRegression(max_iter=2000).fit(
        np.concatenate([Z_cal, Z_test]), np.concatenate([np.zeros(len(Z_cal)), np.ones(len(Z_test))]))
    logit_score = clf.predict_proba(Z_test)[:, 1]

    aurocs = {
        "support_check": safe_auroc(y_test, score_existing),
        "mahalanobis": safe_auroc(y_test, maha),
        "knn10": safe_auroc(y_test, knn_score),
        "logistic_density_ratio": safe_auroc(y_test, logit_score),
    }
    best_baseline = max(v for k, v in aurocs.items() if k != "support_check")
    res = {
        "registration": "R4-3",
        "n_cal": int(len(cal_ff)), "n_pos": int(len(pos_ff)), "n_neg": int(len(neg_ff)),
        "aurocs": aurocs, "best_baseline": best_baseline,
        "refuted": bool(aurocs["support_check"] < best_baseline - 0.02),
    }
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r4_3_support_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------------------------
# R4-4: a stronger camera (LiDAR-projected range instead of the monocular ground/size-prior blend), as a
# sensitivity check on whether the per-object advantage survives a better camera position estimate. Reuses
# the existing frozen-detector cache (2D boxes, class priors, azimuth) -- only the camera's range changes, so
# nothing is re-detected and no file under results/cache or results/candidates is touched.

_R4C = {}


def _r4c_init():
    from .truckscenes_data import build_ts_index
    _R4C["by_id"] = {fr.token: fr for fr in build_ts_index()}


def _project_lidar_range(fr, boxes2d, cam_T, P, min_pts=3):
    import numpy as np
    from .truckscenes_data import read_lidar
    pts = read_lidar(fr)
    xyz = pts[:, :3]
    cam_pts = xyz @ cam_T[:3, :3].T + cam_T[:3, 3]
    infront = cam_pts[:, 2] > 0.5
    xyz, cam_pts = xyz[infront], cam_pts[infront]
    uvw = cam_pts @ P[:3, :3].T
    u, v = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
    rng = np.hypot(xyz[:, 0], xyz[:, 1])
    out = np.full(len(boxes2d), np.nan)
    for i, (u0, v0, u1, v1) in enumerate(boxes2d):
        m = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
        if m.sum() >= min_pts:
            out[i] = np.median(rng[m])
    return out


def _r4c_process(args):
    import numpy as np
    from .candidates import build_sample_candidates, load_sample
    from .truckscenes_data import camera_geometry

    path, cfg = args
    meta, s = load_sample(path)
    fr = _R4C["by_id"].get(meta["frame_id"])
    if fr is None:
        return None
    s = dict(s)
    b2d = s["camera_boxes2d"]
    n_overridden = 0
    if len(b2d):
        from .ts_cache import CAM_SCALE
        cam_T, P = camera_geometry(fr)
        P = P.copy()
        P[:2, :3] *= CAM_SCALE  # boxes2d were detected on the CAM_SCALE-resized image (see ts_cache.process)
        r_lidar = _project_lidar_range(fr, b2d, cam_T, P)
        boxes = s["camera_boxes"].copy()
        ok = np.isfinite(r_lidar)
        n_overridden = int(ok.sum())
        if ok.any():
            L, W, az = boxes[ok, 3], boxes[ok, 4], boxes[ok, 6]
            r_centre = r_lidar[ok] + 0.25 * (L + W)
            boxes[ok, 0] = r_centre * np.cos(az)
            boxes[ok, 1] = r_centre * np.sin(az)
        s["camera_boxes"] = boxes
    c = build_sample_candidates(meta, s, cfg)
    groi = s["gt_in_roi"].astype(bool) if len(s["gt_boxes"]) else np.zeros(0, bool)
    gt = {"xy": s["gt_boxes"][groi, :2], "cls": s["gt_cls"][groi], "npts": s["gt_npts"][groi],
          "dontcare_xy": s["gt_dontcare"][:, :2]}
    frame = {"frame_id": meta["frame_id"], "scene": meta["scene"], "split": meta["split"],
             "condition": meta["condition"], "severity": meta["severity"]}
    return c, gt, frame, len(b2d), n_overridden


def build_stronger_camera_candidates(cfg, workers=8, limit=None):
    """Writes alternate TruckScenes candidate tables under results/r4/candidates_stronger_cam/truckscenes/.
    The original results/cache and results/candidates trees are never opened for writing."""
    import json
    import multiprocessing as mp
    from pathlib import Path

    from .candidates import save_table
    from .common import resolve

    cache_root = resolve(cfg, "cache_root") / "truckscenes"
    out_root = resolve(cfg, "results_root") / "r4" / "candidates_stronger_cam" / "truckscenes"
    out_root.mkdir(parents=True, exist_ok=True)
    totals = {}
    for tag_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        paths = sorted(tag_dir.glob("*.npz"))
        if limit:
            paths = paths[:limit]
        rows, gts, frames = [], [], []
        n_boxes = n_over = 0
        with mp.get_context("spawn").Pool(workers, initializer=_r4c_init) as pool:
            for res in pool.imap_unordered(_r4c_process, [(p, cfg) for p in paths], chunksize=8):
                if res is None:
                    continue
                c, gt, frame, nb, no = res
                fi = len(frames)
                c["frame"] = np.full(len(c["y"]), fi, np.int64)
                rows.append(c)
                gts.append(gt)
                frames.append(frame)
                n_boxes += nb
                n_over += no
        if rows:
            table = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
            save_table({"cand": table, "gts": gts, "frames": frames}, out_root / f"{tag_dir.name}.npz")
            totals[tag_dir.name] = {"n_frames": len(frames), "n_camera_boxes": n_boxes, "n_lidar_overridden": n_over}
            print(f"[r4-4] {tag_dir.name}: {len(frames)} frames, {n_over}/{n_boxes} camera boxes got a LiDAR range")
    (out_root / "_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return out_root


def run_r4_4(k=5, n_boot=10000, workers=8):
    """R4-4 (TruckScenes, 5-fold scene CV): per-object vs per-frame classifier, both retrained on candidates
    built with a LiDAR-projected camera range instead of the monocular ground/size-prior blend (sensitivity
    test: does the decision-unit advantage survive a stronger camera position estimate?).

    Registered in notes/registration4.md. Refuted if the pooled out-of-fold scene-cluster difference is <= 0,
    or the per-object arm is ahead in fewer than 4 of the 5 folds (same rule as R4-2).
    """
    import json

    from .common import load_config, resolve, seed_everything
    from .evaluate import eval_map, map_of_units, per_frame_units
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import TS_OVERRIDES
    from .run_experiments import Ctx
    from .supplementary import scene_paired, scene_units

    base_cfg = load_config(None, TS_OVERRIDES)
    cand_root = resolve(base_cfg, "results_root") / "r4" / "candidates_stronger_cam"
    if not (cand_root / "truckscenes" / "clear0.npz").exists():
        build_stronger_camera_candidates(base_cfg, workers=workers)
    cfg = load_config(None, {**TS_OVERRIDES, "paths.candidates_root": str(cand_root),
                             "paths.results_root": "results/r4/stronger_cam"})
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg, dataset="truckscenes")
    folds = scene_folds(ctx, k, seed=0, stratify=True)
    oof_obj, oof_frame, per_fold = [], [], []
    for i, fs in enumerate(folds):
        d = fold_ctx(ctx, fs, f"r4_4_fold{i}")
        arms = {"object": d.arm("object"), "frame_cls": d.arm("frame_cls")}
        tt = d.test_table(tags=ctx.adverse_tags)
        maps = {a: eval_map(tt, m.predict(tt), cfg)[0] for a, m in arms.items()}
        u_obj = scene_units(tt, per_frame_units(tt, arms["object"].predict(tt), cfg))
        u_frame = scene_units(tt, per_frame_units(tt, arms["frame_cls"].predict(tt), cfg))
        oof_obj.extend(u_obj)
        oof_frame.extend(u_frame)
        per_fold.append({"n_test_scenes": len(fs), "map": maps, "object_minus_frame_cls": maps["object"] - maps["frame_cls"]})
        ctx.log.info(f"[r4-4] fold {i}: {maps}")
    ctx.log.info(f"[r4-4] all folds done, {len(oof_obj)} pooled scene units, starting {n_boot}-resample bootstrap")
    import time as _time
    _t0 = _time.time()
    pooled = scene_paired(oof_obj, oof_frame, cfg, n=n_boot)
    ctx.log.info(f"[r4-4] bootstrap done in {_time.time() - _t0:.0f}s")
    wins = sum(f["object_minus_frame_cls"] > 0 for f in per_fold)
    res = {
        "registration": "R4-4",
        "pool": "ADVERSE_POOLED, out-of-fold, stronger (LiDAR-projected) camera",
        "n_boot": n_boot,
        "oof_map_object": map_of_units(oof_obj, cfg),
        "oof_map_frame_cls": map_of_units(oof_frame, cfg),
        "primary_scene_cluster": pooled,
        "folds_object_ahead": int(wins),
        "per_fold": per_fold,
        "refuted": bool(pooled["diff"] <= 0 or wins < 4),
    }
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r4_4_stronger_camera.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_e3_per_scene():
    """E3 (exploratory, Boreas): per-scene paired difference (object minus per-frame classifier), adverse
    pooled, for every one of the 13 test scenes. Uses the already-fitted H2 models; nothing is retrained."""
    import json

    from .common import load_config, resolve
    from .evaluate import map_of_units, per_frame_units
    from .run_experiments import Ctx
    from .supplementary import scene_units

    cfg = load_config()
    ctx = Ctx(cfg)
    obj, frame_cls = ctx.arm("object"), ctx.arm("frame_cls")
    tt = ctx.test_table(tags=ctx.adverse_tags)
    u_obj = scene_units(tt, per_frame_units(tt, obj.predict(tt), cfg))
    u_frame = scene_units(tt, per_frame_units(tt, frame_cls.predict(tt), cfg))
    scenes = sorted(set(f["scene"] for f in tt["frames"]))
    rows = []
    for s, a, b in zip(scenes, u_obj, u_frame):
        map_a, map_b = map_of_units([a], cfg), map_of_units([b], cfg)
        rows.append({"scene": int(s), "map_object": map_a, "map_frame_cls": map_b, "diff": map_a - map_b})
    res = {"dataset": "boreas", "pool": "ADVERSE_POOLED", "n_scenes": len(rows),
           "n_object_ahead": sum(r["diff"] > 0 for r in rows), "rows": rows}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e3_per_scene_boreas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys

    from .common import load_config

    which = sys.argv[1] if len(sys.argv) > 1 else ""
    runners = {"r4_1": lambda: run_r4_1(load_config()), "r4_2": run_r4_2, "r4_3": run_r4_3, "r4_4": run_r4_4,
               "e3": run_e3_per_scene}
    if which not in runners:
        raise SystemExit("usage: python -m src.r4_experiments {" + "|".join(runners) + "}")
    runners[which]()
