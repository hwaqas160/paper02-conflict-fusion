"""Registration 2 experiments (notes/registration2.md) on the cached candidate tables.

    python -m src.r2_experiments --only B G          # external baselines + metrics breadth
Outputs results/r2/*.json + r2.md. Fitted models are cached in results/models_r2/ WITH a data fingerprint:
a cached model is reused only if it was fitted on exactly the same candidate tables (guards against the
stale-pickle failure mode documented in notes/s4_fov_bug.md).
"""
import argparse
import hashlib
import pickle
import time

import numpy as np

from .arbitrate import merge_tables
from .baselines_ext import BASELINES
from .candidates import load_table
from .common import load_config, resolve, save_json, seed_everything
from .evaluate import choose_tau_f1, eval_map, frame_detections, gt_lists, map_of_units, per_frame_units
from .run_experiments import Ctx, NoRadar, _md_table, fmt
from .supplementary import merge_units, pools, scene_paired, scene_units

LABEL = {"object": "Per-object (ours)", "fixed_clear": "Fixed (status quo)", "frame_cls": "Per-frame classifier",
         "proben": "ProbEn (Chen et al., 2022)", "dempster_shafer": "Dempster--Shafer evidential",
         "clocs": "CLOCs (Pang et al., 2020)", "set_transformer": "Set-transformer fusion (black box)",
         "moe_frame": "Per-frame mixture of experts", "oracle": "Oracle selector (upper bound)"}


def fingerprint(ctx):
    """Data AND method: a cached model is reused only if both the candidate tables and the baseline source code
    (src/baselines_ext.py) are unchanged -- a code fix must never be silently skipped by a stale pickle."""
    from pathlib import Path
    c = ctx.T["cand"]
    h = hashlib.sha256()
    h.update(str((len(c["y"]), len(ctx.T["frames"]), float(c["score"].sum()), int(c["y"].sum()))).encode())
    h.update((Path(__file__).parent / "baselines_ext.py").read_bytes())
    return h.hexdigest()[:16]


def fit_baseline(ctx, name, out_dir):
    fp = fingerprint(ctx)
    path = out_dir / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if getattr(obj, "_fp", None) == fp:
            ctx.log.info(f"[r2 {name}] loaded (fingerprint ok)")
            return obj
        ctx.log.info(f"[r2 {name}] stale cache (fingerprint mismatch) -> refit")
    fit_mask = ctx.is_fit & ctx.train_cond
    val_mask = ctx.is_val & ctx.train_cond
    t0 = time.time()
    obj = BASELINES[name]().fit(ctx.sub(fit_mask), ctx.sub(val_mask), ctx.cfg)
    obj.fit_seconds = time.time() - t0
    train_clear = ctx.sub((ctx.split == "train") & (ctx.tag == "clear0"))
    obj.tau, obj.f1 = choose_tau_f1(train_clear, obj.predict(train_clear), ctx.cfg)
    obj._fp = fp
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    ctx.log.info(f"[r2 {name}] fitted in {obj.fit_seconds:.0f}s, params={obj.n_params()}")
    return obj


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def load_cadc(cfg):
    root = resolve(cfg, "candidates_root") / "cadc"
    tables = sorted(root.glob("*.npz")) if root.exists() else []
    if not tables:
        return None
    T = merge_tables([load_table(p) for p in tables])
    for f in T["frames"]:
        f["condition"], f["severity"] = "snow", 2
    return T


class _DropRadar:
    def __init__(self, m):
        self.m, self.tau = m, m.tau

    def predict(self, tbl, drop=None):
        d = np.zeros((len(tbl["frames"]), 3), bool) if drop is None else drop.copy()
        d[:, 2] = True
        return self.m.predict(tbl, d)


# ------------------------------------------------------------------------------------------------ R2-B

def r2b(ctx, out):
    cfg = ctx.cfg
    mdir = ctx.out / "models_r2"
    arms = {"object": ctx.arm("object"), "fixed_clear": ctx.status_quo(), "frame_cls": ctx.arm("frame_cls")}
    for n in BASELINES:
        arms[n] = fit_baseline(ctx, n, mdir)
    res = {"map": {}, "params": {a: int(m.n_params()) for a, m in arms.items()}, "vs_ours": {}}
    units = {}
    for pname, tags in pools(ctx).items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        for a, m in arms.items():
            pr = m.predict(tt)
            res["map"].setdefault(a, {})[pname] = eval_map(tt, pr, cfg)[0]
            if pname == "ADVERSE_POOLED":
                units[a] = scene_units(tt, per_frame_units(tt, pr, cfg))
    comp = [a for a in arms if a not in ("object", "oracle")]
    raw_p = {}
    for a in comp:
        r = scene_paired(units["object"], units[a], cfg)
        res["vs_ours"][a] = r
        raw_p[a] = r["p_one_sided"]
    for a, p in holm(raw_p).items():
        res["vs_ours"][a]["p_holm"] = p
    sq, orc, ours = (res["map"][k]["ADVERSE_POOLED"] for k in ("fixed_clear", "oracle", "object"))
    res["headroom_captured"] = (ours - sq) / max(orc - sq, 1e-9)
    # CADC real snow (radar absent), zero-shot from Boreas
    T = load_cadc(cfg)
    if T is not None:
        res["cadc"] = {}
        drives = np.array([str(f["frame_id"]).rsplit("_", 1)[0] for f in T["frames"]])
        cu = {}
        for a, m in arms.items():
            pr = _DropRadar(m).predict(T)
            res["cadc"][a] = eval_map(T, pr, cfg)[0]
            u = per_frame_units(T, pr, cfg)
            cu[a] = [merge_units([u[i] for i in np.nonzero(drives == d)[0]]) for d in np.unique(drives)]
        res["cadc_vs_ours"] = {a: scene_paired(cu["object"], cu[a], cfg) for a in comp}
    save_json(res, out / "r2b_baselines.json")
    cols = [p for p, t in pools(ctx).items() if t]
    order = ["oracle", "object", "set_transformer", "clocs", "proben", "dempster_shafer", "moe_frame", "frame_cls",
             "fixed_clear"]
    rows = []
    for a in order:
        v = res["vs_ours"].get(a, {})
        rows.append([LABEL[a], str(res["params"][a])] + [fmt(res["map"][a].get(p)) for p in cols] +
                    ([fmt(res["cadc"][a])] if "cadc" in res else []) +
                    ([f"{fmt(v.get('diff'), 4)} (p_Holm {fmt(v.get('p_holm'), 3)})"] if v else ["--"]))
    md = "## R2-B -- External late-fusion baselines (Car mAP, test scenes)\n\n"
    md += _md_table(["method", "params"] + cols + (["CADC (zero-shot)"] if "cadc" in res else []) +
                    ["ours − method, adverse (scene bootstrap)"], rows)
    md += f"\n\nOracle headroom captured by the per-object arm (adverse pooled): {100 * res['headroom_captured']:.0f} %\n"
    return res, md


# ------------------------------------------------------------------------------------------------ R2-G

def range_band_ap(tbl, pred, cfg, lo, hi):
    """Car mAP restricted to GT and detections whose centre range is in [lo, hi)."""
    import copy as _c
    t2 = _c.copy(tbl)
    gts = []
    for g in tbl["gts"]:
        r = np.hypot(g["xy"][:, 0], g["xy"][:, 1]) if len(g["xy"]) else np.zeros(0)
        keep = (r >= lo) & (r < hi)
        gts.append({**g, "xy": g["xy"][keep], "cls": g["cls"][keep], "npts": g["npts"][keep]})
    t2["gts"] = gts
    r = np.hypot(pred["xy"][:, 0], pred["xy"][:, 1])
    p2 = dict(pred, score=np.where((r >= lo) & (r < hi), pred["score"], 0.0))
    return eval_map(t2, p2, cfg)[0]


def near_miss_rate(tbl, pred, tau, cfg, max_r=30.0):
    dets = frame_detections(tbl, pred, 0, cfg["association"]["nms_center_m"])
    gts, _ = gt_lists(tbl, 0)
    miss = tot = 0
    for (xy, s, _), g in zip(dets, gts):
        if not len(g):
            continue
        g = g[np.hypot(g[:, 0], g[:, 1]) < max_r]
        tot += len(g)
        keep = xy[s >= tau]
        if len(g) == 0:
            continue
        if len(keep) == 0:
            miss += len(g)
            continue
        d = np.linalg.norm(g[:, None] - keep[None], axis=-1)
        miss += int((d.min(1) > 2.0).sum())
    return miss / max(tot, 1)


def r2g(ctx, out):
    cfg = ctx.cfg
    arms = {"object": ctx.arm("object"), "fixed_clear": ctx.status_quo(), "frame_cls": ctx.arm("frame_cls")}
    res = {"range_band": {}, "near_miss": {}, "latency_ms": {}}
    tt = ctx.test_table(tags=ctx.adverse_tags)
    for a, m in arms.items():
        pr = m.predict(tt)
        res["range_band"][a] = {f"{lo}-{hi}": range_band_ap(tt, pr, cfg, lo, hi) for lo, hi in ((0, 20), (20, 40), (40, 70))}
        res["near_miss"][a] = near_miss_rate(tt, pr, m.tau, cfg)
    # latency: router forward on single frames (CPU), median over 200 frames
    import torch
    torch.set_num_threads(1)
    one = ctx.test_table(tags=["snow3"])
    frames = np.unique(one["cand"]["frame"])[:200]
    from .arbitrate import subset_table
    for a in ("object", "frame_cls"):
        m = arms[a]
        ts = []
        for f in frames:
            fm = np.zeros(len(one["frames"]), bool)
            fm[f] = True
            sub = subset_table(one, fm)
            t0 = time.perf_counter()
            m.predict(sub)
            ts.append(1000 * (time.perf_counter() - t0))
        res["latency_ms"][a] = {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)),
                                "candidates_per_frame": float(len(one["cand"]["y"]) / len(one["frames"]))}
    save_json(res, out / "r2g_metrics.json")
    md = "## R2-G -- Range bands, near-range misses, latency (adverse pooled, test)\n\n"
    md += _md_table(["arm", "AP 0-20 m", "AP 20-40 m", "AP 40-70 m", "miss rate < 30 m"],
                    [[LABEL[a], *[fmt(v) for v in res["range_band"][a].values()], fmt(res["near_miss"][a])]
                     for a in arms])
    md += "\n\nRouter latency (CPU, 1 thread, one frame incl. feature assembly): " + "; ".join(
        f"{LABEL[a]} median {v['median']:.1f} ms, p95 {v['p95']:.1f} ms" for a, v in res["latency_ms"].items())
    return res, md


# ------------------------------------------------------------------------------------------------ R2-C

R2C_VARIANTS = {  # name -> (fit_arbiter kwargs, arbitration cfg overrides)
    "no_absence_term": ({"absence_scale": 0.0}, {}),
    "no_learned_precision": ({"learn_precision": False}, {}),
    "no_harm_aux_loss": ({}, {"loss_harm_weight": 0.0}),
    "no_localisation_loss": ({}, {"loss_loc_weight": 0.0}),
    "no_sensor_dropout": ({"modality_dropout": 0.0}, {}),
    "no_frame_context": ({"context": False}, {}),
}


def fit_ablation(ctx, name, out_dir):
    import copy as _c
    from pathlib import Path
    from .arbitrate import fit_arbiter
    kw, over = R2C_VARIANTS[name]
    h = hashlib.sha256((fingerprint(ctx) + name).encode())
    h.update((Path(__file__).parent / "arbitrate.py").read_bytes())
    fp = h.hexdigest()[:16]
    path = out_dir / f"r2c_{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if getattr(obj, "_fp", None) == fp:
            return obj
    cfg = _c.deepcopy(ctx.cfg)
    cfg["arbitration"].update(over)
    h_lab, _, _ = ctx.harm()
    fit_m, val_m = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cand_fit, cand_val = fit_m[ctx.T["cand"]["frame"]], val_m[ctx.T["cand"]["frame"]]
    obj = fit_arbiter("object", ctx.sub(fit_m), ctx.sub(val_m), cfg, harm_labels=h_lab[cand_fit],
                      val_harm=h_lab[cand_val], log=ctx.log, seed=cfg["seed"], **kw)
    train_clear = ctx.sub((ctx.split == "train") & (ctx.tag == "clear0"))
    obj.tau, obj.f1 = choose_tau_f1(train_clear, obj.predict(train_clear), ctx.cfg)
    obj._fp = fp
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def r2c(ctx, out):
    cfg = ctx.cfg
    full = ctx.arm("object")
    arms = {"full": full, **{n: fit_ablation(ctx, n, ctx.out / "models_r2") for n in R2C_VARIANTS}}
    res = {"map": {}, "vs_full": {}}
    units = {}
    for pname, tags in pools(ctx).items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        for a, m in arms.items():
            pr = m.predict(tt)
            res["map"].setdefault(a, {})[pname] = eval_map(tt, pr, cfg)[0]
            if pname == "ADVERSE_POOLED":
                units[a] = scene_units(tt, per_frame_units(tt, pr, cfg))
    raw_p = {}
    for a in R2C_VARIANTS:
        r = scene_paired(units["full"], units[a], cfg)
        res["vs_full"][a] = r
        raw_p[a] = r["p_one_sided"]
    for a, p in holm(raw_p).items():
        res["vs_full"][a]["p_holm"] = p
        res["vs_full"][a]["contributes"] = bool(res["vs_full"][a]["diff"] > 0 and p < 0.05)
    save_json(res, out / "r2c_ablation.json")
    cols = [p for p, t in pools(ctx).items() if t]
    md = "## R2-C -- Mechanism ablations, each retrained (per-object arm, Car mAP, test)\n\n"
    md += _md_table(["variant"] + cols + ["full − variant, adverse (scene bootstrap)", "contributes?"],
                    [["full (ours)"] + [fmt(res["map"]["full"][p]) for p in cols] + ["--", "--"]] +
                    [[a] + [fmt(res["map"][a][p]) for p in cols] +
                     [f"{fmt(res['vs_full'][a]['diff'], 4)} (p_Holm {fmt(res['vs_full'][a]['p_holm'], 3)})",
                      "yes" if res["vs_full"][a]["contributes"] else "no"] for a in R2C_VARIANTS])
    return res, md


# ------------------------------------------------------------------------------------------------ R2-F / R2-A: CV

def scene_folds(ctx, k=5, seed=0, stratify=True):
    """Scene-level folds; stratified by each scene's (majority) condition so every fold sees every weather."""
    scenes = np.unique(ctx.scene)
    cond_of = {s: ctx.cond[ctx.scene == s][0] for s in scenes}
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    groups = {}
    for s in scenes:
        groups.setdefault(cond_of[s] if stratify else "all", []).append(s)
    offset = 0
    for g in sorted(groups):
        members = list(rng.permutation(groups[g]))
        for i, s in enumerate(members):
            folds[(i + offset) % k].append(s)
        offset += len(members)
    return folds


def fold_ctx(ctx, test_scenes, tag):
    import copy as _c
    from pathlib import Path
    d = _c.copy(ctx)
    test = np.isin(ctx.scene, test_scenes)
    rest = sorted(set(ctx.scene[~test]))
    rng = np.random.default_rng(ctx.cfg["seed"] + 1)
    val_scenes = set(rng.choice(rest, max(1, len(rest) // 5), replace=False))
    d.split = np.where(test, "test", "train")
    d.is_val = np.isin(ctx.scene, list(val_scenes)) & ~test
    d.is_fit = ~test & ~d.is_val
    d.models = {}
    for a in ("_harm", "_wclf"):
        if hasattr(d, a):
            delattr(d, a)
    h = hashlib.sha256(fingerprint(ctx).encode())
    h.update((Path(__file__).parent / "arbitrate.py").read_bytes())
    d.model_dir = ctx.out / "models_cv" / f"{h.hexdigest()[:12]}_{tag}"
    d.model_dir.mkdir(parents=True, exist_ok=True)
    return d


def run_cv(ctx, arms=("fixed_clear", "frame_cls", "object"), k=5):
    cfg = ctx.cfg
    folds = scene_folds(ctx, k)
    per_fold, oof = [], {a: [] for a in arms}
    for i, fs in enumerate(folds):
        d = fold_ctx(ctx, fs, f"fold{i}")
        models = {a: (d.status_quo() if a == "fixed_clear" else d.arm(a)) for a in arms}
        row = {"test_scenes": [str(s) for s in fs]}
        for pname, tags in pools(d).items():
            if not tags:
                continue
            tt = d.test_table(tags=tags)
            if not tt["frames"]:
                continue
            row[pname] = {a: eval_map(tt, m.predict(tt), cfg)[0] for a, m in models.items()}
        tt = d.test_table(tags=d.adverse_tags)
        for a, m in models.items():
            oof[a].extend(scene_units(tt, per_frame_units(tt, m.predict(tt), cfg)))
        per_fold.append(row)
        ctx.log.info(f"[cv] fold {i}: {row.get('ADVERSE_POOLED')}")
    res = {"k": k, "per_fold": per_fold,
           "oof_adverse_map": {a: map_of_units(u, cfg) for a, u in oof.items()},
           "oof_object_vs_frame_cls": scene_paired(oof["object"], oof["frame_cls"], cfg),
           "oof_object_vs_fixed": scene_paired(oof["object"], oof["fixed_clear"], cfg)}
    deltas = [f["ADVERSE_POOLED"]["object"] - f["ADVERSE_POOLED"]["frame_cls"] for f in per_fold if "ADVERSE_POOLED" in f]
    res["folds_object_beats_frame_cls"] = int(sum(x > 0 for x in deltas))
    res["fold_deltas"] = deltas
    return res


def r2f(ctx, out):
    res = run_cv(ctx)
    res["refuted"] = bool(res["folds_object_beats_frame_cls"] < 4)
    save_json(res, out / "r2f_cv.json")
    rows = [[str(i), fmt(f["ADVERSE_POOLED"]["fixed_clear"]), fmt(f["ADVERSE_POOLED"]["frame_cls"]),
             fmt(f["ADVERSE_POOLED"]["object"]), fmt(f["ADVERSE_POOLED"]["object"] - f["ADVERSE_POOLED"]["frame_cls"], 4)]
            for i, f in enumerate(res["per_fold"]) if "ADVERSE_POOLED" in f]
    o = res["oof_object_vs_frame_cls"]
    md = "## R2-F -- 5-fold scene cross-validation (Boreas, all 37 scenes, adverse pooled Car mAP)\n\n"
    md += _md_table(["fold", "status quo", "per-frame classifier", "per-object (ours)", "Δ ours − per-frame"], rows)
    md += (f"\n\nOut-of-fold pooled: ours {fmt(res['oof_adverse_map']['object'])} vs per-frame "
           f"{fmt(res['oof_adverse_map']['frame_cls'])}, Δ {fmt(o['diff'], 4)} (95 % CI [{fmt(o['ci95'][0], 4)}, "
           f"{fmt(o['ci95'][1], 4)}], scene p {fmt(o['p_one_sided'], 3)}); ours ahead in "
           f"{res['folds_object_beats_frame_cls']}/{res['k']} folds. R2-F1 refuted: {res['refuted']}\n")
    return res, md


# ------------------------------------------------------------------------------------------------ R2-E

DEG_PROBES = ["l_log_pts", "l_density", "c_contrast", "c_dark", "c_log_lap"]
DEG_FRAME = ["f_l_log_pts", "f_l_log_far_pts", "f_c_contrast", "f_c_dark"]


def _deg_loc_data(arb, tbl, cfg):
    """exp4's localisation data + label-free degradation features for the detected true positives."""
    from .degradation import FRAME_FEATURES, PROBE_NAMES
    from .run_experiments import _loc_data
    d, pr = _loc_data(arb, tbl, cfg)
    c = tbl["cand"]
    det = (c["cls"] == 0) & (c["y"] > 0) & (pr["score"] >= arb.tau) & ~c["ignore"]
    pi = [PROBE_NAMES.index(n) for n in DEG_PROBES]
    fi = [FRAME_FEATURES.index(n) for n in DEG_FRAME]
    psi = np.concatenate([np.nan_to_num(c["probes"][det][:, pi]), c["frame_feats"][det][:, fi]], 1)
    d["psi"] = psi
    return d, pr


def frame_degradation(tbl):
    """Label-free scalar degradation score per frame: fewer far LiDAR returns + lower image contrast."""
    from .arbitrate import _frame_feat_matrix
    from .degradation import FRAME_FEATURES
    ff = _frame_feat_matrix(tbl)
    return -ff[:, FRAME_FEATURES.index("f_l_log_far_pts")] - 3.0 * ff[:, FRAME_FEATURES.index("f_c_contrast")]


def r2e(ctx, out, alpha=0.1):
    from .conformal import LocalisationConformal, crc_threshold, frame_fnr
    cfg = ctx.cfg
    arb = ctx.arm("object")
    tr = ctx.sub((ctx.split == "train") & ctx.train_cond)
    cal_clear = ctx.sub((ctx.split == "cal") & (ctx.tag == "clear0"))
    dtr, _ = _deg_loc_data(arb, tr, cfg)
    dcal, _ = _deg_loc_data(arb, cal_clear, cfg)
    mu, sd = dtr["psi"].mean(0), dtr["psi"].std(0) + 1e-6
    z = lambda d: np.concatenate([d["phi"], (d["psi"] - mu) / sd], 1)
    methods = {"vanilla": LocalisationConformal(False), "conflict": LocalisationConformal(True),
               "degradation_normalised": LocalisationConformal(True)}
    methods["vanilla"].fit_scale(dtr["err"], dtr["rng"], dtr["phi"])
    methods["conflict"].fit_scale(dtr["err"], dtr["rng"], dtr["phi"])
    methods["degradation_normalised"].fit_scale(dtr["err"], dtr["rng"], z(dtr))
    methods["vanilla"].calibrate(dcal["err"], dcal["rng"], dcal["phi"], alpha)
    methods["conflict"].calibrate(dcal["err"], dcal["rng"], dcal["phi"], alpha)
    methods["degradation_normalised"].calibrate(dcal["err"], dcal["rng"], z(dcal), alpha)
    # E2: degradation-Mondrian CRC; calibration = cal scenes, clear + every severity-1 condition + night
    mild = ["clear0"] + [t for t in ctx.tags if t.endswith("1")] + [t for t in ctx.tags if t.startswith("night")]
    cal_m = ctx.sub((ctx.split == "cal") & np.isin(ctx.tag, mild))
    pr_cm = arb.predict(cal_m)
    dets_cm = frame_detections(cal_m, pr_cm, 0, cfg["association"]["nms_center_m"])
    gts_cm, _ = gt_lists(cal_m, 0)
    dg_cm = frame_degradation(cal_m)
    edges = np.quantile(dg_cm, [1 / 3, 2 / 3])
    grid = np.unique(np.concatenate([[0.0], np.quantile(pr_cm["score"][pr_cm["score"] > 0], np.linspace(0, 1, 200))]))
    lam_bin = []
    for b in range(3):
        idx = np.nonzero(np.digitize(dg_cm, edges) == b)[0]
        dsub = [dets_cm[i] for i in idx]
        gsub = [gts_cm[i] for i in idx]
        lam_bin.append(crc_threshold(lambda l: frame_fnr(dsub, gsub, 2.0, l), sum(len(g) > 0 for g in gsub), alpha, list(grid)))
    dets_cc = frame_detections(cal_clear, arb.predict(cal_clear), 0, cfg["association"]["nms_center_m"])
    gts_cc, _ = gt_lists(cal_clear, 0)
    lam_clear = crc_threshold(lambda l: frame_fnr(dets_cc, gts_cc, 2.0, l), sum(len(g) > 0 for g in gts_cc), alpha, list(grid))
    # Exploratory (added after R2-E1 was scored): a label-free support check. Frames whose degradation score lies
    # above the 99th percentile of the calibration frames are outside the certificate's support.
    support_hi = float(np.quantile(dg_cm, 0.99))
    res = {"alpha": alpha, "loc": {}, "crc": {}, "lam_bins": lam_bin, "lam_clear": lam_clear, "deg_edges": edges.tolist(),
           "support_hi": support_hi}
    groups = {t: [t] for t in ctx.tags}
    groups.update({"SNOW_POOLED": [t for t in ctx.tags if t.startswith("snow")], "SEVERE": [t for t in ctx.tags if t.endswith("3")]})
    for g, tags in groups.items():
        tt = ctx.test_table(tags=tags)
        d, pr = _deg_loc_data(arb, tt, cfg)
        res["loc"][g] = {"vanilla": methods["vanilla"].evaluate(d["err"], d["rng"], d["phi"]),
                         "conflict": methods["conflict"].evaluate(d["err"], d["rng"], d["phi"]),
                         "degradation_normalised": methods["degradation_normalised"].evaluate(d["err"], d["rng"], z(d))}
        dets = frame_detections(tt, pr, 0, cfg["association"]["nms_center_m"])
        gts, _ = gt_lists(tt, 0)
        b = np.digitize(frame_degradation(tt), edges)
        f_clear = frame_fnr(dets, gts, 2.0, lam_clear)
        f_mond = np.array([frame_fnr([dets[i]], [gts[i]], 2.0, lam_bin[b[i]])[0] if len(gts[i]) else np.nan
                           for i in range(len(dets))])
        has = np.array([len(x) > 0 for x in gts])
        inside = (frame_degradation(tt) <= support_hi)[has]
        f_clear = np.asarray(f_clear)
        res["crc"][g] = {"fnr_clear_calibrated": float(np.mean(f_clear)) if len(f_clear) else float("nan"),
                         "fnr_degradation_mondrian": float(np.nanmean(f_mond)),
                         "fnr_floor_accept_all": float(np.mean(frame_fnr(dets, gts, 2.0, 0.0))),
                         "flagged_out_of_support": float(1 - inside.mean()) if len(inside) else float("nan"),
                         "fnr_in_support": float(f_clear[inside].mean()) if inside.any() else float("nan")}
    fog = res["loc"]["fog3"]["degradation_normalised"]["coverage"]
    res["E1_refuted"] = bool(fog < 0.85 or res["crc"]["fog3"]["fnr_degradation_mondrian"] > 0.15)
    save_json(res, out / "r2e_conformal.json")
    rows = [[g, *[f"{fmt(v[k]['coverage'])} / {fmt(v[k]['mean_radius'], 2)} m" for k in ("vanilla", "conflict", "degradation_normalised")],
             fmt(res["crc"][g]["fnr_clear_calibrated"]), fmt(res["crc"][g]["fnr_degradation_mondrian"])]
            for g, v in res["loc"].items()]
    md = (f"## R2-E -- Repairing risk control under shift (alpha = {alpha}, nominal coverage {1 - alpha:.2f}, "
          f"FNR target {alpha})\n\nLocalisation: coverage / mean radius; all scales fitted on training scenes, "
          f"calibrated on clear cal-scene frames only. Detection: frame FNR with lambda calibrated on clear only vs "
          f"per-degradation-bin (label-free) on clear + mild cal-scene frames.\n\n")
    md += _md_table(["condition", "vanilla", "conflict", "degradation-normalised", "FNR clear-calibrated",
                     "FNR degradation-Mondrian"], rows)
    md += f"\n\nR2-E1 refuted (fog3 coverage < 0.85 or FNR > 0.15): {res['E1_refuted']}\n"
    md += ("\n### Exploratory: detection floor and label-free support check\n\nFNR floor = frame FNR when EVERY "
           "candidate is accepted (lambda = 0): no threshold-based risk control can go below it. Flagged = frames whose "
           "label-free degradation score exceeds the 99th percentile of the calibration frames.\n\n")
    md += _md_table(["condition", "FNR floor (accept all)", "flagged out of support", "FNR on unflagged frames"],
                    [[g, fmt(v["fnr_floor_accept_all"]), fmt(v["flagged_out_of_support"]), fmt(v["fnr_in_support"])]
                     for g, v in res["crc"].items()])
    return res, md


R2 = {"B": r2b, "C": r2c, "E": r2e, "F": r2f, "G": r2g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    for k in (args.only or list(R2)):
        t0 = time.time()
        _, md = R2[k](ctx, out)
        (out / f"R2{k}.md").write_text(md + "\n", encoding="utf-8")
        ctx.log.info(f"r2 {k} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
