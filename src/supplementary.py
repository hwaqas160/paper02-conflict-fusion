"""Supplementary analyses that close the loopholes a reviewer would attack (run after run_experiments).

    S1  scene-cluster bootstrap of the H2 comparison (frames inside a scene are correlated; 13 test scenes)
    S2  where the per-object gain comes from: AP per distance threshold, and an existence / localisation
        decomposition by swapping fused scores and fused positions between arms
    S3  is it the degradation probes, or just capacity / geometry?  CLOCs-style arm (confidences + geometric
        conflict + detector features, NO degradation probes), probes-only arm, capacity-matched per-frame
        (≈ per-object parameter count) and per-object (≈ per-frame count) arms, negative-evidence ablation
    S4  is it just radar?  every arm retrained with radar removed from the data, evaluated on Boreas and on
        CADC real snow (the proper CADC transfer)
    S5  textbook late-fusion rules (max / noisy-OR / sum-of-logits, hand-set sigmas) and single sensors

    python -m src.supplementary              # all
    python -m src.supplementary --only S1 S4
Outputs: results/supplementary/*.json + supplementary.md
"""
import argparse
import copy
import time

import numpy as np

from .arbitrate import CAM_DET, LID_DET, M_CAM, M_LID, M_RAD, MODALITY_OF_PROBE, NaivePredictor, RAD_DET, \
    merge_tables, subset_table
from .candidates import load_table
from .common import MODALITIES, load_config, resolve, save_json, seed_everything
from .degradation import FRAME_FEATURES, PROBE_NAMES
from .evaluate import eval_map, map_of_units, per_frame_units
from .metrics import bootstrap_ci, paired_bootstrap_pvalue
from .run_experiments import ARMS, Ctx, NoRadar, _md_table, fmt
from . import plots

CLOCS_BLOCKS = ["det_feats", "conflict", "confidence"]
PROBE_BLOCKS = ["geometry", "lidar_probes", "camera_probes", "radar_probes", "frame_feats"]


# ------------------------------------------------------------------------------------------------ helpers

def merge_units(units):
    """Collapse a list of frame units into ONE unit (per threshold: concatenated scores / TP flags, summed #GT)."""
    if not units:
        return None
    return [(np.concatenate([u[k][0] for u in units]), np.concatenate([u[k][1] for u in units]),
             sum(u[k][2] for u in units)) for k in range(len(units[0]))]


def scene_units(tt, units):
    """One merged unit per scene, so a scene bootstrap resamples 13 arrays instead of thousands of frames."""
    scenes = np.array([f["scene"] for f in tt["frames"]])
    return [merge_units([units[i] for i in np.nonzero(scenes == s)[0]]) for s in np.unique(scenes)]


def scene_paired(a, b, cfg, n=1000, seed=0):
    """Paired cluster bootstrap over SCENES: one-sided p for H0 mAP(a) <= mAP(b), and a 95 % CI of the difference."""
    rng = np.random.default_rng(seed)
    k = len(a)
    f = lambda g: map_of_units(g, cfg)
    obs = f(a) - f(b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        diffs.append(f([a[i] for i in idx]) - f([b[i] for i in idx]))
    diffs = np.asarray(diffs)
    return {"diff": float(obs), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_one_sided": float(((diffs <= 0).sum() + 1) / (n + 1)), "n_scenes": k,
            "scenes_won": None}


def per_scene_wins(a, b, cfg):
    return int(sum(map_of_units([x], cfg) > map_of_units([y], cfg) for x, y in zip(a, b)))


def remove_modality(tbl, m):
    """The table as if sensor m had never existed: same masking as Arbiter._mask_inputs, applied to the data,
    radar frame-level features zeroed, and candidates with no remaining member dropped."""
    c = {k: v.copy() for k, v in tbl["cand"].items()}
    n = len(c["y"])
    dm = np.zeros((n, 3), bool)
    dm[:, m] = True
    c["present"] &= ~dm
    c["fov"] = c["fov"] * (~dm)
    for prefix, mm in MODALITY_OF_PROBE.items():
        if mm == m:
            cols = [i for i, nme in enumerate(PROBE_NAMES) if nme.startswith(prefix)]
            c["probes"][:, cols] = np.nan
    for cols, mm in ((CAM_DET, M_CAM), (RAD_DET, M_RAD), (LID_DET, M_LID)):
        if mm == m:
            c["det_feats"][:, cols] = 0
    k = c["conflict"]
    if m in (M_LID, M_RAD):
        k[:, 0] = 0
    if m in (M_CAM, M_LID):
        k[:, 1:3] = 0
    if m in (M_CAM, M_RAD):
        k[:, 3:5] = 0
    k[:, 6] = c["present"].sum(1)
    if m == M_RAD:
        rcols = [i for i, nme in enumerate(FRAME_FEATURES) if nme.startswith("f_r_")]
        c["frame_feats"][:, rcols] = 0
    c["score"] = np.where(c["present"], c["score"], 0.0)
    keep = c["present"].any(1)
    frame_new = c["frame"]
    c = {kk: v[keep] for kk, v in c.items()}
    c["frame"] = frame_new[keep]
    return {"cand": c, "gts": tbl["gts"], "frames": tbl["frames"]}


def derived_ctx(ctx, T, tag):
    """A Ctx over a transformed table, with its own model directory (frames and splits unchanged)."""
    d = copy.copy(ctx)
    d.T, d.models = T, {}
    for a in ("_harm", "_wclf"):
        if hasattr(d, a):
            delattr(d, a)
    d.model_dir = ctx.out / f"models_{tag}"
    d.model_dir.mkdir(parents=True, exist_ok=True)
    return d


def fit_with_cfg(ctx, name, key, overrides=None, blocks=None, context=None):
    cfg0 = ctx.cfg
    if overrides:
        c = copy.deepcopy(cfg0)
        for k, v in overrides.items():
            c["arbitration"][k] = v
        ctx.cfg = c
    try:
        return ctx.arm(name, blocks=blocks, context=context, tag=key)
    finally:
        ctx.cfg = cfg0


def pools(ctx):
    cfg = ctx.cfg
    return {"clear0": ["clear0"], "ADVERSE_POOLED": ctx.adverse_tags,
            "HELDOUT": [t for t in ctx.tags if t.rstrip("0123456789") in cfg["weather"]["heldout_for_head"]],
            "fog3": ["fog3"] if "fog3" in ctx.tags else []}


# ------------------------------------------------------------------------------------------------ S1

def s1(ctx, out):
    cfg = ctx.cfg
    arms = {"fixed_clear": ctx.status_quo(), **{a: ctx.arm(a) for a in ARMS}}
    res = {}
    for pname, tags in pools(ctx).items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        su = {a: scene_units(tt, per_frame_units(tt, m.predict(tt), cfg)) for a, m in arms.items()}
        res[pname] = {"map": {a: map_of_units(u, cfg) for a, u in su.items()}}
        for other in ("frame_cls", "frame_oracle", "frame_feats", "confidence", "fixed_clear"):
            r = scene_paired(su["object"], su[other], cfg)
            r["scenes_won"] = per_scene_wins(su["object"], su[other], cfg)
            res[pname][f"object_vs_{other}"] = r
        # the null the paper states: per-frame routing is no better than fixed weights
        r = scene_paired(su["frame_cls"], su["fixed_clear"], cfg)
        r["scenes_won"] = per_scene_wins(su["frame_cls"], su["fixed_clear"], cfg)
        res[pname]["frame_cls_vs_fixed_clear"] = r
    save_json(res, out / "s1_scene_bootstrap.json")
    rows = []
    for p, v in res.items():
        for k, r in v.items():
            if k == "map":
                continue
            rows.append([p, k.replace("_vs_", " − "), fmt(r["diff"], 4), f"[{fmt(r['ci95'][0], 4)}, {fmt(r['ci95'][1], 4)}]",
                         fmt(r["p_one_sided"], 4), f"{r['scenes_won']}/{r['n_scenes']}"])
    md = ("## S1 — Scene-cluster paired bootstrap (1,000 resamples of the test scenes)\n\n"
          "Frames inside a scene are temporally correlated, so the frame bootstrap of Exp 2 is anti-conservative; "
          "this resamples whole scenes.\n\n")
    md += _md_table(["pool", "comparison (Car mAP)", "Δ", "95 % CI", "one-sided p", "scenes won"], rows)
    return res, md


# ------------------------------------------------------------------------------------------------ S2

def s2(ctx, out):
    cfg = ctx.cfg
    sq, obj, fc = ctx.status_quo(), ctx.arm("object"), ctx.arm("frame_cls")
    res = {}
    for pname, tags in pools(ctx).items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        ps, po, pf = sq.predict(tt), obj.predict(tt), fc.predict(tt)
        variants = {"status_quo": ps, "frame_cls": pf, "object": po,
                    "object_scores+status_quo_positions": dict(po, xy=ps["xy"]),
                    "status_quo_scores+object_positions": dict(ps, xy=po["xy"])}
        res[pname] = {}
        thr = cfg["matching"]["ap_thresholds_m"]
        for k, p in variants.items():
            m, aps = eval_map(tt, p, cfg)  # aps is a dict {threshold: AP}, not a list
            res[pname][k] = {"mAP": m, "AP_by_thr": [float(aps[float(t)]) for t in thr]}
        base, full = res[pname]["status_quo"]["mAP"], res[pname]["object"]["mAP"]
        res[pname]["share_from_scores"] = (res[pname]["object_scores+status_quo_positions"]["mAP"] - base) / max(full - base, 1e-9)
        res[pname]["share_from_positions"] = (res[pname]["status_quo_scores+object_positions"]["mAP"] - base) / max(full - base, 1e-9)
    save_json(res, out / "s2_decomposition.json")
    thr = cfg["matching"]["ap_thresholds_m"]
    md = "## S2 — Where does the per-object gain come from?\n\nCar AP per centre-distance threshold, and swaps of " \
         "fused scores (existence) and fused positions (localisation) between the status quo and the per-object arm.\n\n"
    rows = []
    for p, v in res.items():
        for k in ("status_quo", "frame_cls", "object", "object_scores+status_quo_positions", "status_quo_scores+object_positions"):
            rows.append([p, k, fmt(v[k]["mAP"])] + [fmt(a) for a in v[k]["AP_by_thr"]])
        rows.append([p, "→ share of gain from scores / positions", f"{fmt(100 * v['share_from_scores'], 0)} % / {fmt(100 * v['share_from_positions'], 0)} %"] + [""] * len(thr))
    md += _md_table(["pool", "variant", "mAP"] + [f"AP@{t} m" for t in thr], rows)
    return res, md


# ------------------------------------------------------------------------------------------------ S3

def s3(ctx, out):
    cfg = ctx.cfg
    variants = {
        "per-object (ours)": ctx.arm("object"),
        "CLOCs-style (conf + conflict + det feats, no probes)": ctx.arm("object", blocks=CLOCS_BLOCKS, context=False, tag="s3_clocs"),
        "CLOCs-style + frame context": ctx.arm("object", blocks=CLOCS_BLOCKS, context=True, tag="s3_clocs_ctx"),
        "probes only (no conf / conflict / det feats)": ctx.arm("object", blocks=PROBE_BLOCKS, context=True, tag="s3_probes"),
        "per-frame features, hidden 88 (≈ ours' params)": fit_with_cfg(ctx, "frame_feats", "s3_frame_feats_h88", {"hidden": 88}),
        "per-frame classifier, hidden 88": fit_with_cfg(ctx, "frame_cls", "s3_frame_cls_h88", {"hidden": 88}),
        "per-object, hidden 40 (≈ per-frame params)": fit_with_cfg(ctx, "object", "s3_object_h40", {"hidden": 40}),
        "per-frame classifier (Exp 2)": ctx.arm("frame_cls"),
    }
    obj = variants["per-object (ours)"]
    no_abs = copy.copy(obj)
    no_abs.absence_scale = 0.0
    variants["per-object, negative-evidence term removed (test time)"] = no_abs
    res = {}
    for name, arb in variants.items():
        res[name] = {"params": int(arb.n_params())}
        for pname, tags in pools(ctx).items():
            if tags:
                tt = ctx.test_table(tags=tags)
                res[name][pname] = eval_map(tt, arb.predict(tt), cfg)[0]
    save_json(res, out / "s3_controls.json")
    cols = [p for p, t in pools(ctx).items() if t]
    md = "## S3 — Probes vs geometry vs capacity (Car mAP, test)\n\n"
    md += _md_table(["arm", "params"] + cols, [[n, str(v["params"])] + [fmt(v.get(p)) for p in cols] for n, v in res.items()])
    return res, md


# ------------------------------------------------------------------------------------------------ S4

def s4(ctx, out):
    cfg = ctx.cfg
    T_nr = remove_modality(ctx.T, M_RAD)
    nr = derived_ctx(ctx, T_nr, "noradar")
    arms = {"fixed_clear": nr.status_quo(), **{a: nr.arm(a) for a in ARMS}}
    res = {"boreas": {}, "cadc": {}}
    for a, m in arms.items():
        res["boreas"][a] = {}
        for pname, tags in pools(nr).items():
            if tags:
                tt = nr.test_table(tags=tags)
                res["boreas"][a][pname] = eval_map(tt, m.predict(tt), cfg)[0]
    # H2 on the radar-free data (frame bootstrap + scene bootstrap)
    h2 = {}
    for pname in ("ADVERSE_POOLED", "HELDOUT"):
        tags = pools(nr)[pname]
        tt = nr.test_table(tags=tags)
        uo = per_frame_units(tt, arms["object"].predict(tt), cfg)
        uf = per_frame_units(tt, arms["frame_cls"].predict(tt), cfg)
        d, p = paired_bootstrap_pvalue(uo, uf, lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
        sc = scene_paired(scene_units(tt, uo), scene_units(tt, uf), cfg)
        h2[pname] = {"diff": d, "p_frame": p, "p_scene": sc["p_one_sided"], "ci_scene": sc["ci95"]}
    res["boreas_H2"] = h2
    # CADC real snow, heads trained WITHOUT radar
    root = resolve(cfg, "candidates_root") / "cadc"
    tables = sorted(root.glob("*.npz")) if root.exists() else []
    if tables:
        T = merge_tables([load_table(p) for p in tables])
        for f in T["frames"]:
            f["condition"], f["severity"] = "snow", 2
        # CADC's own candidate-builder sets radar fov=1 wherever the geometry is within a radar's field of
        # view (it has no notion that radar is entirely absent from the dataset, only that no detection
        # exists there). remove_modality() gave every arm here a training distribution where radar fov is
        # identically 0 ("structurally absent"), so its standardizer's radar-fov column has ~zero variance;
        # CADC's un-zeroed fov=1 then explodes through that near-zero std and saturates the router (verified:
        # mAP 0.26 -> 0.73 once this is zeroed). Match the training-time convention.
        T["cand"]["fov"][:, M_RAD] = 0.0
        units = {}
        for a, m in arms.items():
            pr = m.predict(T)
            units[a] = per_frame_units(T, pr, cfg)
            res["cadc"][a] = {"mAP": map_of_units(units[a], cfg),
                              "ci95": bootstrap_ci(units[a], lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])}
        drives = np.array([str(f["frame_id"]).rsplit("_", 1)[0] for f in T["frames"]])
        grp = lambda u: [merge_units([u[i] for i in np.nonzero(drives == d)[0]]) for d in np.unique(drives)]
        for other in ("frame_cls", "fixed_clear", "confidence"):
            d, p = paired_bootstrap_pvalue(units["object"], units[other], lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
            sc = scene_paired(grp(units["object"]), grp(units[other]), cfg)
            res["cadc"][f"object_vs_{other}"] = {"diff": d, "p_frame": p, "p_drive_cluster": sc["p_one_sided"],
                                                 "ci_drive_cluster": sc["ci95"], "n_drives": sc["n_scenes"]}
    save_json(res, out / "s4_noradar.json")
    cols = [p for p, t in pools(nr).items() if t]
    md = "## S4 — Radar removed from the data; every arm retrained on camera + LiDAR only\n\n"
    md += _md_table(["arm"] + [f"Boreas {c}" for c in cols] + (["CADC real snow (95 % CI)"] if res["cadc"] else []),
                    [[plots.ARM_LABEL[a]] + [fmt(res["boreas"][a].get(c)) for c in cols] +
                     ([f"{fmt(res['cadc'][a]['mAP'])} ({fmt(res['cadc'][a]['ci95'][0])}–{fmt(res['cadc'][a]['ci95'][1])})"] if res["cadc"] else [])
                     for a in arms])
    md += "\n\nPer-object − per-frame classifier on radar-free Boreas: " + "; ".join(
        f"{p}: Δ {fmt(v['diff'], 4)} (frame p {fmt(v['p_frame'])}, scene p {fmt(v['p_scene'])})" for p, v in h2.items())
    for k in ("object_vs_frame_cls", "object_vs_fixed_clear", "object_vs_confidence"):
        if k in res["cadc"]:
            v = res["cadc"][k]
            md += (f"\n\nCADC {k.replace('_vs_', ' − ')}: Δ {fmt(v['diff'], 4)} (frame p {fmt(v['p_frame'])}; "
                   f"drive-cluster p {fmt(v['p_drive_cluster'])}, CI [{fmt(v['ci_drive_cluster'][0], 4)}, {fmt(v['ci_drive_cluster'][1], 4)}], "
                   f"{v['n_drives']} drives)")
    return res, md


# ------------------------------------------------------------------------------------------------ S5

def s5(ctx, out):
    cfg = ctx.cfg
    res = {}
    rules = {f"untrained {r}": NaivePredictor(r) for r in ("max", "noisy_or", "sum_logits")}
    sq = ctx.status_quo()
    for pname, tags in pools(ctx).items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        c = tt["cand"]
        row = {}
        for name, r in rules.items():
            row[name] = eval_map(tt, r.predict(tt), cfg)[0]
        for m, name in enumerate(MODALITIES):
            p = c["present"][:, m]
            row[f"{name} only (raw detector)"] = eval_map(tt, {"score": np.where(p, c["score"][:, m], 0.0),
                                                               "xy": np.nan_to_num(c["xy"][:, m]), "present_any": p}, cfg)[0]
        row["fitted fixed (status quo)"] = eval_map(tt, sq.predict(tt), cfg)[0]
        row["per-object (ours)"] = eval_map(tt, ctx.arm("object").predict(tt), cfg)[0]
        res[pname] = row
    save_json(res, out / "s5_baselines.json")
    names = list(next(iter(res.values())))
    md = "## S5 — Textbook late-fusion rules and single sensors (Car mAP, test)\n\n"
    md += _md_table(["method"] + list(res), [[n] + [fmt(res[p][n]) for p in res] for n in names])
    return res, md


S = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    out = ctx.out / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    mds = {}
    md_path = out / "supplementary.md"
    for k in (args.only or list(S)):
        t0 = time.time()
        _, md = S[k](ctx, out)
        (out / f"{k}.md").write_text(md + "\n", encoding="utf-8")
        ctx.log.info(f"supplementary {k} done in {time.time() - t0:.0f}s")
    parts = [(out / f"{k}.md").read_text(encoding="utf-8") for k in S if (out / f"{k}.md").exists()]
    md_path.write_text("# Supplementary analyses\n\n" + "\n\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
