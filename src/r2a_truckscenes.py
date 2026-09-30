"""Registration 2, R2-A (+ R2-B on TruckScenes): real adverse weather with 4D radar, MAN TruckScenes v1.2.

Scored exactly as notes/registration2.md, Amendment A:
  primary  = 5-fold scene CV over all trainval scenes, stratified by weather tag (seed 0), out-of-fold pooled,
             one-sided paired scene-cluster bootstrap (1,000 resamples);
  standard = official train_detect -> val split, reported descriptively.
  A1: per-object - per-frame classifier > 0 on adverse scenes.
  A2: pooled decision-flip harm of the status quo on adverse scenes >= 2.0 %.
  A3: per-object with 4D radar - the same per-object head with radar removed > 0 on adverse scenes.
External baselines (R2-B) are fitted inside every fold on exactly the fold's training scenes.

    python -m src.candidates --dataset truckscenes
    python -m src.r2a_truckscenes [--no-baselines] [--k 5]
Outputs results/truckscenes/r2/r2a_truckscenes.json + R2A.md.
"""
import argparse
import time

import numpy as np

from .baselines_ext import BASELINES
from .common import load_config, save_json, seed_everything
from .evaluate import eval_map, map_of_units, per_frame_units
from .harm import harm_summary
from .r2_experiments import LABEL, fit_baseline, fold_ctx, holm, scene_folds
from .run_experiments import Ctx, NoRadar, _md_table, fmt
from .supplementary import scene_paired, scene_units

TS_OVERRIDES = {"weather.conditions": {"clear": [0], "fog": [1], "rain": [1], "snow": [1], "night": [1]},
                "weather.heldout_for_head": [],  # real weather: every condition is seen in training (R2-A)
                "paths.results_root": "results/truckscenes"}
LABEL = {**LABEL, "object_noradar": "Per-object, radar removed"}


def ts_ctx(cfg_path=None):
    cfg = load_config(cfg_path, TS_OVERRIDES)
    seed_everything(cfg["seed"])
    return Ctx(cfg, dataset="truckscenes")


def ts_pools(ctx):
    p = {"clear0": ["clear0"], "ADVERSE_POOLED": ctx.adverse_tags}
    p.update({t: [t] for t in ctx.adverse_tags})
    return p


def fit_arms(d, baselines=True):
    ours = d.arm("object")
    arms = {"fixed_clear": d.status_quo(), "frame_cls": d.arm("frame_cls"), "object": ours,
            "object_noradar": NoRadar(ours)}
    if baselines:
        for n in BASELINES:
            arms[n] = fit_baseline(d, n, d.model_dir / "r2b")
    return arms


def evaluate_split(d, arms, cfg):
    """mAP per pool, scene units per pool, and status-quo decision-flip harm on the test scenes."""
    maps, units = {}, {}
    for pname, tags in ts_pools(d).items():
        tt = d.test_table(tags=tags)
        if not tt["frames"]:
            continue
        for a, m in arms.items():
            pr = m.predict(tt)
            maps.setdefault(pname, {})[a] = eval_map(tt, pr, cfg)[0]
            if pname in ("ADVERSE_POOLED", "clear0"):
                units.setdefault(pname, {})[a] = scene_units(tt, per_frame_units(tt, pr, cfg))
    h, hp, _ = d.harm()
    c = d.T["cand"]
    sel = (d.split[c["frame"]] == "test") & np.isin(d.tag[c["frame"]], d.adverse_tags)
    harm = (h[sel], hp[sel], c["cls"][sel], c["frame"][sel])
    return maps, units, harm


def ablation_units(d, cfg):
    """R2-C on TruckScenes: each mechanism removed and the head retrained on the fold's training scenes."""
    from .r2_experiments import R2C_VARIANTS, fit_ablation
    tt = d.test_table(tags=d.adverse_tags)
    out = {}
    for n in R2C_VARIANTS:
        m = fit_ablation(d, n, d.model_dir / "r2c")
        out[n] = scene_units(tt, per_frame_units(tt, m.predict(tt), cfg))
    return out


def conformal_arrays(d, cfg, alpha=0.1):
    """R2-E on TruckScenes, one fold. Calibration = the fold's held-out validation scenes, CLEAR only (the H3
    protocol); scales fitted on the fold's fit scenes. Returns raw per-object / per-frame arrays per test tag so
    folds can be pooled."""
    from .conformal import LocalisationConformal, crc_threshold, frame_fnr
    from .evaluate import frame_detections, gt_lists
    from .r2_experiments import _deg_loc_data, frame_degradation
    arb = d.arm("object")
    dtr, _ = _deg_loc_data(arb, d.sub(d.is_fit), cfg)
    cal = d.sub(d.is_val & (d.tag == "clear0"))
    dcal, pr_cal = _deg_loc_data(arb, cal, cfg)
    mu, sd = dtr["psi"].mean(0), dtr["psi"].std(0) + 1e-6
    feats = {"vanilla": lambda x: x["phi"], "conflict": lambda x: x["phi"],
             "degradation_normalised": lambda x: np.concatenate([x["phi"], (x["psi"] - mu) / sd], 1)}
    methods = {"vanilla": LocalisationConformal(False), "conflict": LocalisationConformal(True),
               "degradation_normalised": LocalisationConformal(True)}
    for k, m in methods.items():
        m.fit_scale(dtr["err"], dtr["rng"], feats[k](dtr))
        m.calibrate(dcal["err"], dcal["rng"], feats[k](dcal), alpha)
    dets_c = frame_detections(cal, pr_cal, 0, cfg["association"]["nms_center_m"])
    gts_c, _ = gt_lists(cal, 0)
    grid = np.unique(np.concatenate([[0.0], np.quantile(pr_cal["score"][pr_cal["score"] > 0], np.linspace(0, 1, 200))]))
    lam = crc_threshold(lambda l: frame_fnr(dets_c, gts_c, 2.0, l), sum(len(g) > 0 for g in gts_c), alpha, list(grid))
    hi = float(np.quantile(frame_degradation(cal), 0.99))
    out = {}
    for t in d.tags:
        tt = d.test_table(t)
        if not tt["frames"]:
            continue
        x, pr = _deg_loc_data(arb, tt, cfg)
        dets = frame_detections(tt, pr, 0, cfg["association"]["nms_center_m"])
        gts, _ = gt_lists(tt, 0)
        has = np.array([len(g) > 0 for g in gts], bool)
        r = {k: m.radius(x["rng"], feats[k](x)) for k, m in methods.items()}
        out[t] = {**{f"cov_{k}": x["err"] <= v for k, v in r.items()}, **{f"rad_{k}": v for k, v in r.items()},
                  "fnr": np.asarray(frame_fnr(dets, gts, 2.0, lam)), "floor": np.asarray(frame_fnr(dets, gts, 2.0, 0.0)),
                  "inside": (frame_degradation(tt) <= hi)[has]}
    return out


def summarise_conformal(chunks):
    res = {}
    for t in sorted({t for c in chunks for t in c}):
        cat = {k: np.concatenate([c[t][k] for c in chunks if t in c]) for k in next(c[t] for c in chunks if t in c)}
        ins = cat["inside"]
        res[t] = {**{k: float(cat[k].mean()) for k in cat if k.startswith(("cov_", "rad_"))},
                  "fnr_clear_calibrated": float(cat["fnr"].mean()), "fnr_floor_accept_all": float(cat["floor"].mean()),
                  "flagged_out_of_support": float(1 - ins.mean()),
                  "fnr_in_support": float(cat["fnr"][ins].mean()) if ins.any() else float("nan"),
                  "n_objects": int(len(cat["cov_vanilla"])), "n_frames": int(len(cat["fnr"]))}
    return res


def compare(units, cfg, comp, ref="object"):
    out, raw = {}, {}
    for a in comp:
        if a in units:
            out[a] = scene_paired(units[ref], units[a], cfg)
            raw[a] = out[a]["p_one_sided"]
    for a, p in holm(raw).items():
        out[a]["p_holm"] = p
    return out


def run(ctx, k=5, baselines=True, ablations=True, conformal=True):
    cfg = ctx.cfg
    abl_units, conf_chunks = {}, []
    res = {"n_frames": len(ctx.T["frames"]), "n_scenes": int(len(set(ctx.scene))),
           "scenes_per_condition": {t: int(len(set(ctx.scene[ctx.tag == t]))) for t in ctx.tags},
           "frames_per_condition": {t: int((ctx.tag == t).sum()) for t in ctx.tags}}
    # ---- primary: 5-fold stratified scene CV, out-of-fold pooled
    folds = scene_folds(ctx, k, seed=0, stratify=True)
    oof_units, oof_harm, per_fold = {}, [], []
    for i, fs in enumerate(folds):
        t0 = time.time()
        d = fold_ctx(ctx, fs, f"ts_fold{i}")
        arms = fit_arms(d, baselines)
        maps, units, harm = evaluate_split(d, arms, cfg)
        for pname, u in units.items():
            for a, v in u.items():
                oof_units.setdefault(pname, {}).setdefault(a, []).extend(v)
        oof_harm.append(harm)
        if ablations:
            for n, u in ablation_units(d, cfg).items():
                abl_units.setdefault(n, []).extend(u)
        if conformal:
            conf_chunks.append(conformal_arrays(d, cfg))
        per_fold.append({"n_test_scenes": len(fs), "map": maps})
        ctx.log.info(f"[r2a] fold {i} ({time.time() - t0:.0f}s): {maps.get('ADVERSE_POOLED')}")
    res["per_fold"] = per_fold
    res["oof_map"] = {p: {a: map_of_units(u, cfg) for a, u in au.items()} for p, au in oof_units.items()}
    adv = oof_units["ADVERSE_POOLED"]
    A1 = scene_paired(adv["object"], adv["frame_cls"], cfg)
    A3 = scene_paired(adv["object"], adv["object_noradar"], cfg)
    hh = [np.concatenate(x) for x in zip(*oof_harm)]
    A2 = harm_summary(*hh, n_boot=cfg["bootstrap"]["n"])
    res["A1"] = {**A1, "refuted": bool(A1["diff"] <= 0 or A1["p_one_sided"] >= 0.05)}
    res["A2"] = {**A2["pooled"], "per_modality": {m: A2[m] for m in ("camera", "lidar", "radar") if m in A2},
                 "refuted": bool(A2["pooled"]["harm_rate"] < 0.02)}
    res["A3"] = {**A3, "refuted": bool(A3["diff"] <= 0 or A3["p_one_sided"] >= 0.05)}
    res["A1_clear"] = scene_paired(oof_units["clear0"]["object"], oof_units["clear0"]["frame_cls"], cfg)
    res["object_vs_fixed"] = scene_paired(adv["object"], adv["fixed_clear"], cfg)
    res["fold_deltas_A1"] = [f["map"]["ADVERSE_POOLED"]["object"] - f["map"]["ADVERSE_POOLED"]["frame_cls"]
                             for f in per_fold if "ADVERSE_POOLED" in f["map"]]
    if ablations:
        res["C_map"] = {n: map_of_units(u, cfg) for n, u in abl_units.items()}
        res["C_vs_full"] = compare({**abl_units, "object": adv["object"]}, cfg, list(abl_units))
        for v in res["C_vs_full"].values():
            v["contributes"] = bool(v["diff"] > 0 and v["p_holm"] < 0.05)
    if conformal:
        res["E"] = summarise_conformal(conf_chunks)
    if baselines:
        comp = [n for n in BASELINES if n != "oracle"]
        res["B_vs_ours"] = compare(adv, cfg, comp)
        sq, orc, ours = (res["oof_map"]["ADVERSE_POOLED"][a] for a in ("fixed_clear", "oracle", "object"))
        res["headroom_captured"] = (ours - sq) / max(orc - sq, 1e-9)
    # ---- standard protocol: official train_detect -> val
    arms = fit_arms(ctx, baselines)
    maps, units, harm = evaluate_split(ctx, arms, cfg)
    res["official"] = {"map": maps, "A1": scene_paired(units["ADVERSE_POOLED"]["object"],
                                                       units["ADVERSE_POOLED"]["frame_cls"], cfg),
                       "harm": harm_summary(*harm, n_boot=cfg["bootstrap"]["n"])["pooled"]}
    res["params"] = {a: int(m.n_params()) for a, m in arms.items() if hasattr(m, "n_params")}
    return res


def report(res):
    om = res["oof_map"]
    pools = [p for p in ("clear0", "ADVERSE_POOLED", "rain1", "snow1", "fog1", "night1") if p in om]
    arms = [a for a in ("fixed_clear", "frame_cls", "object", "object_noradar", "proben", "dempster_shafer",
                        "clocs", "set_transformer", "moe_frame", "oracle") if a in om["ADVERSE_POOLED"]]
    md = (f"## R2-A -- MAN TruckScenes (real weather, 4D radar): {res['n_frames']} keyframes, {res['n_scenes']} "
          f"scenes; scenes per condition {res['scenes_per_condition']}\n\n"
          "Primary: 5-fold scene CV stratified by weather, out-of-fold pooled Car mAP (every scene tested once).\n\n")
    md += _md_table(["method"] + pools + ["ours − method, adverse (scene bootstrap)"],
                    [[LABEL.get(a, a)] + [fmt(om[p].get(a)) for p in pools] +
                     [("--" if a == "object" else
                       f"{fmt(res['A1']['diff'], 4)} (p {fmt(res['A1']['p_one_sided'], 3)})" if a == "frame_cls" else
                       f"{fmt(res['A3']['diff'], 4)} (p {fmt(res['A3']['p_one_sided'], 3)})" if a == "object_noradar" else
                       f"{fmt(res['B_vs_ours'][a]['diff'], 4)} (p_Holm {fmt(res['B_vs_ours'][a]['p_holm'], 3)})"
                       if a in res.get("B_vs_ours", {}) else "")] for a in arms])
    a2 = res["A2"]
    md += (f"\n\nR2-A1 (ours > per-frame, adverse): Δ {fmt(res['A1']['diff'], 4)} [{fmt(res['A1']['ci95'][0], 4)}, "
           f"{fmt(res['A1']['ci95'][1], 4)}], p {fmt(res['A1']['p_one_sided'], 3)}, {res['A1']['n_scenes']} adverse "
           f"scenes; folds ahead {sum(x > 0 for x in res['fold_deltas_A1'])}/{len(res['fold_deltas_A1'])}. "
           f"Refuted: {res['A1']['refuted']}\n\n"
           f"R2-A2 (status-quo harm ≥ 2 %): {100 * a2['harm_rate']:.2f} % [{100 * a2['harm_ci'][0]:.2f}, "
           f"{100 * a2['harm_ci'][1]:.2f}] over {a2['n_pairs']} pairs. Refuted: {a2['refuted']}\n\n"
           f"R2-A3 (4D radar adds value, adverse): Δ {fmt(res['A3']['diff'], 4)} [{fmt(res['A3']['ci95'][0], 4)}, "
           f"{fmt(res['A3']['ci95'][1], 4)}], p {fmt(res['A3']['p_one_sided'], 3)}. Refuted: {res['A3']['refuted']}\n")
    if "headroom_captured" in res:
        md += f"\nOracle headroom captured by ours: {100 * res['headroom_captured']:.0f} %\n"
    if "C_vs_full" in res:
        md += "\n### R2-C on TruckScenes (out-of-fold, adverse; each mechanism removed and retrained per fold)\n\n"
        md += _md_table(["removed", "adverse mAP", "full − variant (p_Holm)", "contributes?"],
                        [[n, fmt(res["C_map"][n]), f"{fmt(v['diff'], 4)} ({fmt(v['p_holm'], 3)})",
                          "yes" if v["contributes"] else "no"] for n, v in res["C_vs_full"].items()])
    if "E" in res:
        md += ("\n\n### R2-E on TruckScenes (alpha = 0.1; calibrated on held-out CLEAR scenes only; out-of-fold)\n\n")
        md += _md_table(["tag", "cov vanilla", "cov conflict", "cov degr.-norm. / radius", "FNR", "FNR floor",
                         "flagged", "FNR unflagged"],
                        [[t, fmt(v["cov_vanilla"]), fmt(v["cov_conflict"]),
                          f"{fmt(v['cov_degradation_normalised'])} / {fmt(v['rad_degradation_normalised'], 2)} m",
                          fmt(v["fnr_clear_calibrated"]), fmt(v["fnr_floor_accept_all"]), fmt(v["flagged_out_of_support"]),
                          fmt(v["fnr_in_support"])] for t, v in res["E"].items()])
    o = res["official"]
    md += ("\n### Standard protocol (official train_detect → val)\n\n" +
           _md_table(["method"] + [p for p in pools if p in o["map"]],
                     [[LABEL.get(a, a)] + [fmt(o["map"][p].get(a)) for p in pools if p in o["map"]] for a in arms]) +
           f"\n\nOfficial split, ours − per-frame (adverse): Δ {fmt(o['A1']['diff'], 4)}, p {fmt(o['A1']['p_one_sided'], 3)} "
           f"({o['A1']['n_scenes']} adverse val scenes); status-quo harm {100 * o['harm']['harm_rate']:.2f} %\n")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    ctx = ts_ctx()
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run(ctx, args.k, not args.no_baselines)
    save_json(res, out / "r2a_truckscenes.json")
    (out / "R2A.md").write_text(report(res), encoding="utf-8")
    ctx.log.info(f"r2a done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
