"""Experiments 1–8 of PLAN.md §10, end to end on cached candidate tables.

    python -m src.candidates                      # build tables from the detector cache
    python -m src.run_experiments --exp all       # or: --exp 1 2 4
Outputs: results/exp*/ (JSON + markdown tables + figures), results/REPORT.md.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from .arbitrate import COND_GROUPS, WeatherClassifier, _frame_feat_matrix, fit_arbiter, merge_tables, subset_table
from .candidates import load_table
from .common import MODALITIES, get_logger, load_config, resolve, save_json, source_tree_hash, seed_everything
from .conformal import (LocalisationConformal, WeightedLocalisationConformal, crc_threshold, frame_false_positives,
                        frame_fnr, residual_conflict)
from .evaluate import choose_tau_f1, eval_map, frame_detections, gt_lists, map_of_units, per_frame_units
from .harm import counterfactual_predictions, harm_labels, harm_summary
from .metrics import bootstrap_ci, paired_bootstrap_pvalue, safe_auroc
from . import plots

ARMS = ["fixed", "confidence", "frame_cls", "frame_oracle", "frame_feats", "object"]


class Ctx:
    """Loads data once, trains models lazily, caches fitted models to results/models/."""

    def __init__(self, cfg, dataset="boreas", table=None):
        self.cfg, self.dataset = cfg, dataset
        self.out = resolve(cfg, "results_root")
        self.log = get_logger("exp", self.out / "logs" / "experiments.log")
        if table is None:
            root = resolve(cfg, "candidates_root") / dataset
            tables = [load_table(p) for p in sorted(root.glob("*.npz")) if not p.name.startswith("_")]
            if not tables:
                raise SystemExit(f"no candidate tables in {root}; run python -m src.candidates first")
            table = merge_tables(tables)
        self.T = table
        fr = self.T["frames"]
        self.split = np.array([f["split"] for f in fr])
        self.cond = np.array([f["condition"] for f in fr])
        self.tag = np.array([f"{f['condition']}{f['severity']}" for f in fr])
        self.scene = np.array([f["scene"] for f in fr])
        heldout = cfg["weather"]["heldout_for_head"]
        # internal validation = 20 % of TRAIN scenes (deterministic); cal/test untouched
        train_scenes = sorted(set(self.scene[self.split == "train"]))
        rng = np.random.default_rng(cfg["seed"] + 1)
        val_scenes = set(rng.choice(train_scenes, max(1, len(train_scenes) // 5), replace=False)) if train_scenes else set()
        self.is_val = np.isin(self.scene, list(val_scenes)) & (self.split == "train")
        self.is_fit = (self.split == "train") & ~self.is_val
        self.train_cond = ~np.isin(self.cond, heldout)
        self.tags = [t for t in _tag_order(cfg) if t in set(self.tag)]
        self.adverse_tags = [t for t in self.tags if t != "clear0"]
        self.models = {}
        self.model_dir = self.out / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"frames={len(fr)} candidates={len(self.T['cand']['y'])} tags={self.tags} "
                      f"scenes fit/val/cal/test={len(set(self.scene[self.is_fit]))}/{len(val_scenes)}/"
                      f"{len(set(self.scene[self.split == 'cal']))}/{len(set(self.scene[self.split == 'test']))}")

    def sub(self, mask):
        return subset_table(self.T, mask)

    # ---- status quo and harm labels
    def status_quo(self):
        if "fixed_clear" in self.models:
            return self.models["fixed_clear"]
        clear = self.tag == "clear0"
        arb = self._load_or_fit("fixed_clear", lambda: fit_arbiter(
            "fixed_clear", self.sub(self.is_fit & clear), self.sub(self.is_val & clear), self.cfg, log=self.log,
            modality_dropout=0.0))
        train_clear = self.sub((self.split == "train") & clear)
        arb.tau, arb.f1 = choose_tau_f1(train_clear, arb.predict(train_clear), self.cfg)
        self.models["fixed_clear"] = arb
        return arb

    def harm(self):
        if hasattr(self, "_harm"):
            return self._harm
        path = self.model_dir / "harm_labels.npz"
        sq = self.status_quo()
        if path.exists():
            d = np.load(path)
            self._harm = (d["harm"], d["help"], d["soft"])
        else:
            h, hp, soft, _ = harm_labels(sq, self.T, sq.tau, self.cfg)
            np.savez_compressed(path, harm=h, help=hp, soft=soft)
            self._harm = (h, hp, soft)
        return self._harm

    def weather_clf(self):
        if hasattr(self, "_wclf"):
            return self._wclf
        mask = (self.split == "train") & self.train_cond
        ff = _frame_feat_matrix(self.T)
        clf = WeatherClassifier().fit(ff[mask], self.cond[mask])
        test = (self.split == "test")
        self.wclf_acc = {"train_conditions_test_acc": clf.accuracy(ff[test & self.train_cond], self.cond[test & self.train_cond]),
                         "classes": list(clf.clf.classes_)}
        self._wclf = clf
        return clf

    def arm(self, name, blocks=None, context=None, tag=None, dropout=None):
        key = tag or name
        if key in self.models:
            return self.models[key]
        h, _, _ = self.harm()
        fit_mask = self.is_fit & self.train_cond
        val_mask = self.is_val & self.train_cond
        cand_fit = fit_mask[self.T["cand"]["frame"]]
        cand_val = val_mask[self.T["cand"]["frame"]]
        wclf = self.weather_clf() if name == "frame_cls" else None
        arb = self._load_or_fit(key, lambda: fit_arbiter(
            name, self.sub(fit_mask), self.sub(val_mask), self.cfg, harm_labels=h[cand_fit], val_harm=h[cand_val],
            weather_clf=wclf, log=self.log, blocks=blocks, context=context, seed=self.cfg["seed"],
            modality_dropout=dropout))
        arb.weather_clf = wclf
        train_clear = self.sub((self.split == "train") & (self.tag == "clear0"))
        arb.tau, arb.f1 = choose_tau_f1(train_clear, arb.predict(train_clear), self.cfg)
        self.models[key] = arb
        return arb

    def _load_or_fit(self, key, fit_fn):
        path = self.model_dir / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                arb = pickle.load(f)
            self.log.info(f"[{key}] loaded from {path.name}")
            return arb
        t0 = time.time()
        arb = fit_fn()
        arb.fit_seconds = time.time() - t0
        with open(path, "wb") as f:
            pickle.dump(arb, f)
        return arb

    def test_table(self, tag=None, tags=None):
        m = self.split == "test"
        if tag:
            m &= self.tag == tag
        if tags:
            m &= np.isin(self.tag, tags)
        return self.sub(m)


def _tag_order(cfg):
    out = []
    for c, sevs in cfg["weather"]["conditions"].items():
        out += [f"{c}{s}" for s in sevs]
    return out


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def fmt(v, p=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


# ======================================================================================================
# Exp 1 — harm rate

def exp1(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp1_harm"
    out.mkdir(parents=True, exist_ok=True)
    sq = ctx.status_quo()
    harm, helpful, soft = ctx.harm()
    c = ctx.T["cand"]
    test = ctx.split[c["frame"]] == "test"
    rng_obj = np.hypot(np.nan_to_num(c["xy"][:, :, 0]).max(1), np.nan_to_num(c["xy"][:, :, 1]).max(1))
    anchor_range = c["probes"][:, 0]
    res = {"tau": sq.tau, "f1_clear_train": sq.f1, "per_tag": {}, "ap_level": {}}
    bins = cfg["conformal"]["mondrian_range_bins_m"]
    for tag in ctx.tags + ["ADVERSE_POOLED"]:
        sel = test & (np.isin(ctx.tag[c["frame"]], ctx.adverse_tags) if tag == "ADVERSE_POOLED" else ctx.tag[c["frame"]] == tag)
        if not sel.any():
            continue
        res["per_tag"][tag] = harm_summary(harm[sel], helpful[sel], c["cls"][sel], c["frame"][sel],
                                           n_boot=cfg["bootstrap"]["n"], ranges=anchor_range[sel], bins=bins)
        res["per_tag"][tag]["soft_harm_rate"] = float(np.mean(soft[sel][(soft[sel] >= 0) & (c["cls"][sel][:, None] == 0)])) \
            if ((soft[sel] >= 0) & (c["cls"][sel][:, None] == 0)).any() else float("nan")
    # AP-level harm: mAP with all sensors vs with one removed (status-quo fusion)
    for tag in ctx.tags:
        tt = ctx.test_table(tag)
        if not tt["frames"]:
            continue
        preds = counterfactual_predictions(sq, tt)
        row = {"all": eval_map(tt, preds["all"], cfg)[0]}
        for m in MODALITIES:
            row[f"minus_{m}"] = eval_map(tt, preds[f"minus_{m}"], cfg)[0]
        row["harmful_modalities"] = [m for m in MODALITIES if row[f"minus_{m}"] > row["all"]]
        res["ap_level"][tag] = row
    # Secondary slices (reported, NOT part of the pre-registered rule): the primary denominator is
    # dominated by radar clutter candidates where arbitration cannot matter.
    res["slices"] = {}
    n_present = c["present"].sum(1)
    for sname, smask in (("multimodal_ge2", n_present >= 2), ("gt_backed", c["y"] > 0),
                         ("multimodal_ge2_gt_backed", (n_present >= 2) & (c["y"] > 0))):
        res["slices"][sname] = {}
        for tag in ("clear0", "ADVERSE_POOLED"):
            sel = test & smask & (np.isin(ctx.tag[c["frame"]], ctx.adverse_tags) if tag == "ADVERSE_POOLED"
                                  else ctx.tag[c["frame"]] == tag)
            if sel.any():
                res["slices"][sname][tag] = harm_summary(harm[sel], helpful[sel], c["cls"][sel], c["frame"][sel],
                                                         n_boot=cfg["bootstrap"]["n"])
    # Harm depends on where the system is operated; sweep the decision threshold (secondary).
    preds_all = counterfactual_predictions(sq, ctx.T)
    adverse_sel = test & np.isin(ctx.tag[c["frame"]], ctx.adverse_tags)
    res["tau_sweep"] = {}
    for tau in (0.2, 0.3, sq.tau, 0.5, 0.7, 0.9):
        h_t, help_t, _, _ = harm_labels(sq, ctx.T, tau, cfg, preds=preds_all)
        s = harm_summary(h_t[adverse_sel], help_t[adverse_sel], c["cls"][adverse_sel], c["frame"][adverse_sel], n_boot=100)
        res["tau_sweep"][f"{tau:.3f}"] = {"harm": s["pooled"]["harm_rate"], "help": s["pooled"]["help_rate"],
                                          "camera": s["camera"]["harm_rate"], "lidar": s["lidar"]["harm_rate"],
                                          "radar": s["radar"]["harm_rate"]}
    pooled = res["per_tag"].get("ADVERSE_POOLED", {}).get("pooled", {})
    clear_pooled = res["per_tag"].get("clear0", {}).get("pooled", {})
    rate, ci = pooled.get("harm_rate", float("nan")), pooled.get("harm_ci", [float("nan")] * 2)
    res["H1"] = {"pooled_adverse_harm_rate": rate, "ci95": ci,
                 "refuted": bool(rate < 0.02), "weak_premise_flag": bool(ci[1] < 0.02),
                 "clear_harm_rate": clear_pooled.get("harm_rate"),
                 "weather_excess_harm": (rate - clear_pooled.get("harm_rate", np.nan)),
                 "harm_rate_multimodal_adverse": res["slices"]["multimodal_ge2"].get("ADVERSE_POOLED", {}).get("pooled", {}).get("harm_rate"),
                 "ap_level_harm_cells": sum(len(v["harmful_modalities"]) for v in res["ap_level"].values())}
    save_json(res, out / "exp1_harm.json")
    # table + figure
    rows = []
    for tag, v in res["per_tag"].items():
        rows.append([tag] + [f"{fmt(100 * v[m]['harm_rate'], 1)} ({fmt(100 * v[m]['harm_ci'][0], 1)}–{fmt(100 * v[m]['harm_ci'][1], 1)})"
                             for m in ("pooled",) + MODALITIES] + [fmt(100 * v["pooled"]["help_rate"], 1)])
    md = "# Exp 1 — Harm rate under status-quo fusion (Car, test scenes)\n\n"
    md += f"Status quo = fixed-weight late fusion fitted on clear; operating threshold τ = {sq.tau:.3f} (F1 {sq.f1:.3f} on clear train).\n\n"
    md += "Decision-flip harm rate, % of candidates containing the modality (95 % frame-bootstrap CI)\n\n"
    md += _md_table(["condition", "pooled", "camera", "lidar", "radar", "help % (pooled)"], rows)
    md += "\n\n## AP-level harm (Car mAP, status quo): removing a sensor *improves* mAP when the cell exceeds `all`\n\n"
    md += _md_table(["condition", "all", "− camera", "− lidar", "− radar"],
                    [[t, fmt(v["all"]), fmt(v["minus_camera"]), fmt(v["minus_lidar"]), fmt(v["minus_radar"])] for t, v in res["ap_level"].items()])
    md += "\n\n## Operating-threshold sweep (adverse pooled; τ chosen on clear = {:.3f})\n\n".format(sq.tau)
    md += _md_table(["τ", "pooled harm %", "help %", "camera %", "lidar %", "radar %"],
                    [[t, fmt(100 * v["harm"], 2), fmt(100 * v["help"], 2), fmt(100 * v["camera"], 2),
                      fmt(100 * v["lidar"], 2), fmt(100 * v["radar"], 2)] for t, v in res["tau_sweep"].items()])
    md += "\n\n## Secondary slices (reported, not part of the pre-registered rule)\n\n"
    md += _md_table(["slice", "condition", "pooled harm %", "camera %", "lidar %", "radar %", "help %", "n pairs"],
                    [[s, t, fmt(100 * v["pooled"]["harm_rate"], 2), fmt(100 * v["camera"]["harm_rate"], 2),
                      fmt(100 * v["lidar"]["harm_rate"], 2), fmt(100 * v["radar"]["harm_rate"], 2),
                      fmt(100 * v["pooled"]["help_rate"], 2), str(v["pooled"]["n_pairs"])]
                     for s, per in res["slices"].items() for t, v in per.items()])
    md += f"\n\n**H1 verdict:** pooled adverse harm = {fmt(100 * rate, 2)} % (CI {fmt(100 * ci[0], 2)}–{fmt(100 * ci[1], 2)}); " \
          f"refuted (<2 %): **{res['H1']['refuted']}**; clear-weather harm = {fmt(100 * res['H1']['clear_harm_rate'], 2)} %, " \
          f"weather-induced excess = {fmt(100 * res['H1']['weather_excess_harm'], 2)} pp; " \
          f"AP-level harm cells: {res['H1']['ap_level_harm_cells']}.\n"
    (out / "exp1_harm.md").write_text(md, encoding="utf-8")
    groups = [t for t in ctx.tags if t in res["per_tag"]]
    vals = {m: {t: 100 * res["per_tag"][t][m]["harm_rate"] for t in groups} for m in MODALITIES}
    errs = {m: {t: tuple(100 * np.array(res["per_tag"][t][m]["harm_ci"])) for t in groups} for m in MODALITIES}
    plots.grouped_bars(groups, list(MODALITIES), vals, plots.MOD_COLOR, {m: m for m in MODALITIES},
                       "decision-flip harm rate (%)", "Harm from adding each sensor, status-quo fusion (Car, test)",
                       out / "harm_by_condition.png", errs, ref_line=(2.0, "2 % gate"))
    ctx.log.info(f"exp1: H1 {res['H1']}")
    return res


# ======================================================================================================
# Exp 2 — routing comparison

def exp2(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp2_routing"
    out.mkdir(parents=True, exist_ok=True)
    arms = {"fixed_clear": ctx.status_quo(), **{a: ctx.arm(a) for a in ARMS}}
    harm, _, _ = ctx.harm()
    c = ctx.T["cand"]
    res = {"n_params": {a: m.n_params() for a, m in arms.items()}, "map": {}, "harm_auroc": {},
           "weather_classifier": getattr(ctx, "wclf_acc", None)}
    for a, m in arms.items():
        res["map"][a] = {}
        for tag in ctx.tags:
            tt = ctx.test_table(tag)
            res["map"][a][tag] = eval_map(tt, m.predict(tt), cfg)[0]
    pools = {"ADVERSE_POOLED": ctx.adverse_tags, "HELDOUT": [t for t in ctx.tags if t.rstrip("0123456789") in cfg["weather"]["heldout_for_head"]],
             "SEEN_ADVERSE": [t for t in ctx.adverse_tags if t.rstrip("0123456789") not in cfg["weather"]["heldout_for_head"]]}
    units = {}
    for pname, tags in pools.items():
        if not tags:
            continue
        tt = ctx.test_table(tags=tags)
        sel_c = (ctx.split[c["frame"]] == "test") & np.isin(ctx.tag[c["frame"]], tags)
        units[pname] = {}
        for a, m in arms.items():
            pr = m.predict(tt)
            u = per_frame_units(tt, pr, cfg)
            units[pname][a] = u
            ci = bootstrap_ci(u, lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
            res["map"][a][pname] = map_of_units(u, cfg)
            res["map"][a][pname + "_ci"] = ci
            h = harm[sel_c]
            car = tt["cand"]["cls"] == 0
            ys, ss = [], []
            for mi in range(3):
                ok = (h[:, mi] >= 0) & car
                ys.append(h[ok, mi])
                ss.append(pr["harm_prob"][ok, mi])
            res["harm_auroc"].setdefault(a, {})[pname] = safe_auroc(np.concatenate(ys), np.concatenate(ss))
    # H2 test
    h2 = {}
    for pname in units:
        diff, p = paired_bootstrap_pvalue(units[pname]["object"], units[pname]["frame_cls"],
                                          lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
        diff_ff, p_ff = paired_bootstrap_pvalue(units[pname]["object"], units[pname]["frame_feats"],
                                                lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
        diff_cf, p_cf = paired_bootstrap_pvalue(units[pname]["object"], units[pname]["confidence"],
                                                lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
        h2[pname] = {"map_object_minus_frame_cls": diff, "p_one_sided": p,
                     "map_object_minus_frame_feats": diff_ff, "p_vs_frame_feats": p_ff,
                     # reported, not part of the pre-registered refutation rule
                     "map_object_minus_confidence": diff_cf, "p_vs_confidence": p_cf,
                     "auroc_confidence": res["harm_auroc"]["confidence"][pname],
                     "auroc_object": res["harm_auroc"]["object"][pname], "auroc_frame_cls": res["harm_auroc"]["frame_cls"][pname]}
    main = h2.get("ADVERSE_POOLED", {})
    res["H2"] = {**h2, "refuted": bool(not main or main["map_object_minus_frame_cls"] <= 0 or main["p_one_sided"] >= 0.05
                                       or not (main["auroc_object"] > main["auroc_frame_cls"]))}
    save_json(res, out / "exp2_routing.json")
    arms_order = ["fixed_clear"] + ARMS
    cols = ctx.tags + [p for p in ("SEEN_ADVERSE", "HELDOUT", "ADVERSE_POOLED") if p in units]
    md = "# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)\n\n"
    md += _md_table(["arm", "params"] + cols,
                    [[plots.ARM_LABEL[a], str(res["n_params"][a])] + [fmt(res["map"][a].get(t)) for t in cols] for a in arms_order])
    md += "\n\n## Harm-prediction AUROC (Car candidates, per present modality)\n\n"
    md += _md_table(["arm"] + list(units), [[plots.ARM_LABEL[a]] + [fmt(res["harm_auroc"].get(a, {}).get(p)) for p in units] for a in ARMS])
    md += "\n\n## H2 test (paired frame bootstrap)\n\n"
    md += _md_table(["pool", "Δ mAP obj − frame_cls", "p", "Δ mAP obj − frame_feats", "p", "Δ mAP obj − confidence", "p",
                     "AUROC obj", "AUROC frame_cls", "AUROC confidence"],
                    [[p, fmt(v["map_object_minus_frame_cls"], 4), fmt(v["p_one_sided"]), fmt(v["map_object_minus_frame_feats"], 4),
                      fmt(v["p_vs_frame_feats"]), fmt(v["map_object_minus_confidence"], 4), fmt(v["p_vs_confidence"]),
                      fmt(v["auroc_object"]), fmt(v["auroc_frame_cls"]), fmt(v["auroc_confidence"])] for p, v in h2.items()])
    md += "\n\n(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)"
    md += f"\n\n**H2 refuted:** {res['H2']['refuted']}\n"
    if res["weather_classifier"]:
        md += f"\nWeather classifier accuracy on test frames of training conditions: {fmt(res['weather_classifier']['train_conditions_test_acc'])}\n"
    (out / "exp2_routing.md").write_text(md, encoding="utf-8")
    plots.grouped_bars(ctx.tags, arms_order, res["map"], plots.ARM_COLOR, plots.ARM_LABEL, "Car mAP",
                       "Fused Car mAP by condition and routing arm (test)", out / "map_by_condition.png")
    ctx.log.info(f"exp2: H2 {res['H2']}")
    return res


# ======================================================================================================
# Exp 3 — feature ablation

def exp3(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp3_ablation"
    out.mkdir(parents=True, exist_ok=True)
    from .arbitrate import ALL_BLOCKS
    variants = {"full": (ALL_BLOCKS, True), "no_context": (ALL_BLOCKS, False)}
    for b in ALL_BLOCKS:
        variants[f"-{b}"] = ([x for x in ALL_BLOCKS if x != b], True)
    res = {}
    pools = {"ADVERSE_POOLED": ctx.adverse_tags,
             "HELDOUT": [t for t in ctx.tags if t.rstrip("0123456789") in cfg["weather"]["heldout_for_head"]],
             "clear0": ["clear0"]}
    for name, (blocks, context) in variants.items():
        arb = ctx.models["object"] if name == "full" and "object" in ctx.models else \
            ctx.arm("object", blocks=blocks, context=context, tag=f"ablation{name}")
        res[name] = {}
        for p, tags in pools.items():
            if tags:
                tt = ctx.test_table(tags=tags)
                res[name][p] = eval_map(tt, arb.predict(tt), cfg)[0]
    save_json(res, out / "exp3_ablation.json")
    full = res["full"]
    md = "# Exp 3 — Ablation over degradation-feature groups (per-object arm, Car mAP, test)\n\n"
    md += _md_table(["variant"] + [f"{p} (Δ vs full)" for p in pools],
                    [[n] + [f"{fmt(v.get(p))} ({fmt(v.get(p, np.nan) - full.get(p, np.nan), 3)})" for p in pools] for n, v in res.items()])
    (out / "exp3_ablation.md").write_text(md, encoding="utf-8")
    return res


# ======================================================================================================
# Exp 4 — clear -> adverse risk-control transfer

def _loc_data(arb, tbl, cfg):
    pr = arb.predict(tbl)
    c = tbl["cand"]
    member_xy, lam_r, _ = arb.prior.apply(c)
    det = (c["cls"] == 0) & (c["y"] > 0) & (pr["score"] >= arb.tau) & ~c["ignore"]
    err = np.linalg.norm(pr["xy"] - np.nan_to_num(c["gt_xy"]), axis=1)
    rng = np.hypot(pr["xy"][:, 0], pr["xy"][:, 1])
    phi = residual_conflict(member_xy, c["present"], pr["xy"], lam_r, c["score"])
    ff = c["frame_feats"]
    return {"err": err[det], "rng": rng[det], "phi": phi[det], "ff": ff[det], "frame": c["frame"][det]}, pr


def exp4(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp4_conformal"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    cal_mask = (ctx.split == "cal") & np.isin(ctx.cond, cfg["weather"]["calibration_conditions"])
    for arm_name in ("fixed_clear", "object"):
        arb = ctx.status_quo() if arm_name == "fixed_clear" else ctx.arm("object")
        train_tbl = ctx.sub((ctx.split == "train") & (ctx.train_cond if arm_name == "object" else ctx.tag == "clear0"))
        cal_tbl = ctx.sub(cal_mask)
        dtr, _ = _loc_data(arb, train_tbl, cfg)
        dcal, pr_cal = _loc_data(arb, cal_tbl, cfg)
        res[arm_name] = {"loc": {}, "crc": {}, "n_cal_objects": int(len(dcal["err"])), "n_cal_frames": len(cal_tbl["frames"])}
        cal_dets = frame_detections(cal_tbl, pr_cal, 0, cfg["association"]["nms_center_m"])
        cal_gts, _ = gt_lists(cal_tbl, 0)
        grid = np.unique(np.concatenate([[0.0], np.quantile(pr_cal["score"][pr_cal["score"] > 0], np.linspace(0, 1, 200))]))
        for alpha in cfg["conformal"]["alpha"]:
            methods = {
                "vanilla": LocalisationConformal(False).fit_scale(dtr["err"], dtr["rng"], dtr["phi"]),
                "conflict": LocalisationConformal(True).fit_scale(dtr["err"], dtr["rng"], dtr["phi"]),
                "conflict_mondrian": LocalisationConformal(True, cfg["conformal"]["mondrian_range_bins_m"]).fit_scale(dtr["err"], dtr["rng"], dtr["phi"]),
            }
            for mth in methods.values():
                mth.calibrate(dcal["err"], dcal["rng"], dcal["phi"], alpha)
            lam = crc_threshold(lambda l: frame_fnr(cal_dets, cal_gts, 2.0, l), sum(len(g) > 0 for g in cal_gts), alpha, list(grid))
            ra = res[arm_name]["loc"].setdefault(str(alpha), {})
            rc = res[arm_name]["crc"].setdefault(str(alpha), {"lambda": lam})
            for tag in ctx.tags:
                tt = ctx.test_table(tag)
                d, pr = _loc_data(arb, tt, cfg)
                ra[tag] = {k: m.evaluate(d["err"], d["rng"], d["phi"]) for k, m in methods.items()}
                if cfg["conformal"]["weighted_shift"] and tag != "clear0" and len(d["err"]):
                    # ratio fitted on half the test frames (unlabelled features), coverage measured on the other half
                    frames = np.unique(d["frame"])
                    rs = np.random.default_rng(0).permutation(frames)
                    halves = [rs[: len(rs) // 2], rs[len(rs) // 2:]]
                    covs, rads, infs, esss = [], [], [], []
                    ff_frames = _frame_feat_matrix(tt)
                    for fit_h, ev_h in (halves, halves[::-1]):
                        w = WeightedLocalisationConformal(True).fit_scale(dtr["err"], dtr["rng"], dtr["phi"])
                        w.fit_ratio(_frame_feat_matrix(cal_tbl), ff_frames[fit_h])
                        ev = np.isin(d["frame"], ev_h)
                        r = w.evaluate_weighted(dcal["err"], dcal["rng"], dcal["phi"], dcal["ff"],
                                                d["err"][ev], d["rng"][ev], d["phi"][ev], d["ff"][ev], alpha)
                        covs.append(r["coverage"]); rads.append(r["mean_radius"]); infs.append(r["frac_infinite"]); esss.append(r["ess"])
                    ra[tag]["weighted_conflict"] = {"coverage": float(np.nanmean(covs)), "mean_radius": float(np.mean(rads)),
                                                    "frac_infinite": float(np.mean(infs)), "ess": float(np.mean(esss)),
                                                    "nominal": 1 - alpha, "n": int(len(d["err"]))}
                dets = frame_detections(tt, pr, 0, cfg["association"]["nms_center_m"])
                gts, _ = gt_lists(tt, 0)
                fnr = frame_fnr(dets, gts, 2.0, lam)
                rc[tag] = {"fnr": float(fnr.mean()) if len(fnr) else float("nan"),
                           "fp_per_frame": float(frame_false_positives(dets, gts, 2.0, lam).mean())}
            for pool_name, tags in (("SNOW_POOLED", [t for t in ctx.tags if t.startswith("snow")]),
                                    ("ADVERSE_POOLED", ctx.adverse_tags)):
                if not tags:
                    continue
                tt = ctx.test_table(tags=tags)
                d, pr = _loc_data(arb, tt, cfg)
                ra[pool_name] = {k: m.evaluate(d["err"], d["rng"], d["phi"]) for k, m in methods.items()}
                dets = frame_detections(tt, pr, 0, cfg["association"]["nms_center_m"])
                gts, _ = gt_lists(tt, 0)
                fnr = frame_fnr(dets, gts, 2.0, lam)
                rc[pool_name] = {"fnr": float(fnr.mean()) if len(fnr) else float("nan"),
                                 "fp_per_frame": float(frame_false_positives(dets, gts, 2.0, lam).mean())}
    # H3 on the per-object arm at alpha = 0.10
    a = "0.1"
    snow = res["object"]["loc"].get(a, {}).get("SNOW_POOLED", {})
    van, con = snow.get("vanilla", {}), snow.get("conflict", {})
    fnr_snow = res["object"]["crc"].get(a, {}).get("SNOW_POOLED", {}).get("fnr", float("nan"))
    clear = res["object"]["loc"].get(a, {}).get("clear0", {})
    gap_v = 0.9 - van.get("coverage", np.nan)
    gap_c = 0.9 - con.get("coverage", np.nan)
    res["H3"] = {"snow_vanilla_coverage": van.get("coverage"), "snow_conflict_coverage": con.get("coverage"),
                 "snow_fnr_at_alpha_0.1": fnr_snow,
                 "refuted": bool((van.get("coverage", 1) < 0.85) or (fnr_snow > 0.15)),
                 "conflict_closes_half_gap": bool(gap_v > 0 and gap_c <= 0.5 * gap_v),
                 "clear_radius_ratio_conflict_over_vanilla": (clear.get("conflict", {}).get("mean_radius", np.nan) /
                                                              max(clear.get("vanilla", {}).get("mean_radius", np.nan), 1e-9)) if clear else None}
    save_json(res, out / "exp4_conformal.json")
    md = "# Exp 4 — Calibrate on clear (cal scenes), deploy under weather (test scenes)\n\n"
    for arm_name in res:
        if arm_name == "H3":
            continue
        for alpha in cfg["conformal"]["alpha"]:
            ra = res[arm_name]["loc"][str(alpha)]
            rc = res[arm_name]["crc"][str(alpha)]
            md += f"\n## {plots.ARM_LABEL[arm_name]} — α = {alpha} (nominal coverage {1 - alpha:.2f}), CRC λ̂ = {rc['lambda']:.3f}\n\n"
            rows = []
            for tag, v in ra.items():
                w = v.get("weighted_conflict", {})
                rows.append([tag] + [f"{fmt(v[k]['coverage'])} / {fmt(v[k]['mean_radius'], 2)} m" for k in ("vanilla", "conflict", "conflict_mondrian")]
                            + [f"{fmt(w.get('coverage'))} / {fmt(w.get('mean_radius'), 2)} m (ESS {fmt(w.get('ess'), 0)})" if w else "–",
                               fmt(rc.get(tag, {}).get("fnr")), fmt(rc.get(tag, {}).get("fp_per_frame"), 2), str(v["vanilla"]["n"])])
            md += _md_table(["condition", "vanilla cov / radius", "conflict", "conflict + Mondrian", "weighted conflict",
                             f"FNR (target ≤ {alpha})", "FP/frame", "n objects"], rows)
            md += "\n"
    md += f"\n**H3 (per-object arm, α = 0.1):** {res['H3']}\n"
    (out / "exp4_conformal.md").write_text(md, encoding="utf-8")
    a_res = res["object"]["loc"]["0.1"]
    vals = {k: {t: a_res[t][k]["coverage"] for t in ctx.tags if t in a_res} for k in ("vanilla", "conflict", "conflict_mondrian")}
    colors = {"vanilla": plots.SERIES[0], "conflict": plots.SERIES[1], "conflict_mondrian": plots.SERIES[2]}
    plots.lines(ctx.tags, list(vals), vals, colors, {"vanilla": "vanilla score", "conflict": "conflict-inflated",
                                                     "conflict_mondrian": "conflict + range-Mondrian"},
                "empirical coverage", "test condition", "Localisation coverage, calibrated on clear (nominal 0.90)",
                out / "coverage_transfer.png", ref_line=(0.9, "nominal 0.90"))
    ctx.log.info(f"exp4: H3 {res['H3']}")
    return res


# ======================================================================================================
# Exp 5 — Boreas -> CADC

def exp5(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp5_cadc"
    out.mkdir(parents=True, exist_ok=True)
    root = resolve(cfg, "candidates_root") / "cadc"
    tables = sorted(root.glob("*.npz")) if root.exists() else []
    if not tables:
        msg = ("# Exp 5 — Boreas → CADC\n\n**Not run: no CADC candidate tables.** The CADC server is public: "
               "`python -m src.fetch_cadc`, then `python -m src.cadc_cache` then `python -m src.candidates "
               "--dataset cadc`, then rerun `--exp 5`. CADC has no radar; the per-object head is evaluated with the "
               "radar sensor removed (the modality-dropout training of Exp 7 is what makes this valid).\n")
        (out / "exp5_cadc.md").write_text(msg, encoding="utf-8")
        return {"status": "not_run", "reason": "CADC not downloaded"}
    T = merge_tables([load_table(p) for p in tables])
    fr = T["frames"]
    for f in fr:  # CADC is real snow; the oracle-weather arm is told "snow", moderate severity (not given per frame)
        f["condition"], f["severity"] = "snow", 2
    date = np.array([str(f["frame_id"]).split("_")[0] + "_" + str(f["frame_id"]).split("_")[1] + "_" + str(f["frame_id"]).split("_")[2]
                     for f in fr])
    c = T["cand"]
    res = {"n_frames": len(fr), "n_candidates": int(len(c["y"])), "n_gt_cars": int(sum((g["cls"] == 0).sum() for g in T["gts"])),
           "map": {}, "ap_level": {}, "harm": {}}
    arms = {"fixed_clear": NoRadar(ctx.status_quo()), **{a: NoRadar(ctx.arm(a)) for a in ARMS}}
    units = {}
    for a, arb in arms.items():
        pr = arb.predict(T)
        res["map"][a] = {"ALL": eval_map(T, pr, cfg)[0]}
        for d in sorted(set(date)):
            res["map"][a][d] = eval_map(subset_table(T, date == d), arb.predict(subset_table(T, date == d)), cfg)[0]
        units[a] = per_frame_units(T, pr, cfg)
        res["map"][a]["ALL_ci"] = bootstrap_ci(units[a], lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
    diff, p = paired_bootstrap_pvalue(units["object"], units["frame_cls"], lambda us: map_of_units(us, cfg), n=cfg["bootstrap"]["n"])
    res["object_minus_frame_cls"] = {"diff": diff, "p_one_sided": p}
    # H1 on REAL snow: AP-level and decision-flip harm of the status quo (camera and LiDAR only)
    sq = arms["fixed_clear"]
    preds = counterfactual_predictions(sq, T)
    res["ap_level"] = {k: eval_map(T, v, cfg)[0] for k, v in preds.items() if k != "minus_radar"}
    h, hp, _, _ = harm_labels(sq, T, sq.tau, cfg, preds=preds)
    h[:, 2], hp[:, 2] = -1, -1
    res["harm"] = harm_summary(h, hp, c["cls"], c["frame"], n_boot=cfg["bootstrap"]["n"])
    save_json(res, out / "exp5_cadc.json")
    dates = sorted(set(date))
    md = (f"# Exp 5 — Boreas-trained heads on CADC real snow (radar removed)\n\n{res['n_frames']} frames, "
          f"{res['n_gt_cars']} GT cars (≥ 25 pts, ROI). No retraining, no recalibration: models and τ come from Boreas.\n\n")
    md += _md_table(["arm", "Car mAP (95 % CI)"] + dates,
                    [[plots.ARM_LABEL[a], f"{fmt(v['ALL'])} ({fmt(v['ALL_ci'][0])}–{fmt(v['ALL_ci'][1])})"] + [fmt(v[d]) for d in dates]
                     for a, v in res["map"].items()])
    md += f"\n\nPer-object − per-frame classifier: Δ mAP = {fmt(diff, 4)}, one-sided p = {fmt(p)}.\n"
    md += "\n## Status-quo fusion with a sensor removed (AP-level harm on real snow)\n\n"
    md += _md_table(["all (camera + LiDAR)", "− camera (LiDAR only)", "− LiDAR (camera only)"],
                    [[fmt(res["ap_level"]["all"]), fmt(res["ap_level"]["minus_camera"]), fmt(res["ap_level"]["minus_lidar"])]])
    hs = res["harm"]
    md += "\n\n## Decision-flip harm on real snow (status quo, Car)\n\n" + _md_table(
        ["", "pooled", "camera", "lidar"],
        [["harm %", *[f"{fmt(100 * hs[k]['harm_rate'], 2)} ({fmt(100 * hs[k]['harm_ci'][0], 2)}–{fmt(100 * hs[k]['harm_ci'][1], 2)})"
                      for k in ("pooled", "camera", "lidar")]],
         ["help %", *[fmt(100 * hs[k]["help_rate"], 2) for k in ("pooled", "camera", "lidar")]]])
    (out / "exp5_cadc.md").write_text(md + "\n", encoding="utf-8")
    ctx.log.info(f"exp5: {res['map']['object']['ALL']:.3f} obj vs {res['map']['frame_cls']['ALL']:.3f} frame_cls; harm {hs['pooled']}")
    return res


class NoRadar:
    """Wraps a fitted arbiter so radar is always treated as absent (CADC has no radar)."""

    def __init__(self, arb):
        self.arb, self.tau = arb, arb.tau

    def predict(self, tbl, drop=None):
        d = np.zeros((len(tbl["frames"]), 3), bool) if drop is None else drop.copy()
        d[:, 2] = True
        return self.arb.predict(tbl, d)


# ======================================================================================================
# Exp 6 — positioning against RAF (cited numbers)

RAF_TABLE1 = {  # Park, Jeong, Yoon, arXiv 2607.04587, Table 1 — K-Radar, Sedan, IoU 0.5. Transcribed from the PDF.
    "cols": ["Normal", "Overcast", "Fog", "Rain", "Sleet", "Lightsnow", "Heavysnow", "Total"],
    "L4DR (Baseline, L+R)": {"BEV": [76.8, 88.6, 89.7, 78.2, 59.3, 80.9, 53.8, 77.5], "3D": [53.0, 64.1, 73.2, 53.8, 46.2, 52.4, 37.0, 53.5]},
    "L4DR (Joint, C+L+R)": {"BEV": [77.0, 80.8, 89.4, 78.4, 59.9, 81.1, 53.2, 77.8], "3D": [54.2, 63.2, 68.6, 53.5, 46.4, 54.1, 32.2, 53.9]},
    "L4DR (Frozen, C+L+R)": {"BEV": [79.8, 83.5, 92.2, 79.1, 59.2, 85.7, 58.4, 78.6], "3D": [58.2, 53.6, 71.5, 55.9, 36.6, 46.6, 35.3, 54.2]},
    "L4DR (RAF, C+L+R)": {"BEV": [82.0, 88.2, 93.9, 82.6, 63.0, 88.5, 60.4, 82.0], "3D": [59.4, 67.6, 79.3, 53.3, 45.9, 56.3, 37.8, 57.4]},
    "3D-LRF (Baseline, L+R)": {"BEV": [62.1, 88.5, 84.8, 69.9, 37.7, 84.6, 36.0, 64.5], "3D": [32.9, 49.1, 40.4, 34.5, 25.1, 53.1, 14.8, 35.8]},
    "3D-LRF (Joint, C+L+R)": {"BEV": [70.2, 81.9, 75.8, 74.5, 18.4, 74.2, 40.9, 65.7], "3D": [41.3, 61.3, 49.0, 39.2, 25.5, 59.0, 27.7, 39.3]},
    "3D-LRF (Frozen, C+L+R)": {"BEV": [67.2, 87.8, 81.6, 78.3, 42.4, 80.1, 48.6, 68.3], "3D": [38.2, 57.2, 44.7, 36.5, 25.0, 55.6, 39.5, 39.1]},
    "3D-LRF (RAF, C+L+R)": {"BEV": [70.8, 88.0, 82.3, 74.3, 43.3, 84.3, 49.6, 71.0], "3D": [42.9, 63.8, 46.5, 44.1, 32.6, 63.9, 35.6, 43.2]},
}
RAF_TABLE2 = {  # Table 2 — visibility splits Clean / Mixed / Noisy
    "L4DR Baseline (L+R)": [77.0, 78.4, 59.0], "L4DR Joint": [81.5, 77.5, 55.6], "L4DR Frozen": [80.2, 78.1, 55.6], "L4DR RAF": [80.4, 82.1, 59.0],
    "3D-LRF Baseline (L+R)": [68.8, 73.0, 38.8], "3D-LRF Joint": [69.8, 68.8, 25.4], "3D-LRF Frozen": [70.4, 65.5, 30.8], "3D-LRF RAF": [72.2, 74.5, 38.4],
}


def exp6(ctx):
    out = ctx.out / "exp6_raf"
    out.mkdir(parents=True, exist_ok=True)
    t1 = RAF_TABLE1
    md = "# Exp 6 — Positioning against RAF (cited numbers; different dataset, not a head-to-head)\n\n"
    md += ("> Numbers transcribed from Park, Jeong & Yoon, *RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR*, "
           "arXiv 2607.04587 (Tables 1–2), K-Radar, Sedan class, IoU 0.5. **Verify every digit against the PDF before "
           "citing** — they were read from the rendered page. They are on K-Radar with 4D radar and are *not* comparable "
           "to our Boreas numbers; the table documents what RAF reports, not a ranking.\n\n")
    md += "## RAF Table 1 — AP_BEV\n\n" + _md_table(["method"] + t1["cols"], [[k] + [f"{x:.1f}" for x in v["BEV"]] for k, v in t1.items() if k != "cols"])
    md += "\n\n## RAF Table 2 — AP_BEV by camera visibility (Clean / Mixed / Noisy)\n\n" + \
          _md_table(["method", "Clean", "Mixed", "Noisy"], [[k] + [f"{x:.1f}" for x in v] for k, v in RAF_TABLE2.items()])
    md += ("\n\n**Harm evidence inside RAF's own numbers.** 3D-LRF on *Noisy*: L+R baseline 38.8 → C+L+R Joint 25.4 "
           "(−13.4 AP_BEV). Sleet: 37.7 → 18.4. Adding the camera without reliability handling hurts — the aggregate "
           "form of our H1. RAF recovers to 38.4 (Noisy), i.e. it *neutralises* camera harm; it does not report "
           "per-object harm rates or predict harm.\n\n")
    md += "## Property comparison\n\n" + _md_table(
        ["property", "RAF (2026)", "this work"],
        [["fusion level", "BEV feature fusion (camera branch + BEV fusion encoder + head trained)", "output-level late fusion over frozen detectors"],
         ["what is trained", "camera backbone, projectors, reliability CNN, fusion encoder, detection head (Swin-T; RTX 3090)", "~10⁴-parameter router on cached detections (CPU, minutes)"],
         ["retrofit to an existing stack", "requires re-training the fusion network per backbone", "any detectors that emit boxes + scores"],
         ["which modalities are arbitrated", "camera only (LiDAR–RADAR assumed reliable reference)", "camera, LiDAR and radar symmetrically"],
         ["granularity", "per image pixel", "per object (plus frame context)"],
         ["supervision of reliability", "image-level Clean/Mixed/Noisy labels → cross-modal similarity pseudo-labels", "counterfactual harm labels from GT (no weather labels needed)"],
         ["negative evidence ('should have seen it')", "no", "yes — absence × field-of-view term"],
         ["formal guarantee on output", "none", "split-conformal localisation discs + conformal risk control of FNR"],
         ["datasets", "K-Radar, VoD", "Boreas (+ graded simulation), CADC"]])
    (out / "exp6_raf.md").write_text(md, encoding="utf-8")
    return {"status": "written"}


# ======================================================================================================
# Exp 7 — per-modality dropout stress test

def exp7(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp7_dropout"
    out.mkdir(parents=True, exist_ok=True)
    arms = {"fixed_clear": ctx.status_quo(), **{a: ctx.arm(a) for a in ("fixed", "frame_cls", "object")}}
    res = {}
    tags = ["clear0"] + [t for t in ("fog3", "snow3", "night2") if t in ctx.tags]
    for a, arb in arms.items():
        res[a] = {}
        for tag in tags:
            tt = ctx.test_table(tag)
            n = len(tt["frames"])
            row = {"none": eval_map(tt, arb.predict(tt), cfg)[0]}
            for m, name in enumerate(MODALITIES):
                d = np.zeros((n, 3), bool)
                d[:, m] = True
                row[f"no_{name}"] = eval_map(tt, arb.predict(tt, d), cfg)[0]
            res[a][tag] = row
    save_json(res, out / "exp7_dropout.json")
    md = "# Exp 7 — Whole-sensor failure at test time (Car mAP, test scenes)\n\n"
    for tag in tags:
        md += f"\n## {tag}\n\n" + _md_table(["arm", "all sensors", "camera failed", "LiDAR failed", "radar failed"],
                                             [[plots.ARM_LABEL[a], fmt(res[a][tag]["none"]), fmt(res[a][tag]["no_camera"]),
                                               fmt(res[a][tag]["no_lidar"]), fmt(res[a][tag]["no_radar"])] for a in arms])
    (out / "exp7_dropout.md").write_text(md, encoding="utf-8")
    groups = ["none", "no_camera", "no_lidar", "no_radar"]
    vals = {a: {g: res[a]["snow3" if "snow3" in tags else "clear0"][g] for g in groups} for a in arms}
    plots.grouped_bars(groups, list(arms), vals, plots.ARM_COLOR, plots.ARM_LABEL, "Car mAP",
                       f"Sensor failure under {'snow3' if 'snow3' in tags else 'clear0'} (test)", out / "dropout.png")
    return res


# ======================================================================================================
# Exp 8 — qualitative cases incl. a failure

def exp8(ctx):
    cfg, out = ctx.cfg, ctx.out / "exp8_qualitative"
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("win_*")) + list(out.glob("failure_*")) + list(out.glob("qualitative_cases.*")):
        stale.unlink()  # never keep figures from an earlier run
    from .evaluate import decisions
    import matplotlib.pyplot as plt
    sq, obj = ctx.status_quo(), ctx.arm("object")
    tt = ctx.test_table(tags=ctx.adverse_tags)
    ps, po = sq.predict(tt), obj.predict(tt)
    cs, co = decisions(tt, ps, sq.tau, cfg), decisions(tt, po, obj.tau, cfg)
    c = tt["cand"]
    car = (c["cls"] == 0) & ~c["ignore"] & (c["present"].sum(1) >= 2)
    wins = np.nonzero(car & ~cs & co)[0]
    fails = np.nonzero(car & cs & ~co)[0]
    cases = []
    if len(wins):
        cases.append(("win", wins[np.argmax(np.abs(po["score"][wins] - ps["score"][wins]))]))
    if len(fails):
        cases.append(("failure", fails[np.argmax(np.abs(po["score"][fails] - ps["score"][fails]))]))
    summary = {"n_wins": int(len(wins)), "n_failures": int(len(fails)), "cases": []}
    # ONE figure with both cases side by side (panels a, b) and a single shared legend: symmetric by construction, so
    # it is centred on the page as one object (two separate images each carried their own uneven margins).
    fig, axes = plt.subplots(1, len(cases), figsize=(plots.FIG_W, 3.5), constrained_layout=True, squeeze=False)
    for ax, (kind, i) in zip(axes[0], cases):
        f = c["frame"][i]
        fr = tt["frames"][f]
        same = c["frame"] == f
        g = tt["gts"][f]
        ax.scatter(g["xy"][g["cls"] == 0, 1], g["xy"][g["cls"] == 0, 0], marker="s", s=40, facecolors="none",
                   edgecolors=plots.INK, label="GT car")
        for m, name in enumerate(MODALITIES):
            pts = c["xy"][same & c["present"][:, m], m]
            ax.scatter(pts[:, 1], pts[:, 0], s=10, color=plots.MOD_COLOR[name], label=f"{name}", alpha=0.75)
        for m, name in enumerate(MODALITIES):
            if c["present"][i, m]:
                ax.plot([c["xy"][i, m, 1], po["xy"][i, 1]], [c["xy"][i, m, 0], po["xy"][i, 0]], color=plots.MOD_COLOR[name], lw=1)
        ax.scatter([ps["xy"][i, 1]], [ps["xy"][i, 0]], marker="x", s=80, color=plots.ARM_COLOR["fixed_clear"], label="status-quo fused")
        ax.scatter([po["xy"][i, 1]], [po["xy"][i, 0]], marker="+", s=120, color=plots.ARM_COLOR["object"], label="per-object fused")
        cx, cy = po["xy"][i]
        ax.set_xlim(cy + 15, cy - 15)
        ax.set_ylim(cx - 15, cx + 15)
        ax.set_box_aspect(1)
        plots.style(ax, "x forward (m)", "y left (m)")
        gains = {n: round(float(po["gain"][i, m]), 2) for m, n in enumerate(MODALITIES) if c["present"][i, m]}
        shares = {n: round(float(po["range_share"][i, m]), 2) for m, n in enumerate(MODALITIES) if c["present"][i, m]}
        ax.set_title(f"({'ab'[len(summary['cases'])]}) {kind}: {fr['condition']}{fr['severity']}, score "
                     f"{ps['score'][i]:.2f} \u2192 {po['score'][i]:.2f}, GT-backed={bool(c['y'][i])}",
                     fontsize=7.5, loc="center", color=plots.INK)
        summary["cases"].append({"kind": kind, "frame": fr, "evidence_gain": gains, "range_share": shares,
                                 "score_status_quo": float(ps["score"][i]), "score_object": float(po["score"][i]),
                                 "gt_backed": bool(c["y"][i]), "range_m": float(np.hypot(*po["xy"][i])),
                                 "harm_prob": po["harm_prob"][i].round(3).tolist(), "figure": "qualitative_cases.png"})
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=len(l), loc="outside lower center", handlelength=1.0, columnspacing=1.4)
    plots.save(fig, out / "qualitative_cases.png")
    save_json(summary, out / "exp8_qualitative.json")
    md = f"# Exp 8 — Qualitative cases\n\nAdverse test candidates with ≥ 2 sensors: per-object arm fixes the status quo in " \
         f"{summary['n_wins']} cases and breaks it in {summary['n_failures']} cases.\n\n"
    for cse in summary["cases"]:
        md += f"## {cse['kind']}\n\n![{cse['kind']}]({cse['figure']})\n\n```\n{cse}\n```\n\n"
    (out / "exp8_qualitative.md").write_text(md, encoding="utf-8")
    return summary


def write_report(ctx, results):
    lines = ["# Paper 02 — Results report", "", f"Source hash `{source_tree_hash()}`; generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
             f"Frames: {len(ctx.T['frames'])} (all conditions); tags: {', '.join(ctx.tags)}.", ""]
    for name in ("exp1_harm/exp1_harm.md", "exp2_routing/exp2_routing.md", "exp3_ablation/exp3_ablation.md",
                 "exp4_conformal/exp4_conformal.md", "exp5_cadc/exp5_cadc.md", "exp7_dropout/exp7_dropout.md",
                 "exp8_qualitative/exp8_qualitative.md", "exp6_raf/exp6_raf.md"):
        p = ctx.out / name
        if p.exists():
            lines += [f"\n---\n<!-- {name} -->", p.read_text(encoding="utf-8")]
    (ctx.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


EXPS = {"1": exp1, "2": exp2, "3": exp3, "4": exp4, "5": exp5, "6": exp6, "7": exp7, "8": exp8}


def multiseed(cfg, seeds, log):
    """Repeat Exp 2 over head-training seeds (data splits fixed by the cache); mean ± sd of Car mAP."""
    base = resolve(cfg, "results_root")
    maps = {}
    for s in seeds:
        c = dict(cfg)
        c["seed"] = s
        c["paths"] = dict(cfg["paths"], results_root=str(base / f"seed{s}"))
        seed_everything(s)
        res = exp2(Ctx(c))
        for arm, per in res["map"].items():
            for k, v in per.items():
                if not k.endswith("_ci"):
                    maps.setdefault(arm, {}).setdefault(k, []).append(v)
    out = {arm: {k: [float(np.mean(v)), float(np.std(v))] for k, v in per.items()} for arm, per in maps.items()}
    save_json({"seeds": seeds, "map_mean_sd": out}, base / "multiseed_exp2.json")
    cols = [k for k in next(iter(out.values())) if k in ("clear0", "SEEN_ADVERSE", "HELDOUT", "ADVERSE_POOLED")]
    md = f"# Exp 2 over seeds {seeds} — Car mAP mean ± sd\n\n" + _md_table(
        ["arm"] + cols, [[plots.ARM_LABEL[a]] + [f"{out[a][k][0]:.3f} ± {out[a][k][1]:.3f}" for k in cols]
                         for a in ["fixed_clear"] + ARMS if a in out])
    (base / "multiseed_exp2.md").write_text(md, encoding="utf-8")
    log.info(f"multiseed written for seeds {seeds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--exp", nargs="+", default=["all"])
    ap.add_argument("--retrain", action="store_true", help="delete cached models first")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="multi-seed Exp 2 (e.g. 0 1 2 3 4)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seeds:
        multiseed(cfg, args.seeds, get_logger("exp"))
        return
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    if args.retrain:
        for p in ctx.model_dir.glob("*"):
            p.unlink()
    todo = list(EXPS) if args.exp == ["all"] else args.exp
    results = {}
    for e in todo:
        t0 = time.time()
        results[e] = EXPS[e](ctx)
        ctx.log.info(f"exp{e} done in {time.time() - t0:.0f}s")
    write_report(ctx, results)


if __name__ == "__main__":
    main()
