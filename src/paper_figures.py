"""Re-render every figure of the paper from the saved result files at 600 dpi in the shared theme (src/plots.py).

    python -m src.paper_figures            # the four chart figures (seconds, no models needed)
    python -m src.paper_figures --all      # + qualitative cases (loads cached models) + weather panel (raw Boreas)

The chart figures are drawn from exactly the numbers in results/exp*/*.json, so a figure can never disagree
with the tables. Outputs replace the files beside each JSON; paper/latex/make_paper_data.py stages them.
"""
import argparse
import json

import numpy as np

from . import plots
from .common import MODALITIES, load_config, resolve

ARMS = ["fixed", "confidence", "frame_cls", "frame_oracle", "frame_feats", "object"]


def _load(root, *parts):
    return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))


def _tags(cfg, present):
    out = []
    for c, sevs in cfg["weather"]["conditions"].items():
        out += [f"{c}{s}" for s in sevs]
    return [t for t in out if t in present]


def charts(cfg):
    R = resolve(cfg, "results_root")
    e1 = _load(R, "exp1_harm", "exp1_harm.json")
    groups = _tags(cfg, e1["per_tag"])
    vals = {m: {t: 100 * e1["per_tag"][t][m]["harm_rate"] for t in groups} for m in MODALITIES}
    errs = {m: {t: tuple(100 * np.array(e1["per_tag"][t][m]["harm_ci"])) for t in groups} for m in MODALITIES}
    plots.grouped_bars(groups, list(MODALITIES), vals, plots.MOD_COLOR, {m: m for m in MODALITIES},
                       "decision-flip harm rate (%)", "", R / "exp1_harm" / "harm_by_condition.png", errs,
                       ref_line=(2.0, "2 % gate"))
    e2 = _load(R, "exp2_routing", "exp2_routing.json")
    tags = _tags(cfg, e2["map"]["object"])
    plots.grouped_bars(tags, ["fixed_clear"] + ARMS, e2["map"], plots.ARM_COLOR, plots.ARM_LABEL, "Car mAP", "",
                       R / "exp2_routing" / "map_by_condition.png")
    e4 = _load(R, "exp4_conformal", "exp4_conformal.json")
    a_res = e4["object"]["loc"]["0.1"]
    vals = {k: {t: a_res[t][k]["coverage"] for t in _tags(cfg, a_res)} for k in ("vanilla", "conflict", "conflict_mondrian")}
    colors = {"vanilla": plots.SERIES[0], "conflict": plots.SERIES[1], "conflict_mondrian": plots.SERIES[2]}
    plots.lines(_tags(cfg, a_res), list(vals), vals, colors,
                {"vanilla": "vanilla score", "conflict": "conflict-inflated", "conflict_mondrian": "conflict + range-Mondrian"},
                "empirical coverage", "test condition", "", R / "exp4_conformal" / "coverage_transfer.png",
                ref_line=(0.9, "nominal 0.90"))
    e7 = _load(R, "exp7_dropout", "exp7_dropout.json")
    arms = list(e7)
    tag = "snow3" if "snow3" in e7[arms[0]] else "clear0"
    groups = ["none", "no_camera", "no_lidar", "no_radar"]
    vals = {a: {g: e7[a][tag][g] for g in groups} for a in arms}
    plots.grouped_bars(groups, arms, vals, plots.ARM_COLOR, plots.ARM_LABEL, "Car mAP", "",
                       R / "exp7_dropout" / "dropout.png")
    forest(R)


def forest(R):
    """Forest plot of every headline pre-registered effect size, all pulled straight from the same JSON files
    make_paper_data.py reads, so a plotted point can never disagree with the number in the text."""
    s1 = _load(R, "supplementary", "s1_scene_bootstrap.json")["ADVERSE_POOLED"]["object_vs_frame_cls"]
    a = _load(R / "truckscenes" / "r2", "r2a_truckscenes.json")
    b = _load(R, "r2", "r2b_baselines.json")
    r3ts = _load(R / "truckscenes" / "r3", "r3_final_truckscenes.json")
    r3bo = _load(R, "r3", "r3_final_boreas.json")
    baseline_label = {"clocs": "CLOCs", "proben": "ProbEn", "dempster_shafer": "Dempster–Shafer",
                      "moe_frame": "Mixture of experts", "set_transformer": "Set-transformer"}
    rows, groups = [], []
    groups.append((len(rows), "H2 — per-object vs. per-frame (Boreas, scene-cluster)"))
    rows.append(("per-frame classifier", s1["diff"], s1["ci95"][0], s1["ci95"][1], "object"))
    groups.append((len(rows), "R2-A — TruckScenes real weather (5-fold CV)"))
    rows.append(("per-frame classifier", a["A1"]["diff"], a["A1"]["ci95"][0], a["A1"]["ci95"][1], "object"))
    rows.append(("radar removed", a["A3"]["diff"], a["A3"]["ci95"][0], a["A3"]["ci95"][1], "object"))
    groups.append((len(rows), "R2-B — vs. published baselines (Boreas)"))
    for k in ("clocs", "proben", "dempster_shafer", "moe_frame", "set_transformer"):
        v = b["vs_ours"][k]
        rows.append((baseline_label[k], v["diff"], v["ci95"][0], v["ci95"][1], "object"))
    groups.append((len(rows), "R3 — closing the gap (held-out scenes)"))
    v = r3ts["vs"]["set_transformer"]
    rows.append(("V2 vs. set-transformer (TruckScenes)", v["diff"], v["ci95"][0], v["ci95"][1], "frame_cls"))
    v = r3bo["vs"]["original"]
    rows.append(("V2 vs. original router (Boreas)", v["diff"], v["ci95"][0], v["ci95"][1], "frame_cls"))
    plots.forest(rows, R / "supplementary" / "forest_headline.png",
                xlabel="Car mAP difference (positive favours the router, or V2 for the R3 rows)", groups=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    charts(cfg)
    print("charts re-rendered")
    if args.all:
        from . import viz_weather
        from .run_experiments import Ctx, exp8
        exp8(Ctx(cfg))
        viz_weather.main([])
        print("qualitative cases and weather panel re-rendered")


if __name__ == "__main__":
    main()
