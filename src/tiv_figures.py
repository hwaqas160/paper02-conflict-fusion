"""Figures for the IEEE T-IV version of the paper, drawn directly from the saved result files (same theme as
src/plots.py: colour plus marker plus line style, so they survive greyscale printing and colour-vision deficiency).

    python -m src.tiv_figures            # writes PDF + PNG into submission/IEEE_TIV/figures
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plots
from .plots import GRID, INK2, MARKER, MUTED, SERIES, style

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "submission" / "IEEE_TIV" / "figures"
W1 = 3.5   # one IEEE column, inches


def load(*p):
    return json.loads(RES.joinpath(*p).read_text(encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def summary():
    """Per-object minus comparison arm, mean average precision, with 95 % scene-cluster intervals, in every setting."""
    s1 = load("supplementary", "s1_scene_bootstrap.json")["ADVERSE_POOLED"]["object_vs_frame_cls"]
    ts = load("truckscenes", "r2", "r2a_truckscenes.json")["A1"]
    cam = load("r4", "stronger_cam", "r4", "r4_4_stronger_camera.json")["primary_scene_cluster"]
    r41 = load("r4", "r4_1_decision_unit.json")["primary_scene_cluster"]
    r42 = load("truckscenes", "r4", "r4_2_truckscenes_decision_unit.json")["primary_scene_cluster"]
    e8c = load("ts_detector", "r4", "e8c_indomain_router.json")["primary_object_vs_frame_cls"]
    cad = load("r2", "r2d_cadc.json")["D1_raw"]["object_vs_frame_cls"]
    e13 = load("truckscenes", "r4", "e13_sim2real.json")["object_vs_frame_cls"]
    rows = [
        ("Simulated weather (Boreas)", s1, 0),
        ("Real weather, off-the-shelf LiDAR", ts, 0),
        ("Real weather, stronger camera", cam, 0),
        ("Real weather, in-domain LiDAR", e8c, 1),
        ("CADC real snow, no radar", cad, 1),
        ("Simulation-trained, zero-shot on real", e13, 1),
        ("Equal-information control, Boreas*", r41, 2),
        ("Equal-information control, real*", r42, 2),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    style(ax, xlabel="Difference in mean average precision, per-object minus comparison", grid_axis="x")
    ypos = np.arange(len(rows))[::-1]
    kind_color = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2]}
    kind_marker = {0: "o", 1: "s", 2: "^"}
    for y, (lab, c, k) in zip(ypos, rows):
        lo, hi = c["ci95"]
        ax.plot([lo, hi], [y, y], color=kind_color[k], lw=1.6, solid_capstyle="butt")
        ax.plot([c["diff"]], [y], marker=kind_marker[k], ms=5.6, color=kind_color[k], mec="white", mew=0.5, zorder=3)
        ax.text(0.2, y, ("0.000" if abs(c["diff"]) < 5e-4 else f"{c['diff']:+.3f}"), va="center", ha="right", fontsize=7.0, color=INK2)
    ax.axvline(0, color="#111827", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_xlim(-0.21, 0.205)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.xaxis.label.set_size(7.4)
    ax.text(-0.205, -0.55, "favours per-frame", fontsize=6.6, color=MUTED, ha="left", va="center")
    ax.text(0.197, -0.55, "favours per-object", fontsize=6.6, color=MUTED, ha="right", va="center")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SERIES[0], marker="o", ms=4.5, lw=1.4, label="Advantage holds"),
               Line2D([0], [0], color=SERIES[1], marker="s", ms=4.5, lw=1.4, label="Absent or reversed"),
               Line2D([0], [0], color=SERIES[2], marker="^", ms=4.5, lw=1.4, label="Control*: versus equal-information arm")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.6, frameon=True, framealpha=0.95, edgecolor=GRID, borderpad=0.3, labelspacing=0.3, bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(pad=0.4)
    save(fig, "fig_summary")


def by_condition():
    e2 = load("exp2_routing", "exp2_routing.json")["map"]
    tags = ["clear0", "fog1", "fog2", "fog3", "rain1", "rain2", "rain3", "snow1", "snow2", "snow3", "night2"]
    labels = ["clear", "fog1", "fog2", "fog3", "rain1", "rain2", "rain3", "snow1", "snow2", "snow3", "night"]
    arms = [("fixed_clear", "Fixed (status quo)", SERIES[3], "v", ":"),
            ("frame_cls", "Per-frame, predicted", SERIES[1], "s", "--"),
            ("frame_oracle", "Per-frame, oracle", SERIES[6], "^", "-."),
            ("object", "Per-object (ours)", SERIES[0], "o", "-")]
    fig, ax = plt.subplots(figsize=(W1, 2.25))
    style(ax, ylabel="Car mAP", grid_axis="y")
    x = np.arange(len(tags))
    for key, lab, col, mk, ls in arms:
        ax.plot(x, [e2[key][t] for t in tags], color=col, marker=mk, ls=ls, lw=1.5 if key == "object" else 1.1,
                ms=4.2 if key == "object" else 3.4, label=lab, mec="white", mew=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.yaxis.label.set_size(6.8)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=0.9, edgecolor=GRID, handlelength=2.6, borderpad=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_cond")


def coverage():
    e4 = load("exp4_conformal", "exp4_conformal.json")["object"]["loc"]["0.1"]
    tags = ["clear0", "fog1", "fog2", "fog3", "rain1", "rain2", "rain3", "snow1", "snow2", "snow3", "night2"]
    labels = ["clear", "fog1", "fog2", "fog3", "rain1", "rain2", "rain3", "snow1", "snow2", "snow3", "night"]
    series = [("vanilla", "Vanilla score", SERIES[0], "o", "-"),
              ("conflict", "Conflict-inflated", SERIES[1], "s", "--"),
              ("conflict_mondrian", "Conflict + range-Mondrian", SERIES[2], "^", "-.")]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    style(ax, ylabel="Empirical coverage", grid_axis="y")
    x = np.arange(len(tags))
    for key, lab, col, mk, ls in series:
        ax.plot(x, [e4[t][key]["coverage"] for t in tags], color=col, marker=mk, ls=ls, lw=1.2, ms=3.6, label=lab,
                mec="white", mew=0.4)
    ax.axhline(0.90, color="#111827", lw=0.9, ls=(0, (1, 1.5)))
    ax.text(len(tags) - 0.6, 0.903, "nominal 0.90", fontsize=6.0, ha="right", va="bottom", color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.yaxis.label.set_size(6.8)
    ax.set_ylim(0.805, 0.97)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=0.9, edgecolor=GRID, handlelength=2.6, borderpad=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_cov")


def share():
    """How much of the per-object gain needs degradation probes, by dataset."""
    bo = load("r4", "e10_minimal_boreas.json")
    ts = load("truckscenes", "r4", "e10_minimal_truckscenes.json")
    arms = [("geom", "Range, azimuth"), ("geom_conf", "+ detector scores"), ("geom_clocs", "+ conflict features")]
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    style(ax, ylabel="Gain over per-frame classifier (mAP)", grid_axis="y")
    x = np.arange(len(arms) + 1)
    w = 0.38
    for off, res, lab, col, hatch in ((-w / 2, bo, "Simulated (Boreas)", SERIES[0], None), (w / 2, ts, "Real (TruckScenes)", SERIES[1], "//")):
        vals = [res["vs_frame_cls"][a]["diff"] for a, _ in arms] + [res["object_vs_frame_cls"]["diff"]]
        ax.bar(x + off, vals, w, color=col, hatch=hatch, edgecolor="white", linewidth=0.5, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=5.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms] + ["Full router\n(adds probes)"], fontsize=6.0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.yaxis.label.set_size(6.4)
    ax.set_ylim(0, 0.098)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, framealpha=0.9, edgecolor=GRID, borderpad=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_share")


if __name__ == "__main__":
    summary()
    by_condition()
    coverage()
    share()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
