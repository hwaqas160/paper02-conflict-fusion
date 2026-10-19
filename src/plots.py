"""Figure style for the paper: a clean, modern publication theme, exported at 600 dpi.

White ground, closed-box axes (all four spines, matching the reference figures in paper/sample figures/),
recessive horizontal grid, sans-serif type matching the manuscript body, legend above the plot, no in-figure
title (the LaTeX caption carries it). Colours are the dataviz reference palette in its fixed order; an arm keeps
its colour -- and, now, its hatch pattern and line marker -- in every figure regardless of how many arms a
figure shows, so redundant (not just colour) encoding survives grayscale printing and colour-blind readers, the
same reasoning behind the hatching in the reference figures. Every figure is sized to the printed double-column
width (FIG_W inches) so that type is the size it appears on the page, and is saved as a 600-dpi PNG plus a
vector PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 600
FIG_W = 7.0          # inches: full text width of the two-column class (figure* at width=\linewidth)
SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#111827", "#4b5563", "#9ca3af", "#e5e7eb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Paired with SERIES by position (not by name), so any subset of arms/modalities picks up the same hatch/marker
# wherever that colour appears, the same "an arm keeps its identity everywhere" rule the colours already follow.
HATCH = [None, "//", "..", "xx", "\\\\", "oo", "++", "--"]
MARKER = ["o", "s", "^", "v", "D", "P", "X", "*"]
COLOR_HATCH = dict(zip(SERIES, HATCH))
COLOR_MARKER = dict(zip(SERIES, MARKER))
ARM_ORDER = ["fixed_clear", "fixed", "confidence", "frame_cls", "frame_oracle", "frame_feats", "object"]
ARM_LABEL = {"fixed_clear": "Fixed (clear-fit, status quo)", "fixed": "Fixed weights", "confidence": "Confidence-only",
             "frame_cls": "Per-frame weather classifier", "frame_oracle": "Per-frame oracle weather",
             "frame_feats": "Per-frame features", "object": "Per-object (ours)"}
ARM_COLOR = {a: SERIES[i] for i, a in enumerate(ARM_ORDER)}
MOD_COLOR = {"camera": SERIES[0], "lidar": SERIES[1], "radar": SERIES[2]}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "text.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150, "savefig.dpi": DPI,
    "axes.spines.top": True, "axes.spines.right": True, "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax, ylabel=None, xlabel=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)


def save(fig, path):
    """600-dpi PNG (what the manuscript includes) plus a vector PDF next to it."""
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(groups, series, values, colors, labels, ylabel, title, path, errors=None, ref_line=None):
    """values[series][group] -> bar; errors same shape as (lo, hi). `title` is kept for API compatibility and
    is intentionally not drawn: the caption in the manuscript is the title."""
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9), constrained_layout=True)
    n = len(series)
    width = 0.82 / n
    x = np.arange(len(groups))
    # Hatching is a readability aid (redundant colour+pattern encoding) up to a handful of series; past that,
    # on bars this narrow, it adds visual noise faster than it adds distinguishing power, so colour alone carries
    # the encoding and only the black edge (still legible, still grayscale-safe as a boundary) remains.
    use_hatch = n <= 4
    for i, s in enumerate(series):
        v = np.array([values[s].get(g, np.nan) for g in groups], dtype=float)
        pos = x - 0.41 + width * (i + 0.5)
        ax.bar(pos, v, width * 0.9, color=colors[s], label=labels[s], edgecolor=INK, linewidth=0.7,
               hatch=COLOR_HATCH.get(colors[s]) if use_hatch else None)
        if errors is not None:
            lo = np.array([errors[s].get(g, (np.nan, np.nan))[0] for g in groups], dtype=float)
            hi = np.array([errors[s].get(g, (np.nan, np.nan))[1] for g in groups], dtype=float)
            ax.vlines(pos, lo, hi, color=INK, linewidth=0.6)
    if ref_line is not None:
        ax.axhline(ref_line[0], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(len(groups) - 0.5, ref_line[0], ref_line[1], color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    style(ax, ylabel)
    ax.legend(frameon=False, ncol=min(n, 4), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.6)
    save(fig, path)


def forest(rows, path, xlabel="Car mAP difference", groups=None):
    """rows: list of (label, diff, lo, hi, colour_key) top-to-bottom in the order given.
    groups: optional list of (first_row_index, group_title); a gap opens before that row and the title sits in it,
    clear of both the group above and the group's own first row."""
    n = len(rows)
    GAP = 0.9
    starts = {idx for idx, _ in groups} if groups else set()
    y_top = [GAP if 0 in starts else 0.0]
    for i in range(1, n):
        y_top.append(y_top[-1] + 1.0 + (GAP if i in starts else 0.0))
    total = y_top[-1]
    top_margin = GAP if 0 in starts else 0.3
    ys = [total - y for y in y_top]  # first row at the top
    fig, ax = plt.subplots(figsize=(FIG_W, 0.34 * (total + top_margin + 1) + 1.1), constrained_layout=True)
    for y, (label, diff, lo, hi, ckey) in zip(ys, rows):
        c = ARM_COLOR.get(ckey, SERIES[0]) if ckey else INK2
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.3, solid_capstyle="round")
        ax.plot([diff], [y], marker="o", markersize=4.5, color=c, markeredgecolor="white", markeredgewidth=0.5)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, total + top_margin)
    style(ax, xlabel=xlabel, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    if groups:
        x0 = ax.get_xlim()[0]
        for idx, title in groups:
            y = ys[idx] + (GAP + 1.0) / 2 - 0.05
            ax.text(x0, y, title, fontsize=7.5, fontweight="bold", color=INK, ha="left", va="center")
    save(fig, path)


def lines(xs, series, values, colors, labels, ylabel, xlabel, title, path, ref_line=None):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9), constrained_layout=True)
    for s in series:
        v = [values[s].get(x, np.nan) for x in xs]
        ax.plot(range(len(xs)), v, color=colors[s], linewidth=1.6, marker=COLOR_MARKER.get(colors[s], "o"),
                markersize=4.5, label=labels[s], markeredgecolor="white", markeredgewidth=0.6)
    if ref_line is not None:
        ax.axhline(ref_line[0], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(len(xs) - 0.6, ref_line[0], ref_line[1], color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    style(ax, ylabel, xlabel)
    ax.legend(frameon=False, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
              columnspacing=1.6)
    save(fig, path)
