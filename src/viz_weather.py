"""Visual sanity check of the weather simulator: one frame, every condition, camera + LiDAR BEV + radar.

    python -m src.viz_weather --frame-index 100 --out results/figures/weather_panel.png

Sized to the printed text width and exported at 600 dpi (PNG) plus a vector-free PDF copy, in the paper theme.
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import plots
from . import weather as W
from .boreas_data import IMG_H, IMG_W, build_index, load_calib, read_camera, read_lidar, read_radar
from .cache_detections import CAM_SCALE, weather_params
from .common import iter_conditions, load_config, resolve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-index", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    wcfg = weather_params(cfg)
    frames = build_index(resolve(cfg, "boreas_root"), require_data=True)
    f = frames[min(args.frame_index, len(frames) - 1)]
    calib = load_calib(f.root)
    P = calib["P"].copy()
    P[:2, :3] *= CAM_SCALE
    img = cv2.resize(read_camera(f), (int(IMG_W * CAM_SCALE), int(IMG_H * CAM_SCALE)), interpolation=cv2.INTER_AREA)
    pts = read_lidar(f)
    rad = read_radar(f)
    conds = list(iter_conditions(cfg))
    fig, axes = plt.subplots(3, len(conds), figsize=(plots.FIG_W, 3.9), gridspec_kw={"height_ratios": [1.0, 1.7, 1.3]},
                             constrained_layout=True)
    for j, (c, s) in enumerate(conds):
        im = W.degrade_camera(img, c, s, wcfg, W.rng_for(f.frame_id, c, s, "camera"), pts, f.T_cam_lidar(), P)
        pl = W.degrade_lidar(pts, c, s, wcfg, W.rng_for(f.frame_id, c, s, "lidar"))
        rd = W.degrade_radar(rad, c, s, wcfg, W.rng_for(f.frame_id, c, s, "radar"))
        axes[0, j].imshow(im)
        axes[0, j].set_title(f"{c}{s}", fontsize=6.5, pad=2)
        m = (np.abs(pl[:, 0]) < 40) & (np.abs(pl[:, 1]) < 40)
        axes[1, j].scatter(pl[m, 1], pl[m, 0], s=0.03, c=np.clip(pl[m, 3], 0, 60), cmap="viridis", rasterized=True)
        axes[1, j].set_title(f"{len(pl) / 1e3:.0f}k pts", fontsize=6, pad=2, color=plots.INK2)
        axes[1, j].set_aspect("equal")
        axes[2, j].imshow(rd["polar"][:, :1300], aspect="auto", cmap="gray", vmin=0, vmax=0.8)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
                sp.set_color(plots.MUTED)
    for i, lab in enumerate(("camera", "LiDAR (BEV)", "radar (polar)")):
        axes[i, 0].set_ylabel(lab, fontsize=6.5, color=plots.INK2)
    out = args.out or resolve(cfg, "results_root") / "figures" / "weather_panel.png"
    out = resolve({"paths": {"o": str(out)}}, "o")
    out.parent.mkdir(parents=True, exist_ok=True)
    plots.save(fig, out)
    print(out)


if __name__ == "__main__":
    main()
