"""E21 stages after camera training: candidate tables from cache_cam and the five-fold router comparison (E8-C protocol)."""
import sys

from .common import load_config, resolve

cfg = load_config()
base = resolve(cfg, "results_root") / "ts_detector"
if sys.argv[1] == "candidates":
    from .candidates import build_table, save_table
    dst = base / "cache_cam" / "truckscenes"
    for tag in sorted(p.name for p in dst.iterdir() if p.is_dir()):
        tbl = build_table(dst / tag, cfg)
        if tbl is not None:
            save_table(tbl, base / "candidates_cam" / "truckscenes" / f"{tag}.npz")
            print(tag, len(tbl["frames"]), flush=True)
else:
    from .e8c_router import run
    r = run(n_boot=10000, n_boot2=1000, cand_name="candidates_cam", out_name="e21_camera_retrained_router.json", router_dir="router_cam")
    c = r["primary_object_vs_frame_cls"]
    print({k: round(v, 3) for k, v in r["oof_map"].items()})
    print("primary", f"{c['diff']:+.4f}", [round(x, 4) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}", f"folds {r['folds_object_ahead_of_frame_cls']}/5")
