"""E19 (exploratory): recompute the CAMERA degradation probes at the corrected camera boxes of the E16 fully in-domain cache.

E16 shifted camera boxes along their ray by a learned range correction but left camera_probes at the original boxes.
This writes results/ts_detector/cache_full2 with camera_probes recomputed at the corrected boxes (same compute_probes call
and inputs as ts_cache.py); everything else is copied.

    python -m src.e19_reprobe probes [--workers 10]
    python -m src.e19_reprobe candidates
    python -m src.e19_reprobe router
"""
import multiprocessing as mp
import sys

import numpy as np

from .common import load_config, resolve
from . import e8c_rebuild as R


def roots(cfg):
    base = resolve(cfg, "results_root") / "ts_detector"
    return base / "cache_full" / "truckscenes", base / "cache_full2" / "truckscenes", base / "candidates_full2" / "truckscenes"


def _one(item):
    from .candidates import load_sample
    from .truckscenes_data import read_lidar
    path, out = item
    meta, s = load_sample(path)
    fr = R._W["by_id"][meta["frame_id"]]
    s = dict(s)
    if len(s["camera_scores"]):
        s["camera_probes"] = R.probes_for(fr, read_lidar(fr), s["camera_boxes"].astype(np.float64))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(repr(meta)), **s)
    tmp.replace(out)
    return meta["frame_id"]


def main():
    cmd = sys.argv[1]
    cfg = load_config()
    src, dst, cand = roots(cfg)
    if cmd == "probes":
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 10
        ground_z = R.load_json(R.paths(cfg)[0] / "manifest.json")["ground_z"]
        items = [(p, dst / p.parent.name / p.name) for p in sorted(src.glob("*/*.npz"))
                 if not (dst / p.parent.name / p.name).exists()]
        print(len(items), "frames", flush=True)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "manifest.json").write_text((src / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
        done = 0
        with mp.get_context("spawn").Pool(workers, initializer=R._init, initargs=(cfg, ground_z, 2)) as pool:
            for res, err in pool.imap_unordered(lambda_safe, items, chunksize=2):
                done += 1
                if err:
                    print(err, flush=True)
                if done % 1000 == 0:
                    print("probes", done, flush=True)
    elif cmd == "candidates":
        from .candidates import build_table, save_table
        for tag in sorted(p.name for p in dst.iterdir() if p.is_dir()):
            tbl = build_table(dst / tag, cfg)
            if tbl is not None:
                save_table(tbl, cand / f"{tag}.npz")
                print(tag, len(tbl["frames"]), flush=True)
    else:
        from .e8c_router import run
        r = run(n_boot=10000, n_boot2=1000, cand_name="candidates_full2", out_name="e19_reprobe_router.json", router_dir="router_full2")
        c = r["primary_object_vs_frame_cls"]
        print({k: round(v, 3) for k, v in r["oof_map"].items()})
        print("primary", f"{c['diff']:+.4f}", [round(x, 4) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}", f"folds {r['folds_object_ahead_of_frame_cls']}/5")


def lambda_safe(item):
    return R._safe(_one, item)


if __name__ == "__main__":
    main()
