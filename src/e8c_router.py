"""E8-C step 3: the registered TruckScenes five-fold scene cross-validation on candidates built with the cross-fitted
in-domain LiDAR Car detector (notes/e8c_indomain_detector.md). Per-object router vs per-frame classifier, plus the
equal-information frame-pooled arm, the fixed-weight status quo and the E10 minimal per-object arm.

    python -m src.e8c_router [n_boot_primary] [n_boot_secondary]
"""
import json
import sys

import numpy as np

from .e10_minimal import ARMS, compare


def run(k=5, n_boot=10000, n_boot2=1000):
    from .common import load_config, resolve, seed_everything
    from .evaluate import eval_map, map_of_units, per_frame_units
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import TS_OVERRIDES
    from .r4_experiments import register
    from .run_experiments import Ctx
    from .supplementary import scene_units

    register()
    base = load_config(None, TS_OVERRIDES)
    cand_root = resolve(base, "results_root").parent / "ts_detector" / "candidates"
    assert (cand_root / "truckscenes" / "clear0.npz").exists(), "run e8c_rebuild candidates first"
    cfg = load_config(None, {**TS_OVERRIDES, "paths.candidates_root": str(cand_root),
                             "paths.results_root": "results/ts_detector/router"})
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg, dataset="truckscenes")
    names = ["fixed_clear", "frame_cls", "object", "frame_pooled", "geom_clocs"]
    oof = {n: [] for n in names}
    per_fold = []
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        d = fold_ctx(ctx, fs, f"e8c_fold{i}")
        models = {"fixed_clear": d.status_quo(), "frame_cls": d.arm("frame_cls"), "object": d.arm("object"),
                  "frame_pooled": d.arm("frame_pooled"),
                  "geom_clocs": d.arm("object", blocks=ARMS["geom_clocs"], context=False, tag="e8c_geom_clocs")}
        tt = d.test_table(tags=ctx.adverse_tags)
        maps = {}
        for n, m in models.items():
            pr = m.predict(tt)
            maps[n] = eval_map(tt, pr, cfg)[0]
            oof[n].extend(scene_units(tt, per_frame_units(tt, pr, cfg)))
        per_fold.append({"n_test_scenes": len(fs), "map": maps})
        ctx.log.info(f"[e8c] fold {i}: {maps}")
    ctx.log.info("[e8c] folds done; bootstraps")
    res = {"candidates": "cross-fitted in-domain LiDAR Car detector", "n_boot_primary": n_boot, "n_boot_secondary": n_boot2,
           "oof_map": {n: map_of_units(u, cfg) for n, u in oof.items()}, "per_fold": per_fold}
    res["folds_object_ahead_of_frame_cls"] = int(sum(f["map"]["object"] > f["map"]["frame_cls"] for f in per_fold))
    res["folds_object_ahead_of_frame_pooled"] = int(sum(f["map"]["object"] > f["map"]["frame_pooled"] for f in per_fold))
    res["primary_object_vs_frame_cls"] = compare(oof["object"], oof["frame_cls"], cfg, n_boot)
    res["primary_refuted"] = bool(res["primary_object_vs_frame_cls"]["diff"] <= 0
                                  or res["primary_object_vs_frame_cls"]["p_one_sided"] >= 0.05
                                  or res["folds_object_ahead_of_frame_cls"] < 4)
    res["object_vs_frame_pooled"] = compare(oof["object"], oof["frame_pooled"], cfg, n_boot2)
    res["object_vs_fixed_clear"] = compare(oof["object"], oof["fixed_clear"], cfg, n_boot2)
    res["geom_clocs_vs_frame_cls"] = compare(oof["geom_clocs"], oof["frame_cls"], cfg, n_boot2)
    res["object_vs_geom_clocs"] = compare(oof["object"], oof["geom_clocs"], cfg, n_boot2)
    out = resolve(cfg, "results_root").parent / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e8c_indomain_router.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run(n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 10000,
            n_boot2=int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    print({k: round(v, 3) for k, v in r["oof_map"].items()})
    c = r["primary_object_vs_frame_cls"]
    print("primary", f"{c['diff']:+.3f}", [round(x, 3) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}",
          f"folds ahead {r['folds_object_ahead_of_frame_cls']}/5", "refuted" if r["primary_refuted"] else "not refuted")
