"""E10 (exploratory, not registered): how much of the per-object advantage do the most minimal per-object
inputs already give?

Three small per-object routers use the same fusion rule, loss, optimiser and stopping rule as every other arm and
differ only in their input blocks:
  geom        range and azimuth only
  geom_conf   range, azimuth + detector logits, presence, field-of-view and class (no degradation probes)
  geom_clocs  range, azimuth + the CLOCs-style blocks (detector logits, detector features, conflict features)
None of them reads a degradation probe or a frame-level statistic. Each is compared with the per-frame classifier and
with the full per-object router on adverse-pooled Car mAP, by a scene-cluster bootstrap and by an exact two-sided
sign test over scenes (the bootstrap is fragile with few clusters).

    python -m src.e10_minimal boreas
    python -m src.e10_minimal truckscenes
"""
import json
import sys
from math import comb

import numpy as np

ARMS = {
    "geom": ["geometry"],
    "geom_conf": ["geometry", "confidence"],
    "geom_clocs": ["geometry", "det_feats", "conflict", "confidence"],
}


def sign_test_p(wins, n):
    """Exact two-sided binomial sign test against p=0.5 (ties already removed)."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n))


def compare(u_a, u_b, cfg, n_boot):
    from .evaluate import map_of_units
    from .supplementary import scene_paired
    boot = scene_paired(u_a, u_b, cfg, n=n_boot)
    diffs = [map_of_units([a], cfg) - map_of_units([b], cfg) for a, b in zip(u_a, u_b)]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    boot["scenes_ahead"], boot["scenes_behind"] = int(wins), int(losses)
    boot["sign_test_p_two_sided"] = sign_test_p(int(wins), int(wins + losses))
    return boot


def run_boreas(n_boot=10000):
    from .common import load_config, resolve
    from .evaluate import map_of_units, per_frame_units
    from .run_experiments import Ctx
    from .supplementary import scene_units

    cfg = load_config()
    ctx = Ctx(cfg)
    tt = ctx.test_table(tags=ctx.adverse_tags)
    models = {"frame_cls": ctx.arm("frame_cls"), "object": ctx.arm("object")}
    for name, blocks in ARMS.items():
        models[name] = ctx.arm("object", blocks=blocks, context=False, tag=f"e10_{name}")
    units = {n: scene_units(tt, per_frame_units(tt, m.predict(tt), cfg)) for n, m in models.items()}
    res = {"dataset": "boreas", "pool": "ADVERSE_POOLED", "n_boot": n_boot,
           "map": {n: map_of_units(u, cfg) for n, u in units.items()},
           "params": {n: m.n_params() for n, m in models.items()},
           "vs_frame_cls": {n: compare(units[n], units["frame_cls"], cfg, n_boot) for n in ARMS},
           "object_vs_frame_cls": compare(units["object"], units["frame_cls"], cfg, n_boot),
           "object_vs_minimal": {n: compare(units["object"], units[n], cfg, n_boot) for n in ARMS}}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e10_minimal_boreas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_truckscenes(k=5, n_boot=10000):
    from .evaluate import eval_map, map_of_units, per_frame_units
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import ts_ctx
    from .common import resolve
    from .supplementary import scene_units

    ctx = ts_ctx()
    cfg = ctx.cfg
    names = ["frame_cls", "object"] + list(ARMS)
    oof = {n: [] for n in names}
    per_fold = []
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        d = fold_ctx(ctx, fs, f"ts_fold{i}")  # same dir as R2-A: frame_cls and object load from cache
        models = {"frame_cls": d.arm("frame_cls"), "object": d.arm("object")}
        for name, blocks in ARMS.items():
            models[name] = d.arm("object", blocks=blocks, context=False, tag=f"e10_{name}")
        tt = d.test_table(tags=ctx.adverse_tags)
        maps = {}
        for n, m in models.items():
            pr = m.predict(tt)
            maps[n] = eval_map(tt, pr, cfg)[0]
            oof[n].extend(scene_units(tt, per_frame_units(tt, pr, cfg)))
        per_fold.append({"n_test_scenes": len(fs), "map": maps})
        ctx.log.info(f"[e10] fold {i}: {maps}")
    res = {"dataset": "truckscenes", "pool": "ADVERSE_POOLED, out-of-fold", "n_boot": n_boot,
           "oof_map": {n: map_of_units(u, cfg) for n, u in oof.items()},
           "vs_frame_cls": {n: compare(oof[n], oof["frame_cls"], cfg, n_boot) for n in ARMS},
           "object_vs_frame_cls": compare(oof["object"], oof["frame_cls"], cfg, n_boot),
           "object_vs_minimal": {n: compare(oof["object"], oof[n], cfg, n_boot) for n in ARMS},
           "folds_ahead_of_frame_cls": {n: int(sum(f["map"][n] > f["map"]["frame_cls"] for f in per_fold))
                                        for n in ARMS},
           "per_fold": per_fold}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e10_minimal_truckscenes.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "boreas"
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    r = run_boreas(nb) if which == "boreas" else run_truckscenes(n_boot=nb)
    print(json.dumps({k: r[k] for k in r if k in ("map", "oof_map", "vs_frame_cls", "object_vs_minimal")}, indent=2))
