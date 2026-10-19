"""E11 (exploratory, not registered): the real-weather TruckScenes result broken out by condition.

TruckScenes has few fog and snow scenes (12 and 23 against 119 rain and 76 night), so the pooled-adverse headline is
dominated by rain and night. This reports per-object minus per-frame-classifier Car mAP separately for each condition,
out-of-fold, with the number of scenes, a scene-cluster bootstrap and an exact sign test. Uses the cached R2-A fold
models; nothing is retrained.

    python -m src.e11_percond [n_boot]
"""
import json
import sys

from .e10_minimal import compare


def run(k=5, n_boot=1000):
    from .common import resolve
    from .evaluate import map_of_units, per_frame_units
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import ts_ctx
    from .supplementary import scene_units

    ctx = ts_ctx()
    cfg = ctx.cfg
    tags = list(ctx.tags)
    oof = {t: {"object": [], "frame_cls": []} for t in tags}
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        d = fold_ctx(ctx, fs, f"ts_fold{i}")
        models = {"object": d.arm("object"), "frame_cls": d.arm("frame_cls")}
        for t in tags:
            tt = d.test_table(tag=t)
            if not tt["frames"]:
                continue
            for n, m in models.items():
                oof[t][n].extend(scene_units(tt, per_frame_units(tt, m.predict(tt), cfg)))
        ctx.log.info(f"[e11] fold {i} done")
    res = {"dataset": "truckscenes", "n_boot": n_boot, "conditions": {}}
    for t in tags:
        a, b = oof[t]["object"], oof[t]["frame_cls"]
        if len(a) < 2:
            continue
        c = compare(a, b, cfg, n_boot)
        c["map_object"], c["map_frame_cls"] = map_of_units(a, cfg), map_of_units(b, cfg)
        res["conditions"][t] = c
        ctx.log.info(f"[e11] {t}: {c['diff']:+.3f} over {c['n_scenes']} scenes")
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e11_per_condition.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run(n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    for t, c in r["conditions"].items():
        print(t, f"{c['diff']:+.3f}", [round(x, 3) for x in c["ci95"]], f"{c['scenes_ahead']}/{c['n_scenes']}",
              "sign_p", f"{c['sign_test_p_two_sided']:.2g}")
