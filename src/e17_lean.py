"""E17 (exploratory, not registered): a lean router -- keep only the one mechanism that R2-C found to contribute.

R2-C retrained the router with each of six mechanisms removed and found that only the harm auxiliary loss contributes
(adverse mAP -0.0066, p_Holm 0.030); the other five (absence term, learned position precision, localisation loss, sensor
dropout, frame context) do not. This trains ONE router with all five removed and the harm loss kept, and compares it
with the full router and the per-frame classifier on Boreas simulated weather and on real TruckScenes (five-fold scene CV).

    python -m src.e17_lean
"""
import hashlib
import json
import pickle

import numpy as np

LEAN_KW = {"absence_scale": 0.0, "learn_precision": False, "modality_dropout": 0.0, "context": False}
LEAN_OVER = {"loss_loc_weight": 0.0}


def fit_lean(ctx, out_dir, tag="lean"):
    import copy as _c
    from pathlib import Path
    from .arbitrate import fit_arbiter
    from .evaluate import choose_tau_f1
    from .r2_experiments import fingerprint

    h = hashlib.sha256((fingerprint(ctx) + tag).encode())
    h.update((Path(__file__).parent / "arbitrate.py").read_bytes())
    fp = h.hexdigest()[:16]
    path = Path(out_dir) / f"e17_{tag}.pkl"
    if path.exists():
        obj = pickle.load(open(path, "rb"))
        if getattr(obj, "_fp", None) == fp:
            return obj
    cfg = _c.deepcopy(ctx.cfg)
    cfg["arbitration"].update(LEAN_OVER)
    h_lab, _, _ = ctx.harm()
    fit_m, val_m = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cand_fit, cand_val = fit_m[ctx.T["cand"]["frame"]], val_m[ctx.T["cand"]["frame"]]
    obj = fit_arbiter("object", ctx.sub(fit_m), ctx.sub(val_m), cfg, harm_labels=h_lab[cand_fit],
                      val_harm=h_lab[cand_val], log=ctx.log, seed=cfg["seed"], **LEAN_KW)
    train_clear = ctx.sub((ctx.split == "train") & (ctx.tag == "clear0"))
    obj.tau, obj.f1 = choose_tau_f1(train_clear, obj.predict(train_clear), ctx.cfg)
    obj._fp = fp
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(path, "wb"))
    return obj


def boreas(res):
    from .common import load_config, seed_everything
    from .evaluate import eval_map, per_frame_units
    from .run_experiments import Ctx
    from .supplementary import scene_paired, scene_units

    cfg = load_config()
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    arms = {"frame_cls": ctx.arm("frame_cls"), "full": ctx.arm("object"), "lean": fit_lean(ctx, ctx.out / "models_r2")}
    tt = ctx.test_table(tags=ctx.adverse_tags)
    units, maps = {}, {}
    for n, m in arms.items():
        pr = m.predict(tt)
        maps[n] = eval_map(tt, pr, cfg)[0]
        units[n] = scene_units(tt, per_frame_units(tt, pr, cfg))
    res["boreas"] = {"map": maps, "params": {n: int(m.n_params()) for n, m in arms.items()},
                     "lean_vs_frame_cls": scene_paired(units["lean"], units["frame_cls"], cfg),
                     "lean_vs_full": scene_paired(units["lean"], units["full"], cfg)}
    print("boreas", {k: round(v, 3) for k, v in maps.items()}, flush=True)


def truckscenes(res, k=5, n_boot=10000):
    from .e10_minimal import compare
    from .evaluate import eval_map, map_of_units, per_frame_units
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import ts_ctx
    from .r4_experiments import register
    from .supplementary import scene_units

    register()
    ctx = ts_ctx()
    cfg = ctx.cfg
    oof = {n: [] for n in ("frame_cls", "full", "lean")}
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        d = fold_ctx(ctx, fs, f"e17_fold{i}")
        tt = d.test_table(tags=ctx.adverse_tags)
        arms = {"frame_cls": d.arm("frame_cls"), "full": d.arm("object"), "lean": fit_lean(d, d.model_dir)}
        for n, m in arms.items():
            oof[n].extend(scene_units(tt, per_frame_units(tt, m.predict(tt), cfg)))
        ctx.log.info(f"[e17] fold {i} done")
    res["truckscenes"] = {"oof_map": {n: map_of_units(u, cfg) for n, u in oof.items()},
                          "lean_vs_frame_cls": compare(oof["lean"], oof["frame_cls"], cfg, n_boot),
                          "lean_vs_full": compare(oof["lean"], oof["full"], cfg, n_boot)}
    print("truckscenes", {k_: round(v, 3) for k_, v in res["truckscenes"]["oof_map"].items()}, flush=True)


if __name__ == "__main__":
    import sys
    from .common import load_config, resolve

    res = {"lean_kwargs": LEAN_KW, "lean_overrides": LEAN_OVER}
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "boreas"):
        boreas(res)
    if which in ("both", "truckscenes"):
        truckscenes(res)
    out = resolve(load_config(), "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e17_lean_{which}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for k in ("boreas", "truckscenes"):
        if k in res:
            for c in ("lean_vs_frame_cls", "lean_vs_full"):
                r = res[k][c]
                print(k, c, f"{r['diff']:+.4f}", [round(x, 4) for x in r["ci95"]], f"p={r['p_one_sided']:.4f}")
