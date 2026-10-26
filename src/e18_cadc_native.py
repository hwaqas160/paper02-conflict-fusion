"""E18 (exploratory, not registered): a router trained on CADC's own drives (no Boreas data at all).

R2-D showed that Boreas-trained routers give no advantage on real CADC snow (no radar), zero-shot or with a few labelled
drives mixed into the Boreas training set. Here the router and the fixed-weight status quo are both trained on CADC alone,
by five-fold drive cross-validation, so every test drive is scored by models that never saw it. CADC has one condition
(snow), so its frames are relabelled clear/0 internally: the "fixed" arm is then the fixed-weight rule fitted on the
training drives themselves, and a per-frame weather classifier is undefined (one class); the comparison is per-object
router versus in-domain fixed weights.

    python -m src.e18_cadc_native
"""
import copy
import json

import numpy as np


def run(k=5, n_boot_seed=0):
    from .arbitrate import merge_tables
    from .common import load_config, resolve, seed_everything
    from .evaluate import eval_map, map_of_units
    from .r2d_cadc import NoRadar, drive_units, load_cadc_table
    from .run_experiments import Ctx
    from .supplementary import merge_units, scene_paired

    cfg0 = load_config()
    seed_everything(cfg0["seed"])
    T = load_cadc_table(cfg0, "cadc_norm")
    drives = sorted({f["scene"] for f in T["frames"]})
    rng = np.random.default_rng(0)
    order = list(rng.permutation(drives))
    folds = [order[i::k] for i in range(k)]
    units = {"fixed": {}, "object": {}}
    maps = {"fixed": [], "object": []}
    for i, test_drives in enumerate(folds):
        Tc = copy.deepcopy(T)
        for f in Tc["frames"]:
            f["split"] = "test" if f["scene"] in test_drives else "train"
            f["condition"], f["severity"] = "clear", 0
        cfg = copy.deepcopy(cfg0)
        cfg["paths"]["results_root"] = str(resolve(cfg0, "results_root") / "e18" / f"fold{i}")
        d = Ctx(cfg, table=Tc)
        arms = {"fixed": NoRadar(d.status_quo()), "object": NoRadar(d.arm("object"))}
        test = d.test_table()
        for a, m in arms.items():
            pr = m.predict(test)
            maps[a].append(eval_map(test, pr, cfg)[0])
            for dv, u in drive_units(test, pr, cfg).items():
                units[a].setdefault(dv, []).append(u)
        d.log.info(f"[e18] fold {i}: fixed {maps['fixed'][-1]:.4f} object {maps['object'][-1]:.4f}")
    ds = sorted(units["object"])
    U = {a: [merge_units(units[a][dv]) for dv in ds] for a in units}
    res = {"n_drives": len(ds), "fold_map": maps, "oof_map": {a: map_of_units(U[a], cfg0) for a in U},
           "object_vs_fixed": scene_paired(U["object"], U["fixed"], cfg0),
           "folds_object_ahead": int(sum(o > f for o, f in zip(maps["object"], maps["fixed"])))}
    out = resolve(cfg0, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e18_cadc_native.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run()
    c = r["object_vs_fixed"]
    print({a: round(v, 4) for a, v in r["oof_map"].items()})
    print("object vs fixed", f"{c['diff']:+.4f}", [round(x, 4) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}",
          f"folds ahead {r['folds_object_ahead']}/5")
