"""E12 (exploratory, not registered): is the degradation-normalised conformal radius better than simply inflating the
plain radius?

Inflating any radius restores coverage, so coverage alone is not evidence for a repair. For each real TruckScenes
condition this rescales the vanilla and the conflict-inflated radii by one global factor so that their mean radius equals
the degradation-normalised radius's mean radius on that condition, then reports the coverage each reaches at that equal
mean radius. Calibration protocol, scales and fold models are exactly those of R2-E (clear-only calibration, fold
validation scenes); nothing is retrained.

    python -m src.e12_matched_radius
"""
import json

import numpy as np


def fold_arrays(d, cfg, alpha=0.1):
    from .conformal import LocalisationConformal
    from .r2_experiments import _deg_loc_data

    arb = d.arm("object")
    dtr, _ = _deg_loc_data(arb, d.sub(d.is_fit), cfg)
    cal = d.sub(d.is_val & (d.tag == "clear0"))
    dcal, _ = _deg_loc_data(arb, cal, cfg)
    mu, sd = dtr["psi"].mean(0), dtr["psi"].std(0) + 1e-6
    feats = {"vanilla": lambda x: x["phi"], "conflict": lambda x: x["phi"],
             "degradation_normalised": lambda x: np.concatenate([x["phi"], (x["psi"] - mu) / sd], 1)}
    methods = {"vanilla": LocalisationConformal(False), "conflict": LocalisationConformal(True),
               "degradation_normalised": LocalisationConformal(True)}
    for k, m in methods.items():
        m.fit_scale(dtr["err"], dtr["rng"], feats[k](dtr))
        m.calibrate(dcal["err"], dcal["rng"], feats[k](dcal), alpha)
    out = {}
    for t in d.tags:
        tt = d.test_table(t)
        if not tt["frames"]:
            continue
        x, _ = _deg_loc_data(arb, tt, cfg)
        out[t] = {"err": np.asarray(x["err"]), **{f"rad_{k}": np.asarray(m.radius(x["rng"], feats[k](x)))
                                                  for k, m in methods.items()}}
    return out


def run(k=5):
    from .common import resolve
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import ts_ctx

    ctx = ts_ctx()
    cfg = ctx.cfg
    chunks = []
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        chunks.append(fold_arrays(fold_ctx(ctx, fs, f"ts_fold{i}"), cfg))
        ctx.log.info(f"[e12] fold {i} done")
    res = {"dataset": "truckscenes", "alpha": 0.1, "conditions": {}}
    for t in sorted({t for c in chunks for t in c}):
        cat = {key: np.concatenate([c[t][key] for c in chunks if t in c]) for key in next(c[t] for c in chunks if t in c)}
        ok = np.isfinite(cat["err"]) & np.all([np.isfinite(cat[key]) for key in cat if key.startswith("rad_")], axis=0)
        cat = {key: v[ok] for key, v in cat.items()}
        err = cat["err"]
        dn = cat["rad_degradation_normalised"]
        row = {"n_objects": int(len(err)), "mean_radius_normalised": float(dn.mean()),
               "coverage_normalised": float((err <= dn).mean())}
        for name in ("vanilla", "conflict"):
            r = cat[f"rad_{name}"]
            s = dn.mean() / r.mean()
            row[f"coverage_{name}"] = float((err <= r).mean())
            row[f"mean_radius_{name}"] = float(r.mean())
            row[f"scale_to_match_{name}"] = float(s)
            row[f"coverage_{name}_at_matched_radius"] = float((err <= s * r).mean())
        res["conditions"][t] = row
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e12_matched_radius.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    for t, r in run()["conditions"].items():
        print(t, r["n_objects"], f"normalised {r['coverage_normalised']:.3f} @ {r['mean_radius_normalised']:.2f} m",
              f"| vanilla@matched {r['coverage_vanilla_at_matched_radius']:.3f}",
              f"| conflict@matched {r['coverage_conflict_at_matched_radius']:.3f}")
