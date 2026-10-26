"""E14 (exploratory, not registered): does condition-aware (Mondrian) calibration beat plain inflation at equal radius?

E12 showed the label-free degradation-normalised radius equals a global inflation of the plain radius. Here the
calibration set also contains a few LABELED adverse validation frames (a realistic one-off calibration drive), and the
conformal score quantile is taken per degradation bin (Mondrian conformal). A bin is the tercile of the predicted
log-scale multiplier psi'kappa of the degradation-normalised model, fitted on training scenes only. Baselines at the
same mean radius: (a) pooled conformal on the same clear+adverse calibration set, (b) the clear-only vanilla radius
inflated by one global factor. Scene-disjoint folds and models are exactly those of E12; nothing is retrained.

    python -m src.e14_mondrian
"""
import json

import numpy as np


def fold_arrays(d, cfg, alpha=0.1, n_bins=3):
    from .conformal import LocalisationConformal, conformal_quantile
    from .r2_experiments import _deg_loc_data

    arb = d.arm("object")
    dtr, _ = _deg_loc_data(arb, d.sub(d.is_fit), cfg)
    clear = d.sub(d.is_val & (d.tag == "clear0"))
    adv = d.sub(d.is_val & (d.tag != "clear0"))
    dcl, _ = _deg_loc_data(arb, clear, cfg)
    dad, _ = _deg_loc_data(arb, adv, cfg)
    mu, sd = dtr["psi"].mean(0), dtr["psi"].std(0) + 1e-6
    feat = lambda x: np.concatenate([x["phi"], (x["psi"] - mu) / sd], 1)
    m = LocalisationConformal(True)
    m.fit_scale(dtr["err"], dtr["rng"], feat(dtr))
    van = LocalisationConformal(False)
    van.fit_scale(dtr["err"], dtr["rng"], dtr["phi"])
    van.calibrate(dcl["err"], dcl["rng"], dcl["phi"], alpha)
    # degradation level = predicted log-scale multiplier (label-free); bin edges from training scenes
    level = lambda x: np.clip(feat(x) @ m.kappa, -3, 3)
    edges = np.quantile(level(dtr), np.linspace(0, 1, n_bins + 1)[1:-1])
    b_of = lambda x: np.digitize(level(x), edges)
    cal = {k: np.concatenate([dcl[k], dad[k]]) for k in ("err", "rng")}
    cal["f"] = np.concatenate([feat(dcl), feat(dad)])
    cal["level"] = np.concatenate([level(dcl), level(dad)])
    s = cal["err"] / m._scale(cal["rng"], cal["f"])
    q_pool = conformal_quantile(s, alpha)
    cb = np.digitize(cal["level"], edges)
    q_bin = {b: conformal_quantile(s[cb == b], alpha) if (cb == b).sum() >= 30 else q_pool for b in range(n_bins)}
    out = {}
    for t in d.tags:
        tt = d.test_table(t)
        if not tt["frames"]:
            continue
        x, _ = _deg_loc_data(arb, tt, cfg)
        f, sc = feat(x), None
        sc = m._scale(x["rng"], f)
        out[t] = {"err": np.asarray(x["err"]), "rad_pooled": sc * q_pool,
                  "rad_mondrian": sc * np.array([q_bin[b] for b in b_of(x)]),
                  "rad_vanilla": np.asarray(van.radius(x["rng"], x["phi"]))}
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
        ctx.log.info(f"[e14] fold {i} done")
    res = {"dataset": "truckscenes", "alpha": 0.1, "conditions": {}}
    for t in sorted({t for c in chunks for t in c}):
        cat = {key: np.concatenate([c[t][key] for c in chunks if t in c]) for key in next(c[t] for c in chunks if t in c)}
        ok = np.isfinite(cat["err"]) & np.all([np.isfinite(cat[key]) for key in cat if key.startswith("rad_")], axis=0)
        cat = {key: v[ok] for key, v in cat.items()}
        err, mon = cat["err"], cat["rad_mondrian"]
        row = {"n_objects": int(len(err)), "mean_radius_mondrian": float(mon.mean()),
               "coverage_mondrian": float((err <= mon).mean())}
        for name in ("pooled", "vanilla"):
            r = cat[f"rad_{name}"]
            s = mon.mean() / r.mean()
            row[f"coverage_{name}"] = float((err <= r).mean())
            row[f"coverage_{name}_at_matched_radius"] = float((err <= s * r).mean())
        res["conditions"][t] = row
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e14_mondrian.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    for t, r in run()["conditions"].items():
        print(t, r["n_objects"], f"mondrian {r['coverage_mondrian']:.3f} @ {r['mean_radius_mondrian']:.2f} m",
              f"| pooled@matched {r['coverage_pooled_at_matched_radius']:.3f}",
              f"| vanilla@matched {r['coverage_vanilla_at_matched_radius']:.3f}")
