"""E15b (exploratory, not registered): selective conformal risk control on REAL TruckScenes weather.

Same method as E15 (accept a frame only if its label-free degradation score is at most tau; choose tau and the score
threshold lambda jointly with conformal risk control so the miss rate among accepted frames is at most alpha), applied
per scene fold, with LEAVE-ONE-CONDITION-OUT calibration: for each real adverse condition (fog, snow) the calibration
set holds the fold's labeled validation frames of every OTHER condition, so the evaluated condition is genuinely unseen.

    python -m src.e15b_selective_ts
"""
import json

import numpy as np


def frame_fnr_each(dets, gts, lam=0.02):
    from .conformal import frame_fnr
    return [float(frame_fnr([x], [g], 2.0, lam)[0]) for x, g in zip(dets, gts)]


def fold_chunks(d, cfg, alpha=0.1, kind="degradation"):
    from .conformal import crc_threshold, frame_fnr
    from .evaluate import frame_detections, gt_lists
    from .arbitrate import _frame_feat_matrix
    from .r2_experiments import frame_degradation
    from sklearn.ensemble import GradientBoostingRegressor

    arb = d.arm("object")
    thr = cfg["association"]["nms_center_m"]

    def prep(tbl):
        pr = arb.predict(tbl)
        dets = frame_detections(tbl, pr, 0, thr)
        gts, _ = gt_lists(tbl, 0)
        has = np.array([len(g) > 0 for g in gts], bool)
        dd = [x for x, h in zip(dets, has) if h]
        gg = [g for g, h in zip(gts, has) if h]
        return dd, gg, frame_degradation(tbl)[has], pr["score"][pr["score"] > 0], np.nan_to_num(_frame_feat_matrix(tbl))[has]

    if kind.startswith("learned"):
        fit = prep(d.sub(d.is_fit))
        y = np.asarray(frame_fnr_each(fit[0], fit[1]))
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0).fit(fit[4], y)
    sc_of = (lambda r: gb.predict(r[4])) if kind.startswith("learned") else (lambda r: r[2])
    val = {t: prep(d.sub(d.is_val & (d.tag == t))) for t in d.tags}
    test = {t: prep(d.test_table(t)) for t in d.tags if d.test_table(t)["frames"]}
    out = {}
    for hold in [t for t in d.tags if t != "clear0"]:
        seen = [t for t in d.tags if (t != hold or kind.endswith("-seen")) and len(val[t][0])]
        if kind.endswith("-own"):
            seen = [hold] if len(val[hold][0]) else seen
        dets = [x for t in seen for x in val[t][0]]
        gts = [x for t in seen for x in val[t][1]]
        deg = np.concatenate([sc_of(val[t]) for t in seen])
        sc = np.concatenate([val[t][3] for t in seen])
        grid = list(np.unique(np.concatenate([[0.0], np.quantile(sc, np.linspace(0, 1, 120))])))
        best = None
        for q in np.linspace(1.0, 0.05, 40):
            tau = float(np.quantile(deg, q))
            idx = np.nonzero(deg <= tau)[0]
            ds, gs = [dets[i] for i in idx], [gts[i] for i in idx]
            n = len(idx)
            lam = crc_threshold(lambda l: frame_fnr(ds, gs, 2.0, l), n, alpha, grid)
            r = float(np.mean(frame_fnr(ds, gs, 2.0, lam)))
            if n / (n + 1) * r + 1 / (n + 1) <= alpha:
                best = (tau, float(lam))
                break
        if hold not in test or best is None:
            continue
        tdets, tgts = test[hold][0], test[hold][1]
        tdeg = sc_of(test[hold])
        f = np.asarray(frame_fnr(tdets, tgts, 2.0, best[1])) if len(tdets) else np.zeros(0)
        out[hold] = {"fnr": f, "accept": tdeg <= best[0], "tau": best[0]}
    return out


def run(k=5, alpha=0.1, kind="degradation"):
    from .common import resolve
    from .r2_experiments import fold_ctx, scene_folds
    from .r2a_truckscenes import ts_ctx

    ctx = ts_ctx()
    cfg = ctx.cfg
    chunks = []
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        chunks.append(fold_chunks(fold_ctx(ctx, fs, f"ts_fold{i}"), cfg, alpha, kind))
        ctx.log.info(f"[e15b] fold {i} done")
    res = {"dataset": "truckscenes", "alpha": alpha, "score": kind, "protocol": "calibration includes the evaluated condition" if kind.endswith("-seen") else "leave-one-condition-out calibration", "conditions": {}}
    for t in sorted({t for c in chunks for t in c}):
        f = np.concatenate([c[t]["fnr"] for c in chunks if t in c])
        a = np.concatenate([c[t]["accept"] for c in chunks if t in c])
        res["conditions"][t] = {"n_frames": int(len(f)), "accepted": float(a.mean()), "fnr_all_frames": float(f.mean()),
                                "fnr_accepted": float(f[a].mean()) if a.any() else None}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e15b_selective_ts_{kind}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "degradation"
    for t, v in run(kind=kind)["conditions"].items():
        a = v["fnr_accepted"]
        print(f"{t:8s} n={v['n_frames']:5d} accepted={v['accepted']:.2f} FNR all={v['fnr_all_frames']:.3f} "
              f"FNR accepted={'n/a' if a is None else f'{a:.3f}'}")
