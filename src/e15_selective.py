"""E15 (exploratory, not registered): selective conformal risk control -- control the miss rate by ABSTAINING.

R2-E1 showed that no score threshold can control the frame miss rate in severe fog: even accepting every candidate
misses a third of the cars (the detection floor). Thresholding cannot create detections that do not exist, so the
resolvable form of the guarantee is selective: accept a frame only if its label-free degradation score is at most tau,
and calibrate (tau, lambda) jointly with conformal risk control so that the miss rate AMONG ACCEPTED frames is at most
alpha. Severe conditions (tags ending in 3) are held out of calibration entirely, so a frame from an unseen, harsher
condition can only be accepted if its degradation score says it looks like a calibrated one.

    python -m src.e15_selective
"""
import json

import numpy as np


def run(alpha=0.1):
    from .common import load_config, resolve, seed_everything
    from .conformal import crc_threshold, frame_fnr
    from .evaluate import frame_detections, gt_lists
    from .r2_experiments import frame_degradation
    from .run_experiments import Ctx

    cfg = load_config()
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    arb = ctx.arm("object")
    thr = cfg["association"]["nms_center_m"]
    seen = [t for t in ctx.tags if not t.endswith("3")]
    cal = ctx.sub((ctx.split == "cal") & np.isin(ctx.tag, seen))
    pr = arb.predict(cal)
    dets = frame_detections(cal, pr, 0, thr)
    gts, _ = gt_lists(cal, 0)
    deg = frame_degradation(cal)
    has = np.array([len(g) > 0 for g in gts])
    dets = [d for d, h in zip(dets, has) if h]
    gts = [g for g, h in zip(gts, has) if h]
    deg = deg[has]
    grid = list(np.unique(np.concatenate([[0.0], np.quantile(pr["score"][pr["score"] > 0], np.linspace(0, 1, 120))])))
    best = None
    for q in np.linspace(1.0, 0.05, 40):
        tau = float(np.quantile(deg, q))
        idx = np.nonzero(deg <= tau)[0]
        ds, gs = [dets[i] for i in idx], [gts[i] for i in idx]
        n = len(idx)
        lam = crc_threshold(lambda l: frame_fnr(ds, gs, 2.0, l), n, alpha, grid)
        r = float(np.mean(frame_fnr(ds, gs, 2.0, lam)))
        if n / (n + 1) * r + 1 / (n + 1) <= alpha:
            best = {"tau": tau, "lambda": float(lam), "cal_accept": float(q), "cal_risk": r, "n_cal": n}
            break
    res = {"alpha": alpha, "calibration_conditions": seen, "chosen": best, "conditions": {}}
    if best is None:
        res["note"] = "no feasible (tau, lambda): detection floor above alpha even on the least degraded frames"
    else:
        for t in ctx.tags:
            tt = ctx.test_table(tags=[t])
            if not len(tt["frames"]):
                continue
            p = arb.predict(tt)
            d = frame_detections(tt, p, 0, thr)
            g, _ = gt_lists(tt, 0)
            h = np.array([len(x) > 0 for x in g])
            acc = (frame_degradation(tt) <= best["tau"])[h]
            f = np.asarray(frame_fnr([x for x, k in zip(d, h) if k], [x for x, k in zip(g, h) if k], 2.0, best["lambda"]))
            res["conditions"][t] = {"n_frames": int(len(f)), "accepted": float(acc.mean()),
                                    "fnr_all_frames": float(f.mean()),
                                    "fnr_accepted": float(f[acc].mean()) if acc.any() else None,
                                    "held_out_condition": bool(t.endswith("3"))}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e15_selective.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run()
    print("chosen:", r["chosen"])
    for t, v in r["conditions"].items():
        a = v["fnr_accepted"]
        print(f"{t:10s} held-out={v['held_out_condition']!s:5s} n={v['n_frames']:4d} accepted={v['accepted']:.2f} "
              f"FNR all={v['fnr_all_frames']:.3f} FNR accepted={'n/a' if a is None else f'{a:.3f}'}")
