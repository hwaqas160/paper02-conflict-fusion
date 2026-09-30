"""Detection and calibration metrics.

Primary detection metric: nuScenes-style BEV centre-distance AP (Caesar et al., CVPR 2020),
101-point interpolated, averaged over distance thresholds. Centre distance rather than IoU because
the camera stream contributes range estimates from monocular geometry — IoU would conflate
"did the fusion find the object" with "how well did a monocular lift guess its footprint".
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def match_frame(pred_xy, pred_scores, gt_xy, thr, dontcare_xy=None):
    """Greedy score-ordered matching (nuScenes). Returns tp (P,), ignore (P,), matched gt idx (P,)."""
    order = np.argsort(-pred_scores)
    tp = np.zeros(len(pred_xy), bool)
    ignore = np.zeros(len(pred_xy), bool)
    gt_idx = np.full(len(pred_xy), -1)
    taken = np.zeros(len(gt_xy), bool)
    for i in order:
        if len(gt_xy):
            d = np.linalg.norm(gt_xy - pred_xy[i], axis=1)
            d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= thr:
                tp[i], taken[j], gt_idx[i] = True, True, j
                continue
        if dontcare_xy is not None and len(dontcare_xy):
            if np.min(np.linalg.norm(dontcare_xy - pred_xy[i], axis=1)) <= thr:
                ignore[i] = True
    return tp, ignore, gt_idx


def ap_from_matches(scores, tp, n_gt, min_recall=0.1, min_precision=0.1):
    """nuScenes AP: 101-point precision envelope, recall/precision floors of 0.1 removed."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    rec_grid = np.linspace(0, 1, 101)
    prec_env = np.maximum.accumulate(prec[::-1])[::-1]
    p_interp = np.interp(rec_grid, rec, prec_env, right=0)
    p_interp[rec_grid > rec[-1]] = 0
    p = p_interp[int(round(min_recall * 100)) + 1:]
    p = p - min_precision
    p[p < 0] = 0
    return float(np.mean(p) / (1.0 - min_precision))


def average_precision_center(preds, gts, thr, dontcares=None):
    """preds: list of (boxes (P,>=2), scores (P,)); gts: list of (G,>=2). Returns AP at one threshold."""
    all_s, all_tp, n_gt = [], [], 0
    for k, ((pb, ps), gb) in enumerate(zip(preds, gts)):
        dc = None if dontcares is None else dontcares[k][:, :2]
        tp, ign, _ = match_frame(np.asarray(pb)[:, :2].reshape(-1, 2), np.asarray(ps), np.asarray(gb)[:, :2].reshape(-1, 2), thr, dc)
        all_s.append(np.asarray(ps)[~ign])
        all_tp.append(tp[~ign])
        n_gt += len(gb)
    return ap_from_matches(np.concatenate(all_s) if all_s else np.zeros(0),
                           np.concatenate(all_tp) if all_tp else np.zeros(0, bool), n_gt)


def mean_ap_center(preds, gts, thresholds, dontcares=None):
    aps = {float(t): average_precision_center(preds, gts, t, dontcares) for t in thresholds}
    vals = [v for v in aps.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), aps


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(p, y, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def safe_auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def ece(p, y, n_bins=15):
    p, y = np.asarray(p), np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def bootstrap_ci(values_by_unit, stat_fn, n=1000, seed=0, alpha=0.05):
    """Cluster bootstrap: resample units (frames), recompute stat. values_by_unit: list of per-unit data."""
    rng = np.random.default_rng(seed)
    k = len(values_by_unit)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        stats.append(stat_fn([values_by_unit[i] for i in idx]))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def paired_bootstrap_pvalue(a_by_unit, b_by_unit, stat_fn, n=1000, seed=0):
    """One-sided p-value for H0: stat(a) <= stat(b), resampling the SAME units for both arms."""
    rng = np.random.default_rng(seed)
    k = len(a_by_unit)
    observed = stat_fn(a_by_unit) - stat_fn(b_by_unit)
    le = 0
    for _ in range(n):
        idx = rng.integers(0, k, k)
        d = stat_fn([a_by_unit[i] for i in idx]) - stat_fn([b_by_unit[i] for i in idx])
        le += d <= 0
    return float(observed), float((le + 1) / (n + 1))
