"""H1 — the harm case: when does adding a modality make the fused output worse?

Counterfactual per object: the same frozen detections and the same fusion model are evaluated with
all sensors, and with sensor m removed entirely (its detections, probes and field-of-view flag).
    harm_m(i) = m present in candidate i  AND  decision(all) wrong  AND  decision(all \\ m) right
    help_m(i) = m present                 AND  decision(all) right  AND  decision(all \\ m) wrong
Soft variant: per-candidate log-loss increase from adding m greater than `soft_margin_nats`.
"""
import numpy as np

from .arbitrate import targets
from .common import MODALITIES
from .evaluate import decisions


def counterfactual_predictions(arb, tbl):
    preds = {"all": arb.predict(tbl)}
    n_frames = len(tbl["frames"])
    for m, name in enumerate(MODALITIES):
        drop = np.zeros((n_frames, 3), bool)
        drop[:, m] = True
        preds[f"minus_{name}"] = arb.predict(tbl, drop)
    return preds


def harm_labels(arb, tbl, tau, cfg, preds=None):
    """(N,3) int8: 1 harm, 0 no harm, -1 modality absent. Also returns help labels and soft harm."""
    preds = preds or counterfactual_predictions(arb, tbl)
    c = tbl["cand"]
    correct_all = decisions(tbl, preds["all"], tau, cfg)
    harm = np.full((len(c["y"]), 3), -1, np.int8)
    helpful = np.full((len(c["y"]), 3), -1, np.int8)
    soft = np.full((len(c["y"]), 3), -1, np.int8)
    y_all, _ = targets(tbl, cfg)
    eps = 1e-4
    ll = lambda p, y: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    loss_all = ll(preds["all"]["score"], y_all)
    for m, name in enumerate(MODALITIES):
        pm = preds[f"minus_{name}"]
        correct_minus = decisions(tbl, pm, tau, cfg)
        present = c["present"][:, m] & ~c["ignore"]
        harm[present, m] = (~correct_all & correct_minus)[present]
        helpful[present, m] = (correct_all & ~correct_minus)[present]
        drop = np.zeros((len(tbl["frames"]), 3), bool)
        drop[:, m] = True
        y_minus, _ = targets(tbl, cfg, drop)
        # compare against the best achievable label without m, so the soft score measures fusion damage
        loss_minus = ll(pm["score"], y_minus)
        soft[present, m] = ((loss_all - loss_minus) > cfg["harm"]["soft_margin_nats"])[present]
    return harm, helpful, soft, preds


def harm_summary(harm, helpful, cls, frame_idx, class_id=0, n_boot=1000, seed=0, ranges=None, bins=None):
    """Pooled and per-modality harm / help rates with frame-cluster bootstrap CIs."""
    m_cls = cls == class_id
    out = {}

    def rate(h):
        v = h[h >= 0]
        return float(v.mean()) if len(v) else float("nan")

    def boot(h_mat):
        frames = np.unique(frame_idx[m_cls])
        if len(frames) < 2:
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(seed)
        order = np.argsort(frame_idx, kind="stable")
        fsorted = frame_idx[order]
        stats = []
        for _ in range(n_boot):
            pick = rng.choice(frames, len(frames))
            idx = np.concatenate([order[np.searchsorted(fsorted, f):np.searchsorted(fsorted, f, "right")] for f in pick])
            idx = idx[m_cls[idx]]
            v = h_mat[idx]
            v = v[v >= 0]
            stats.append(v.mean() if len(v) else np.nan)
        stats = np.asarray(stats)
        stats = stats[~np.isnan(stats)]
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))] if len(stats) else [float("nan")] * 2

    pooled = harm[m_cls]
    out["pooled"] = {"harm_rate": rate(pooled), "harm_ci": boot(harm), "help_rate": rate(helpful[m_cls]),
                     "n_pairs": int((pooled >= 0).sum())}
    for m, name in enumerate(MODALITIES):
        h = harm[m_cls, m]
        hm = np.full_like(harm, -1)
        hm[:, m] = harm[:, m]
        out[name] = {"harm_rate": rate(h), "harm_ci": boot(hm), "help_rate": rate(helpful[m_cls, m]),
                     "n_present": int((h >= 0).sum())}
    if ranges is not None and bins is not None:
        out["by_range"] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            rb = m_cls & (ranges >= lo) & (ranges < hi)
            out["by_range"][f"{lo}-{hi}"] = {name: rate(harm[rb, m]) for m, name in enumerate(MODALITIES)}
    return out
