"""Risk-controlled fusion (H3, contribution 2).

Paper 01's conformal.py is not in this workspace, so the machinery is implemented here from the
primary sources:
  * split conformal prediction            — Vovk et al. 2005; Lei et al., JASA 2018
  * conformal risk control (monotone loss) — Angelopoulos et al., ICLR 2024
  * weighted conformal under covariate shift — Tibshirani et al., NeurIPS 2019
  * Mondrian (group-conditional) calibration — Vovk 2012

Two guarantees are attached to the fused output:
  A. Localisation: for a detected true object, the disc ‖p − p̂‖ ≤ q̂·σ̂(x) covers the true centre
     with probability ≥ 1 − α. Nonconformity s = ‖p − p̂‖ / σ̂(x).
       vanilla  σ̂ = a + b·range                    (fitted on training scenes)
       conflict σ̂ = (a + b·range)·exp(κᵀφ(x))       φ = residual cross-sensor conflict
     Residual conflict = dispersion of member positions around the fused centre, plus score
     disagreement: when the sensors disagree after arbitration, the disc must grow.
  B. Detection: threshold λ̂ on fused score such that E[frame false-negative rate] ≤ α.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def weighted_quantile(scores, cal_w, test_w, alpha):
    """Tibshirani et al. (2019) weighted quantile for one test point with weight test_w."""
    order = np.argsort(scores)
    s, w = np.asarray(scores)[order], np.asarray(cal_w)[order]
    total = w.sum() + test_w
    cdf = np.cumsum(w) / total
    k = np.searchsorted(cdf, 1 - alpha)
    return np.inf if k >= len(s) else float(s[k])


def residual_conflict(xy_members, present, fused_xy, lam_r, score):
    """φ(x): [log1p(weighted member dispersion / fused scale), score std, n_present-1]."""
    w = present * np.exp(np.clip(lam_r, -20, 20))
    d2 = np.sum((np.nan_to_num(xy_members) - fused_xy[:, None]) ** 2, axis=-1)
    disp = np.sqrt((w * d2).sum(1) / np.maximum(w.sum(1), 1e-12))
    unweighted = np.sqrt(np.where(present, d2, 0).sum(1) / np.maximum(present.sum(1), 1))
    sc = np.where(present, score, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sstd = np.nan_to_num(np.nanstd(sc, axis=1))
    return np.stack([np.log1p(disp), np.log1p(unweighted), sstd, present.sum(1) - 1.0], 1)


class LocalisationConformal:
    def __init__(self, use_conflict=True, range_bins=None):
        self.use_conflict, self.range_bins = use_conflict, range_bins

    def _scale(self, rng, phi):
        base = np.maximum(self.a + self.b * rng, 0.05)
        if self.use_conflict:
            return base * np.exp(np.clip(phi @ self.kappa, -3, 3))
        return base

    def fit_scale(self, err, rng, phi):
        """Fit on TRAINING scenes: base scale by least squares on |err|, κ by regressing log residual ratio."""
        A = np.stack([np.ones_like(rng), rng], 1)
        coef, *_ = np.linalg.lstsq(A, err, rcond=None)
        self.a, self.b = float(max(coef[0], 0.05)), float(max(coef[1], 0.0))
        if self.use_conflict:
            base = np.maximum(self.a + self.b * rng, 0.05)
            reg = LinearRegression(fit_intercept=True).fit(phi, np.log(np.maximum(err, 0.02) / base))
            self.kappa = reg.coef_
        return self

    def calibrate(self, err, rng, phi, alpha):
        s = err / self._scale(rng, phi)
        self.alpha = alpha
        if self.range_bins is None:
            self.q = conformal_quantile(s, alpha)
        else:
            self.q = {}
            idx = np.digitize(rng, self.range_bins) - 1
            for b in range(len(self.range_bins) - 1):
                self.q[b] = conformal_quantile(s[idx == b], alpha)
        self.cal_scores = s
        return self

    def radius(self, rng, phi):
        sc = self._scale(rng, phi)
        if isinstance(self.q, dict):
            idx = np.clip(np.digitize(rng, self.range_bins) - 1, 0, len(self.range_bins) - 2)
            return sc * np.array([self.q[b] for b in idx])
        return sc * self.q

    def evaluate(self, err, rng, phi):
        r = self.radius(rng, phi)
        finite = np.isfinite(r)
        return {"coverage": float(np.mean(err <= r)) if len(err) else float("nan"),
                "mean_radius": float(np.mean(r[finite])) if finite.any() else float("inf"),
                "median_radius": float(np.median(r[finite])) if finite.any() else float("inf"),
                "n": int(len(err)), "nominal": 1 - self.alpha}


class WeightedLocalisationConformal(LocalisationConformal):
    """Covariate-shift correction: calibration weights w(x) = dP_test / dP_cal estimated by a
    probabilistic classifier on frame-level degradation features (unlabelled test frames only)."""

    def fit_ratio(self, cal_frame_feats, target_frame_feats, clip=(1e-3, 1e3)):
        X = np.concatenate([cal_frame_feats, target_frame_feats])
        y = np.concatenate([np.zeros(len(cal_frame_feats)), np.ones(len(target_frame_feats))])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        self.mu, self.sd, self.clip = mu, sd, clip
        self.clf = LogisticRegression(max_iter=2000, C=0.5).fit((X - mu) / sd, y)
        self.prior = len(cal_frame_feats) / len(target_frame_feats)
        return self

    def ratio(self, frame_feats):
        p = self.clf.predict_proba((frame_feats - self.mu) / self.sd)[:, 1]
        return np.clip(p / np.maximum(1 - p, 1e-9) * self.prior, *self.clip)

    def evaluate_weighted(self, cal_err, cal_rng, cal_phi, cal_ff, err, rng, phi, ff, alpha):
        s_cal = cal_err / self._scale(cal_rng, cal_phi)
        w_cal = self.ratio(cal_ff)
        w_test = self.ratio(ff)
        ess = float(w_cal.sum() ** 2 / np.maximum((w_cal ** 2).sum(), 1e-12))
        # all test points in a frame share a weight; quantile per unique weight for speed
        uniq, inv = np.unique(np.round(w_test, 6), return_inverse=True)
        qs = np.array([weighted_quantile(s_cal, w_cal, wt, alpha) for wt in uniq])
        r = self._scale(rng, phi) * qs[inv]
        finite = np.isfinite(r)
        return {"coverage": float(np.mean(err <= r)) if len(err) else float("nan"),
                "mean_radius": float(np.mean(r[finite])) if finite.any() else float("inf"),
                "frac_infinite": float(np.mean(~finite)), "ess": ess, "n_cal": int(len(s_cal)),
                "n": int(len(err)), "nominal": 1 - alpha}


def crc_threshold(frame_losses_fn, n_frames, alpha, grid, B=1.0):
    """Conformal risk control: largest λ with (n/(n+1))·R̂(λ) + B/(n+1) ≤ α.
    frame_losses_fn(λ) -> per-frame losses in [0, B], non-increasing as λ decreases."""
    n = n_frames
    best = None
    for lam in sorted(grid, reverse=True):
        r = float(np.mean(frame_losses_fn(lam)))
        if n / (n + 1) * r + B / (n + 1) <= alpha:
            best = lam
            break
    return grid[0] if best is None else best


def frame_fnr(frame_dets, gts, thr, lam):
    """Per-frame FNR at score threshold lam; frames without GT excluded by the caller."""
    losses = []
    for (xy, s, _), g in zip(frame_dets, gts):
        if len(g) == 0:
            continue
        keep = xy[s >= lam]
        if len(keep) == 0:
            losses.append(1.0)
            continue
        d = np.linalg.norm(g[:, None] - keep[None], axis=-1)
        # one-to-one greedy by distance
        matched, used = 0, set()
        for gi in np.argsort(d.min(1)):
            j = [k for k in np.argsort(d[gi]) if k not in used and d[gi, k] <= thr]
            if j:
                used.add(j[0])
                matched += 1
        losses.append(1 - matched / len(g))
    return np.asarray(losses)


def frame_false_positives(frame_dets, gts, thr, lam):
    fps = []
    for (xy, s, _), g in zip(frame_dets, gts):
        keep = xy[s >= lam]
        if len(g) == 0:
            fps.append(len(keep))
            continue
        d = np.linalg.norm(keep[:, None] - g[None], axis=-1) if len(keep) else np.zeros((0, len(g)))
        fps.append(int((d.min(1) > thr).sum()) if len(keep) else 0)
    return np.asarray(fps)
