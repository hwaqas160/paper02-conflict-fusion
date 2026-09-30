"""External late-fusion baselines (Registration 2, R2-B), all on the same cached candidates as our arms.

Every baseline sees the same frozen detections, the same train/val scenes, the same fitted PositionPrior and the
same operating-threshold rule (F1 on clear train frames) as the per-object router, so differences are due to the
fusion method only. Implemented from the published descriptions:

    ProbEn        Chen et al., ECCV 2022 -- Bayesian product of per-detector calibrated posteriors (conditional
                  independence), variance-weighted box fusion.
    DS            Dempster-Shafer evidential fusion with per-sensor reliability discounting and an explicit
                  "missed by an expected sensor" mass (evidential late fusion, e.g. Chavez-Garcia & Aycard 2016).
    CLOCs         Pang et al., IROS 2020 -- learned re-scoring of candidates from detector scores and pairwise
                  geometric consistency; keeps the LiDAR box when present.
    SetFusion     black-box transformer over each frame's candidate set, given exactly our per-object inputs,
                  regressing score and position directly (no interpretable fusion rule) -- tests whether our
                  structured rule matters, not just the inputs.
    MoE           per-frame mixture of K constant fusion-parameter experts gated by frame degradation features.
    Oracle        GT-informed per-object selector (upper bound, not deployable): best member position, perfect
                  existence among candidates the detectors produced.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .arbitrate import (ALL_BLOCKS, CAM_DET, LID_DET, M_CAM, M_LID, M_RAD, MODALITY_OF_PROBE, PositionPrior,
                        Standardizer, _frame_feat_matrix, _logit, fusion_rule, object_inputs, targets)
from .degradation import PROBE_NAMES

RAD_DET_ = None


# ------------------------------------------------------------------------------------------------ shared helpers

def masked_cand(c, drop):
    """Candidate dict with whole sensors removed per frame (same semantics as Arbiter._mask_inputs)."""
    if drop is None:
        return c
    from .arbitrate import RAD_DET
    dm = drop[c["frame"]]
    c2 = dict(c)
    c2["present"] = c["present"] & ~dm
    c2["fov"] = c["fov"] * (~dm)
    pr = c["probes"].copy()
    for prefix, m in MODALITY_OF_PROBE.items():
        cols = [i for i, n in enumerate(PROBE_NAMES) if n.startswith(prefix)]
        pr[np.ix_(dm[:, m], cols)] = np.nan
    c2["probes"] = pr
    df = c["det_feats"].copy()
    for cols, m in ((CAM_DET, M_CAM), (RAD_DET, M_RAD), (LID_DET, M_LID)):
        df[np.ix_(dm[:, m], cols)] = 0
    c2["det_feats"] = df
    k = c["conflict"].copy()
    k[dm[:, M_LID] | dm[:, M_RAD], 0] = 0
    k[dm[:, M_CAM] | dm[:, M_LID], 1:3] = 0
    k[dm[:, M_CAM] | dm[:, M_RAD], 3:5] = 0
    k[:, 6] = c2["present"].sum(1)
    c2["conflict"] = k
    return c2


def dropout_drops(n_frames, cfg, seed=0):
    """The SAME whole-sensor dropout augmentation our router is trained with (arbitration.dropout_modality_p,
    arbitration.dropout_variants): None (clean) + K random per-frame sensor-drop masks. Applied to every learned
    baseline so that robustness to a missing sensor is not an unfair advantage of our arm."""
    a = cfg["arbitration"]
    rng = np.random.default_rng(seed)
    out = [None]
    for _ in range(a["dropout_variants"]):
        d = rng.random((n_frames, 3)) < a["dropout_modality_p"]
        d[d.all(1)] = False
        out.append(d)
    return out


def prior_xy(prior, c, present):
    """Inverse-variance (fitted prior) fusion of bias-corrected member positions; numpy."""
    xy, lam_r, lam_a = prior.apply(c)
    rng = np.hypot(xy[..., 0], xy[..., 1])
    az = np.arctan2(xy[..., 1], xy[..., 0])
    w_r = present * np.exp(np.clip(lam_r, -20, 20))
    w_a = present * np.exp(np.clip(lam_a, -20, 30))
    r = (w_r * rng).sum(1) / np.maximum(w_r.sum(1), 1e-12)
    a = np.arctan2((w_a * np.sin(az)).sum(1), (w_a * np.cos(az)).sum(1) + 1e-12)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1).astype(np.float32), xy


def _out(score, xy, present):
    any_p = present.any(1)
    score = np.where(any_p, score, 0.0)
    n = len(score)
    return {"score": score.astype(np.float32), "xy": xy, "present_any": any_p,
            "harm_prob": np.zeros((n, 3), np.float32)}


class PlattCalibrators:
    """Per-sensor logistic calibration of detector scores against 'this member is a true positive'."""

    def fit(self, c, cfg):
        thr = np.where(c["cls"] == 0, cfg["matching"]["center_dist_m"]["Car"], cfg["matching"]["center_dist_m"]["Pedestrian"])
        self.ab = {}
        for m in range(3):
            p = c["present"][:, m]
            if p.sum() < 50:
                self.ab[m] = (1.0, 0.0)
                continue
            d = np.linalg.norm(np.nan_to_num(c["xy"][p, m]) - np.nan_to_num(c["gt_xy"][p], nan=1e6), axis=1)
            y = ((c["y"][p] > 0) & (d <= thr[p])).astype(int)
            x = _logit(c["score"][p, m])[:, None]
            if y.min() == y.max():
                self.ab[m] = (1.0, 0.0)
                continue
            lr = LogisticRegression(C=10.0, max_iter=500).fit(x, y)
            self.ab[m] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    def logit(self, c):
        out = np.zeros(c["score"].shape, np.float32)
        for m, (a, b) in self.ab.items():
            out[:, m] = a * _logit(c["score"][:, m]) + b
        return out


# ------------------------------------------------------------------------------------------------ ProbEn

class ProbEn:
    arm = "proben"

    def fit(self, train, val, cfg):
        c = train["cand"]
        self.prior = PositionPrior().fit(train)
        self.cal = PlattCalibrators().fit(c, cfg)
        y, _ = targets(train, cfg)
        pi = np.clip(y[c["present"].any(1)].mean(), 1e-3, 1 - 1e-3)
        self.logit_pi = float(np.log(pi / (1 - pi)))
        return self

    def n_params(self):
        return 7

    def predict(self, tbl, drop=None):
        c = masked_cand(tbl["cand"], drop)
        p = c["present"]
        L = self.cal.logit(c)
        z = self.logit_pi + (p * (L - self.logit_pi)).sum(1)
        xy, _ = prior_xy(self.prior, c, p)
        return _out(1 / (1 + np.exp(-z)), xy, p)


# ------------------------------------------------------------------------------------------------ Dempster-Shafer

class DempsterShafer:
    arm = "dempster_shafer"

    def fit(self, train, val, cfg):
        c = train["cand"]
        self.prior = PositionPrior().fit(train)
        self.cal = PlattCalibrators().fit(c, cfg)
        yv, _ = targets(val, cfg)
        grid = [0.3, 0.5, 0.7, 0.85, 1.0]
        miss = [0.0, 0.1, 0.25, 0.4]
        self.r, self.s = np.ones(3), np.zeros(3)
        best = math.inf
        vc = val["cand"]
        valid = vc["present"].any(1)
        for _ in range(2):  # coordinate descent over sensors
            for m in range(3):
                for r in grid:
                    for s in miss:
                        rr, ss = self.r.copy(), self.s.copy()
                        rr[m], ss[m] = r, s
                        q = np.clip(self._belief(vc, rr, ss), 1e-4, 1 - 1e-4)
                        nll = -np.mean((yv * np.log(q) + (1 - yv) * np.log(1 - q))[valid])
                        if nll < best:
                            best, self.r, self.s = nll, rr, ss
        return self

    def _belief(self, c, r, s):
        q = 1 / (1 + np.exp(-self.cal.logit(c)))
        p, f = c["present"], c["fov"] > 0
        mO, mN, mT = np.zeros(len(q)), np.zeros(len(q)), np.ones(len(q))
        for m in range(3):
            o = np.where(p[:, m], r[m] * q[:, m], 0.0)
            n = np.where(p[:, m], r[m] * (1 - q[:, m]), np.where(f[:, m], s[m], 0.0))
            t = 1 - o - n
            k = mO * n + mN * o  # conflict
            nO = mO * o + mO * t + mT * o
            nN = mN * n + mN * t + mT * n
            nT = mT * t
            z = np.maximum(1 - k, 1e-9)
            mO, mN, mT = nO / z, nN / z, nT / z
        return np.clip(mO + 0.5 * mT, 0.0, 1.0)  # pignistic probability of "object"

    def n_params(self):
        return 6 + 6

    def predict(self, tbl, drop=None):
        c = masked_cand(tbl["cand"], drop)
        xy, _ = prior_xy(self.prior, c, c["present"])
        return _out(self._belief(c, self.r, self.s), xy, c["present"])


# ------------------------------------------------------------------------------------------------ CLOCs

class _MLP(nn.Module):
    def __init__(self, d, h=64, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h),
                                 nn.GELU(), nn.Linear(h, out))

    def forward(self, x):
        return self.net(x)


def _train_rows(model, X, y, Xv, yv, lr=2e-3, epochs=30, bs=8192, seed=0, patience=5):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    Xvt, yvt = torch.as_tensor(Xv), torch.as_tensor(yv, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best, state, bad = math.inf, None, 0
    for _ in range(epochs):
        model.train()
        for i in np.array_split(rng.permutation(len(X)), max(1, len(X) // bs)):
            loss = F.binary_cross_entropy_with_logits(model(Xt[i]).squeeze(1), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(model(Xvt).squeeze(1), yvt))
        if vl < best - 1e-4:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


class CLOCs:
    arm = "clocs"

    @staticmethod
    def _x(c):
        p = c["present"].astype(np.float32)
        L = np.where(c["present"], _logit(c["score"]), 0.0)
        return np.concatenate([L, p, c["fov"], c["conflict"], np.stack([c["cls"] == 0, c["cls"] == 1], 1)],
                              1).astype(np.float32)

    def fit(self, train, val, cfg):
        self.prior = PositionPrior().fit(train)
        c, vc = train["cand"], val["cand"]
        Xs, ys = [], []
        for d in dropout_drops(len(train["frames"]), cfg):
            cd = masked_cand(c, d)
            m = cd["present"].any(1)
            Xs.append(self._x(cd)[m])
            ys.append(targets(train, cfg, d)[0][m])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        mv = vc["present"].any(1)
        Xv = self._x(vc)
        self.std = Standardizer(X.mean(0), X.std(0) + 1e-3)
        yv, _ = targets(val, cfg)
        self.net = _train_rows(_MLP(X.shape[1]), self.std(X), y, self.std(Xv)[mv], yv[mv])
        return self

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def predict(self, tbl, drop=None):
        c = masked_cand(tbl["cand"], drop)
        z = self.net(torch.as_tensor(self.std(self._x(c)))).squeeze(1).numpy()
        xy, members = prior_xy(self.prior, c, c["present"])
        lid = c["present"][:, M_LID]
        xy[lid] = members[lid, M_LID]  # CLOCs keeps the 3-D (LiDAR) box
        return _out(1 / (1 + np.exp(-z)), xy, c["present"])


# ------------------------------------------------------------------------------------------------ Set transformer

class _SetNet(nn.Module):
    def __init__(self, d, h=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d, h)
        enc = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(h, 3)  # logit, dx, dy

    def forward(self, x, pad):
        return self.head(self.enc(self.inp(x), src_key_padding_mask=pad))


class SetFusion:
    """Black-box learned fusion: same per-object inputs as ours + member geometry, no structured rule."""
    arm = "set_transformer"

    def _x(self, c):
        xy, lam_r, _ = self.prior.apply(c)
        geo = np.concatenate([np.nan_to_num(xy).reshape(len(xy), -1) / 50.0, lam_r / 10.0], 1)
        return np.concatenate([object_inputs(c, ALL_BLOCKS), c["present"].astype(np.float32), c["fov"], geo],
                              1).astype(np.float32)

    def _batches(self, tbl, c, X, frames):
        order = np.argsort(c["frame"], kind="stable")
        fs = c["frame"][order]
        bounds = np.searchsorted(fs, np.arange(len(tbl["frames"]) + 1))
        for fr in frames:
            idx = [order[bounds[f]:bounds[f + 1]] for f in fr]
            idx = [i for i in idx if len(i)]
            if not idx:
                continue
            L = max(len(i) for i in idx)
            xb = np.zeros((len(idx), L, X.shape[1]), np.float32)
            pad = np.ones((len(idx), L), bool)
            for k, i in enumerate(idx):
                xb[k, :len(i)] = X[i]
                pad[k, :len(i)] = False
            yield idx, torch.as_tensor(xb), torch.as_tensor(pad)

    def fit(self, train, val, cfg, epochs=35, bs_frames=32, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.prior = PositionPrior().fit(train)
        vc = val["cand"]
        variants = []
        for d in dropout_drops(len(train["frames"]), cfg, seed):
            cd = masked_cand(train["cand"], d)
            y_d, thr_d = targets(train, cfg, d)
            variants.append([cd, self._x(cd), y_d, thr_d, prior_xy(self.prior, cd, cd["present"])[0]])
        allX = np.concatenate([v[1] for v in variants])
        self.std = Standardizer(np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-3)
        del allX
        for v in variants:
            v[1] = np.nan_to_num(self.std(v[1]))
        Xv = np.nan_to_num(self.std(self._x(vc)))
        yv, thrv = targets(val, cfg)
        basev, _ = prior_xy(self.prior, vc, vc["present"])
        self.net = _SetNet(variants[0][1].shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        nF = len(train["frames"])
        best, state, bad = math.inf, None, 0

        def loss_on(tbl_, c_, X_, y_, thr_, base_, frames):
            tot, n = 0.0, 0
            for idx, xb, pad in self._batches(tbl_, c_, X_, frames):
                out = self.net(xb, pad)
                ii = np.concatenate(idx)
                o = torch.cat([out[k, :len(i)] for k, i in enumerate(idx)])
                valid = torch.as_tensor(c_["present"][ii].any(1))
                yt = torch.as_tensor(y_[ii], dtype=torch.float32)
                l = F.binary_cross_entropy_with_logits(o[valid, 0], yt[valid])
                pos = valid & (yt > 0)
                if pos.any():
                    pxy = torch.as_tensor(base_[ii]) + 2.0 * o[:, 1:]
                    err = torch.linalg.norm(pxy[pos] - torch.as_tensor(np.nan_to_num(c_["gt_xy"][ii]))[pos], dim=1)
                    l = l + 0.5 * F.smooth_l1_loss(err / torch.as_tensor(thr_[ii])[pos], torch.zeros_like(err), beta=0.5)
                yield l

        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(nF)
            cd, Xd, yd, thrd, based = variants[ep % len(variants)]
            for l in loss_on(train, cd, Xd, yd, thrd, based, np.array_split(perm, max(1, nF // bs_frames))):
                opt.zero_grad()
                l.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vls = [float(l) for l in loss_on(val, vc, Xv, yv, thrv, basev,
                                                  np.array_split(np.arange(len(val["frames"])), max(1, len(val["frames"]) // 64)))]
            vl = float(np.mean(vls))
            if vl < best - 1e-4:
                best, state, bad = vl, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= 4:
                    break
        self.net.load_state_dict(state)
        return self

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def predict(self, tbl, drop=None):
        self.net.eval()
        c = masked_cand(tbl["cand"], drop)
        X = np.nan_to_num(self.std(self._x(c)))
        base, _ = prior_xy(self.prior, c, c["present"])
        z = np.zeros(len(X), np.float32)
        xy = base.copy()
        frames = np.array_split(np.arange(len(tbl["frames"])), max(1, len(tbl["frames"]) // 64))
        for idx, xb, pad in self._batches(tbl, c, X, frames):
            out = self.net(xb, pad)
            for k, i in enumerate(idx):
                o = out[k, :len(i)].numpy()
                z[i] = o[:, 0]
                xy[i] = base[i] + 2.0 * o[:, 1:]
        return _out(1 / (1 + np.exp(-z)), xy, c["present"])


# ------------------------------------------------------------------------------------------------ MoE (per frame)

class MoE:
    """K constant fusion-parameter experts, softmax-gated by standardised frame-level degradation features,
    through the SAME fusion rule as our arms (so it isolates per-frame gating vs per-object routing)."""
    arm = "moe_frame"

    def fit(self, train, val, cfg, K=4, epochs=40, seed=0):
        torch.manual_seed(seed)
        self.prior = PositionPrior().fit(train)
        ff = _frame_feat_matrix(train)
        self.std = Standardizer(ff.mean(0), ff.std(0) + 1e-3)
        self.gate = nn.Linear(ff.shape[1], K)
        self.experts = nn.Parameter(0.01 * torch.randn(K, 19))
        params = list(self.gate.parameters()) + [self.experts]
        opt = torch.optim.AdamW(params, lr=2e-2)
        drops = dropout_drops(len(train["frames"]), cfg, seed)
        Ts = [(self._tensors(train, d), targets(train, cfg, d)[0]) for d in drops]
        Tv, yv = self._tensors(val), targets(val, cfg)[0]
        best, state, bad = math.inf, None, 0
        for it in range(epochs * 10):
            T, y = Ts[it % len(Ts)]
            l = self._loss(T, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            with torch.no_grad():
                vl = float(self._loss(Tv, yv))
            if vl < best - 1e-5:
                best, state, bad = vl, (copy.deepcopy(self.gate.state_dict()), self.experts.detach().clone()), 0
            else:
                bad += 1
                if bad >= 30:
                    break
        self.gate.load_state_dict(state[0])
        self.experts = nn.Parameter(state[1])
        return self

    def _tensors(self, tbl, drop=None):
        c = masked_cand(tbl["cand"], drop)
        xy, lam_r, lam_a = self.prior.apply(c)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
        ff = np.nan_to_num(self.std(_frame_feat_matrix(tbl)))
        return {"ff": t(ff)[t(c["frame"], torch.long)], "present": t(c["present"], torch.bool), "fov": t(c["fov"]),
                "logit": t(np.where(c["present"], _logit(c["score"]), 0.0)), "xy": t(xy), "lam_r": t(lam_r),
                "lam_a": t(lam_a)}

    def _raw(self, T):
        g = torch.softmax(self.gate(T["ff"]), 1)
        return g @ self.experts

    def _loss(self, T, y):
        z, _, _, _ = fusion_rule(self._raw(T), T["present"], T["fov"], T["logit"], T["xy"], T["lam_r"], T["lam_a"])
        v = T["present"].any(1)
        return F.binary_cross_entropy_with_logits(z[v], torch.as_tensor(y)[v])

    def n_params(self):
        return sum(p.numel() for p in self.gate.parameters()) + self.experts.numel()

    @torch.no_grad()
    def predict(self, tbl, drop=None):
        T = self._tensors(tbl, drop)
        z, xy, _, _ = fusion_rule(self._raw(T), T["present"], T["fov"], T["logit"], T["xy"], T["lam_r"], T["lam_a"])
        return _out(torch.sigmoid(z).numpy(), xy.numpy(), T["present"].numpy())


# ------------------------------------------------------------------------------------------------ Oracle

class Oracle:
    """Upper bound on any rule that selects among the detections actually produced: existence = GT-backed,
    position = the member closest to GT. Uses labels; reported only as headroom."""
    arm = "oracle"

    def fit(self, train, val, cfg):
        self.prior = PositionPrior().fit(train)
        return self

    def n_params(self):
        return 0

    def predict(self, tbl, drop=None):
        c = masked_cand(tbl["cand"], drop)
        p = c["present"]
        xy, members = prior_xy(self.prior, c, p)
        d = np.linalg.norm(members - np.nan_to_num(c["gt_xy"], nan=1e6)[:, None], axis=-1)
        d = np.where(p, d, np.inf)
        best = d.argmin(1)
        dmin = d.min(1)
        gt = (c["y"] > 0) & np.isfinite(dmin)
        xy[gt] = members[gt, best[gt]]
        # one detection per GT object (duplicates would be false positives): keep the closest candidate
        idx = np.nonzero(gt)[0]
        key = np.stack([c["frame"][idx], np.round(np.nan_to_num(c["gt_xy"][idx, 0]), 2),
                        np.round(np.nan_to_num(c["gt_xy"][idx, 1]), 2)], 1)
        order = np.lexsort((dmin[idx], key[:, 2], key[:, 1], key[:, 0]))
        ks = key[order]
        first = np.ones(len(order), bool)
        first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
        keep = np.zeros(len(c["y"]), bool)
        keep[idx[order][first]] = True
        score = np.where(keep, 1.0 - 0.01 * np.minimum(dmin, 50) / 50, 0.0)
        return _out(score, xy, p)


BASELINES = {"proben": ProbEn, "dempster_shafer": DempsterShafer, "clocs": CLOCs, "set_transformer": SetFusion,
             "moe_frame": MoE, "oracle": Oracle}
