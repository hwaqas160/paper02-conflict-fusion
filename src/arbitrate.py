"""Degradation-conditioned arbitration over frozen detectors (the method), and all routing baselines.

Every arm shares ONE differentiable late-fusion rule; arms differ only in what the *router* that
produces the rule's parameters is allowed to look at:

    existence logit   z = b + Σ_m p_m (a_m·ℓ_m + c_m) + Σ_m (1 − p_m) f_m u_m
    fused range       r̂ = Σ_m p_m π^r_m r_m / Σ_m p_m π^r_m ,   π^r_m = exp(λ^r_m + Δ^r_m)
    fused bearing     θ̂ = circular mean with π^θ_m = exp(λ^θ_m + Δ^θ_m)

p = presence, ℓ = detector logit, f = in-field-of-view, λ = detector-prior log-precision
(camera: its own monocular range sigma), and (a ≥ 0, c, u ≤ 0, Δ^r, Δ^θ, b) come from the router.
Absent-but-visible modalities contribute *negative* evidence u — "the camera should have seen it".

Routers (arms):
    fixed_clear   constants, fitted on clear only                (status quo; defines harm)
    fixed         constants, fitted on all training conditions   (H2 arm a)
    frame_cls     MLP(predicted weather-class probabilities)     (H2 arm b, LRC-WeatherNet style)
    frame_oracle  MLP(true condition one-hot, severity)          (H2 arm b′, upper bound per-frame)
    frame_feats   MLP(frame-level degradation features)          (stronger continuous per-frame arm)
    confidence    MLP(detector logits, presence, class)          (H2 arm c)
    object        MLP(per-object probes, detector feats, conflict, presence, logits, frame feats)
                  + DeepSets frame context                        (H2 arm d, ours)
Each router also emits per-modality harm logits (auxiliary), so harm-prediction AUROC is
comparable across arms.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .candidates import CONFLICT_FEATURES, DET_FEATURES, M_CAM, M_LID, M_RAD
from .degradation import FRAME_FEATURES, PROBE_NAMES

COND_GROUPS = ["clear", "fog", "rain", "snow", "night"]
N_PARAMS = 3 * 5 + 1
PROBE_GROUPS = {
    "geometry": [n for n in PROBE_NAMES if n.startswith("g_")],
    "lidar_probes": [n for n in PROBE_NAMES if n.startswith("l_")],
    "camera_probes": [n for n in PROBE_NAMES if n.startswith("c_")],
    "radar_probes": [n for n in PROBE_NAMES if n.startswith("r_")],
}
MODALITY_OF_PROBE = {"l_": M_LID, "c_": M_CAM, "r_": M_RAD}
CAM_DET = [i for i, n in enumerate(DET_FEATURES) if n.startswith("cam_")]
RAD_DET = [i for i, n in enumerate(DET_FEATURES) if n.startswith("rad_")]
LID_DET = [i for i, n in enumerate(DET_FEATURES) if n.startswith("lid_")]
ALL_BLOCKS = ["geometry", "lidar_probes", "camera_probes", "radar_probes", "det_feats", "conflict", "confidence",
              "frame_feats"]


# ------------------------------------------------------------------------------------------------
# Data assembly

def merge_tables(tables):
    """Concatenate per-condition tables; frame indices become global."""
    cand, gts, frames, off = {}, [], [], 0
    for t in tables:
        c = dict(t["cand"])
        c["frame"] = c["frame"] + off
        for k, v in c.items():
            cand.setdefault(k, []).append(v)
        gts += t["gts"]
        frames += t["frames"]
        off += len(t["frames"])
    return {"cand": {k: np.concatenate(v) for k, v in cand.items()}, "gts": gts, "frames": frames}


def subset_table(tbl, frame_mask):
    """Keep frames where frame_mask (per frame bool) is True; reindex."""
    keep_frames = np.nonzero(frame_mask)[0]
    remap = -np.ones(len(tbl["frames"]), np.int64)
    remap[keep_frames] = np.arange(len(keep_frames))
    m = frame_mask[tbl["cand"]["frame"]]
    cand = {k: v[m] for k, v in tbl["cand"].items()}
    cand["frame"] = remap[cand["frame"]]
    return {"cand": cand, "gts": [tbl["gts"][i] for i in keep_frames], "frames": [tbl["frames"][i] for i in keep_frames]}


def restrict_class(tbl, class_id):
    m = tbl["cand"]["cls"] == class_id
    cand = {k: v[m] for k, v in tbl["cand"].items()}
    return {"cand": cand, "gts": tbl["gts"], "frames": tbl["frames"]}


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x):
        return (x - self.mean) / self.std


class PositionPrior:
    """Per-sensor measurement model fitted on TRAINING positives (never cal/test):
    range bias(r), range σ(r) and lateral σ(r), each a + b·r from robust (1.4826·MAD) per-bin statistics.
    Measured on clear Boreas train frames the camera's bearing error is 0.8–2.5° and its self-reported range σ
    under-states the error 3–7×; radar has a −1 m near-face range bias. Hand-set priors were wrong, so they
    are estimated."""

    BINS = np.array([0, 10, 20, 30, 45, 70])

    def fit(self, tbl):
        c = tbl["cand"]
        pos = (c["y"] > 0) & ~c["ignore"]
        g = np.nan_to_num(c["gt_xy"][pos])
        rg, ag = np.hypot(g[:, 0], g[:, 1]), np.arctan2(g[:, 1], g[:, 0])
        self.params = {}
        for m in range(3):
            ok = c["present"][pos, m]
            defaults = {M_CAM: (0.0, 0.0, 0.3, 0.08, 0.3, 0.01), M_LID: (0.0, 0.0, 0.15, 0.003, 0.05, 0.002),
                        M_RAD: (-1.0, 0.0, 0.5, 0.015, 0.2, 0.01)}[m]
            if ok.sum() < 30:
                self.params[m] = defaults
                continue
            xy = c["xy"][pos, m][ok]
            r, a = np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])
            dr, lat, rr = r - rg[ok], wrap(a - ag[ok]) * rg[ok], rg[ok]
            centres, bias, s_r, s_l, w = [], [], [], [], []
            for lo, hi in zip(self.BINS[:-1], self.BINS[1:]):
                b = (rr >= lo) & (rr < hi)
                if b.sum() < 8:
                    continue
                med = np.median(dr[b])
                centres.append(np.median(rr[b]))
                bias.append(med)
                s_r.append(1.4826 * np.median(np.abs(dr[b] - med)))
                s_l.append(1.4826 * np.median(np.abs(lat[b] - np.median(lat[b]))))
                w.append(b.sum())
            if len(centres) < 2:
                self.params[m] = defaults
                continue
            fitl = lambda y: np.polyfit(centres, y, 1, w=np.sqrt(w))
            (bb, ba), (sb, sa), (lb, la) = fitl(bias), fitl(s_r), fitl(s_l)
            self.params[m] = (ba, bb, max(sa, 0.05), max(sb, 0.0), max(la, 0.03), max(lb, 0.0))
        return self

    def apply(self, c):
        """Bias-corrected member xy and log-precisions (range, bearing)."""
        xy = np.nan_to_num(c["xy"]).astype(np.float64)
        rng = np.hypot(xy[..., 0], xy[..., 1])
        az = np.arctan2(xy[..., 1], xy[..., 0])
        lam_r, lam_a = np.zeros(rng.shape), np.zeros(rng.shape)
        out = xy.copy()
        for m, (ba, bb, sa, sb, la, lb) in self.params.items():
            r = rng[:, m]
            rc = np.maximum(r - (ba + bb * r), 0.5)
            out[:, m, 0], out[:, m, 1] = rc * np.cos(az[:, m]), rc * np.sin(az[:, m])
            sig_r = np.maximum(sa + sb * rc, 0.05)
            sig_a = np.maximum(la + lb * rc, 0.03) / np.maximum(rc, 1.0)
            lam_r[:, m], lam_a[:, m] = -2 * np.log(sig_r), -2 * np.log(sig_a)
        out[~c["present"]] = 0.0
        return out.astype(np.float32), lam_r.astype(np.float32), lam_a.astype(np.float32)

    def summary(self):
        names = ("camera", "lidar", "radar")
        return {names[m]: dict(zip(["bias_a", "bias_b", "sig_r_a", "sig_r_b", "sig_lat_a", "sig_lat_b"], map(float, p)))
                for m, p in self.params.items()}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def object_inputs(c, blocks):
    parts = []
    probes = c["probes"].copy()
    missing = np.isnan(probes)
    probes = np.nan_to_num(probes, nan=0.0)
    for g in ("geometry", "lidar_probes", "camera_probes", "radar_probes"):
        if g in blocks:
            idx = [PROBE_NAMES.index(n) for n in PROBE_GROUPS[g]]
            parts.append(probes[:, idx])
            if g in ("camera_probes", "radar_probes"):
                parts.append(missing[:, idx].any(1, keepdims=True).astype(np.float32))
    if "det_feats" in blocks:
        parts.append(c["det_feats"])
    if "conflict" in blocks:
        parts.append(c["conflict"])
    if "confidence" in blocks:
        parts.append(np.where(c["present"], _logit(c["score"]), 0.0))
        parts.append(c["present"].astype(np.float32))
        parts.append(c["fov"])
        parts.append(np.stack([c["cls"] == 0, c["cls"] == 1], 1).astype(np.float32))
    if "frame_feats" in blocks:
        parts.append(c["frame_feats"])
    return np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((len(c["y"]), 0), np.float32)


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def condition_inputs(frames, frame_idx):
    onehot = np.zeros((len(frames), len(COND_GROUPS) + 1), np.float32)
    for i, f in enumerate(frames):
        onehot[i, COND_GROUPS.index(f["condition"])] = 1
        onehot[i, -1] = f["severity"] / 3.0
    return onehot[frame_idx]


class WeatherClassifier:
    """Per-frame weather classifier on frame-level features only (the LRC-WeatherNet-style baseline)."""

    def fit(self, frame_feats, groups):
        self.classes_ = sorted(set(groups))
        self.std = Standardizer(frame_feats.mean(0), frame_feats.std(0) + 1e-6)
        self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(self.std(frame_feats), groups)
        return self

    def proba(self, frame_feats):
        p = self.clf.predict_proba(self.std(frame_feats))
        out = np.zeros((len(frame_feats), len(COND_GROUPS)), np.float32)
        for j, cl in enumerate(self.clf.classes_):
            out[:, COND_GROUPS.index(cl)] = p[:, j]
        return out

    def accuracy(self, frame_feats, groups):
        return float(np.mean(self.clf.predict(self.std(frame_feats)) == np.asarray(groups)))


ARM_BLOCKS = {
    "fixed_clear": [], "fixed": [],
    "confidence": ["confidence"],
    "frame_feats": ["frame_feats"],
    "frame_cls": ["frame_cls"],
    "frame_oracle": ["frame_oracle"],
    "object": ALL_BLOCKS,
}


def arm_inputs(arm_blocks, tbl, weather_clf=None):
    c = tbl["cand"]
    parts = []
    obj_blocks = [b for b in arm_blocks if b in ALL_BLOCKS]
    if obj_blocks:
        parts.append(object_inputs(c, obj_blocks))
    if "frame_cls" in arm_blocks:
        ff = np.stack([f for f in _frame_feat_matrix(tbl)])
        parts.append(weather_clf.proba(ff)[c["frame"]])
    if "frame_oracle" in arm_blocks:
        parts.append(condition_inputs(tbl["frames"], c["frame"]))
    return np.concatenate(parts, 1) if parts else np.zeros((len(c["y"]), 0), np.float32)


def _frame_feat_matrix(tbl):
    c = tbl["cand"]
    out = np.zeros((len(tbl["frames"]), len(FRAME_FEATURES)), np.float32)
    if len(c["frame"]):
        out[c["frame"]] = c["frame_feats"]
    return out


# ------------------------------------------------------------------------------------------------
# Model

class Router(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=2, context=False):
        super().__init__()
        self.in_dim, self.context = in_dim, context and in_dim > 0
        if in_dim == 0:
            self.const = nn.Parameter(torch.zeros(N_PARAMS + 3))
            return
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.enc = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(hidden * (2 if self.context else 1), hidden), nn.GELU(),
                                 nn.Linear(hidden, N_PARAMS + 3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, frame, n_frames):
        if self.in_dim == 0:
            return self.const.expand(len(frame), -1)
        h = self.enc(x)
        if self.context:
            s = torch.zeros(n_frames, h.shape[1], device=h.device).index_add_(0, frame, h)
            cnt = torch.zeros(n_frames, device=h.device).index_add_(0, frame, torch.ones_like(frame, dtype=h.dtype))
            ctx = s / cnt.clamp(min=1).unsqueeze(1)
            h = torch.cat([h, ctx[frame]], 1)
        return self.out(h)


class NaivePredictor:
    """Untrained late-fusion rules, as commonly implemented: max-score, noisy-OR, or sum-of-logits, with
    inverse-variance position weighting from *hand-set* (unfitted) sensor sigmas. Used by Exp 9 to show how
    much of the reported 'adding a modality hurts' effect is an artefact of the baseline, not the weather."""

    HAND_SIGMA_R = {M_CAM: None, M_LID: 0.35, M_RAD: 0.9}   # camera uses its own reported sigma
    HAND_SIGMA_AZ = {M_CAM: 0.006, M_LID: None, M_RAD: 0.012}  # lidar: 0.35/r

    def __init__(self, rule="max"):
        self.rule, self.arm, self.tau = rule, f"naive_{rule}", 0.5
        self.prior = None

    def n_params(self):
        return 0

    def predict(self, tbl, drop=None):
        c = tbl["cand"]
        present = c["present"].copy()
        if drop is not None:
            present &= ~drop[c["frame"]]
        s = np.where(present, c["score"], 0.0)
        if self.rule == "max":
            score = s.max(1)
        elif self.rule == "noisy_or":
            score = 1 - np.prod(1 - s, axis=1)
        elif self.rule == "sum_logits":
            score = 1 / (1 + np.exp(-np.where(present, _logit(c["score"]), 0.0).sum(1)))
        else:
            raise ValueError(self.rule)
        xy = np.nan_to_num(c["xy"])
        rng = np.hypot(xy[..., 0], xy[..., 1])
        az = np.arctan2(xy[..., 1], xy[..., 0])
        sig_r = np.stack([np.nan_to_num(c["sigma_r"][:, M_CAM], nan=1.0),
                          np.full(len(rng), self.HAND_SIGMA_R[M_LID]), np.full(len(rng), self.HAND_SIGMA_R[M_RAD])], 1)
        sig_a = np.stack([np.full(len(rng), self.HAND_SIGMA_AZ[M_CAM]),
                          np.maximum(0.35 / np.maximum(rng[:, M_LID], 1.0), 0.002),
                          np.full(len(rng), self.HAND_SIGMA_AZ[M_RAD])], 1)
        w_r = present / np.maximum(sig_r, 0.05) ** 2
        w_a = present / np.maximum(sig_a, 0.002) ** 2
        r_hat = (w_r * rng).sum(1) / np.maximum(w_r.sum(1), 1e-9)
        az_hat = np.arctan2((w_a * np.sin(az)).sum(1), (w_a * np.cos(az)).sum(1) + 1e-12)
        any_p = present.any(1)
        score = np.where(any_p, score, 0.0)
        return {"score": score, "xy": np.stack([r_hat * np.cos(az_hat), r_hat * np.sin(az_hat)], 1),
                "present_any": any_p, "harm_prob": np.zeros((len(score), 3), np.float32),
                "gain": present.astype(np.float32), "range_share": w_r / np.maximum(w_r.sum(1, keepdims=True), 1e-9),
                "log_precision": np.log(np.maximum(w_r.sum(1), 1e-9))}


def fusion_rule(raw, present, fov, logit_s, xy, lam_r, lam_a, absence_scale=1.0):
    """raw (N, 19) router output -> fused logit (N,), fused xy (N,2), log-precision sums, harm logits (N,3)."""
    P = raw[:, :15].view(-1, 3, 5)
    a = F.softplus(P[..., 0] + 1.0)
    c = P[..., 1]
    u = -F.softplus(P[..., 2] - 1.0)
    d_r, d_a = P[..., 3].clamp(-6, 6), P[..., 4].clamp(-6, 6)
    b = raw[:, 15]
    harm_logits = raw[:, 16:19]
    p = present.float()
    z = b + (p * (a * logit_s + c)).sum(1) + absence_scale * ((1 - p) * fov * u).sum(1)
    rng = torch.linalg.norm(torch.nan_to_num(xy), dim=-1)
    az = torch.atan2(torch.nan_to_num(xy[..., 1]), torch.nan_to_num(xy[..., 0]))
    w_r = p * torch.exp((lam_r + d_r).clamp(-20, 20))
    w_a = p * torch.exp((lam_a + d_a).clamp(-20, 30))
    sw_r = w_r.sum(1).clamp(min=1e-12)
    r_hat = (w_r * rng).sum(1) / sw_r
    s_ = (w_a * torch.sin(az)).sum(1)
    c_ = (w_a * torch.cos(az)).sum(1)
    # candidates whose every member was dropped have s_ = c_ = 0; atan2(0,0) has a NaN gradient that
    # would poison the router through 0 * NaN even though those rows are masked out of the loss
    empty = ~present.any(1)
    c_ = torch.where(empty, torch.ones_like(c_), c_)
    az_hat = torch.atan2(s_, c_)
    fused_xy = torch.stack([r_hat * torch.cos(az_hat), r_hat * torch.sin(az_hat)], 1)
    return z, fused_xy, torch.log(sw_r), harm_logits


class Arbiter:
    """Router + standardiser + frozen fusion rule; handles masking of whole sensors."""

    def __init__(self, arm, in_dim, cfg, weather_clf=None, blocks=None, context=None):
        acfg = cfg["arbitration"]
        self.arm, self.cfg, self.weather_clf = arm, cfg, weather_clf
        self.blocks = list(blocks) if blocks is not None else ARM_BLOCKS[arm]
        if context is None:
            context = acfg["frame_context"] and arm == "object"
        self.absence_scale = 1.0   # set to 0 to ablate the "should have seen it" negative-evidence term
        self.router = Router(in_dim, acfg["hidden"], acfg["depth"], context=context)
        self.std = None

    def n_params(self):
        return sum(p.numel() for p in self.router.parameters())

    def tensors(self, tbl, drop=None):
        """drop: (n_frames, 3) bool — whole sensor removed for that frame."""
        c = tbl["cand"]
        x = arm_inputs(self.blocks, tbl, self.weather_clf)
        present = c["present"].copy()
        fov = c["fov"].copy()
        if drop is not None:
            dm = drop[c["frame"]]
            present &= ~dm
            fov = fov * (~dm)
            x = self._mask_inputs(x, c, dm)
        if self.std is not None and x.shape[1]:
            x = self.std(x)
        xy, lam_r, lam_a = self.prior.apply(c)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
        return {"x": t(np.nan_to_num(x)), "present": t(present, torch.bool), "fov": t(fov),
                "logit": t(np.where(present, _logit(c["score"]), 0.0)), "xy": t(xy),
                "lam_r": t(lam_r), "lam_a": t(lam_a), "frame": t(c["frame"], torch.long),
                "n_frames": len(tbl["frames"])}

    def _mask_inputs(self, x, c, dm):
        """Recompute object inputs with the dropped sensor's evidence removed (only object-level blocks)."""
        obj_blocks = [b for b in self.blocks if b in ALL_BLOCKS]
        if not obj_blocks:
            return x
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
        rest = x[:, object_inputs(c, obj_blocks).shape[1]:]
        return np.concatenate([object_inputs(c2, obj_blocks), rest], 1)

    def forward(self, T, idx=None):
        if idx is None:
            sel = {k: v for k, v in T.items()}
        else:
            sel = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in T.items()}
        raw = self.router(sel["x"], sel["frame"], sel["n_frames"])
        if not getattr(self, "learn_precision", True):  # fix position precisions to the fitted prior
            raw = raw.clone()
            raw[:, [3, 4, 8, 9, 13, 14]] = 0.0
        return fusion_rule(raw, sel["present"], sel["fov"], sel["logit"], sel["xy"], sel["lam_r"], sel["lam_a"],
                           getattr(self, "absence_scale", 1.0))

    @torch.no_grad()
    def predict(self, tbl, drop=None):
        self.router.eval()
        T = self.tensors(tbl, drop)
        z, xy, logprec, harm = self.forward(T)
        present_any = T["present"].any(1).numpy()
        score = torch.sigmoid(z).numpy()
        score[~present_any] = 0.0
        raw = self.router(T["x"], T["frame"], T["n_frames"])
        P = raw[:, :15].view(-1, 3, 5)
        p = T["present"].float()
        gain = (F.softplus(P[..., 0] + 1.0) * p).numpy()
        w_r = (p * torch.exp((T["lam_r"] + P[..., 3].clamp(-6, 6)).clamp(-20, 20))).numpy()
        share = w_r / np.maximum(w_r.sum(1, keepdims=True), 1e-12)
        return {"score": score, "xy": xy.numpy(), "log_precision": logprec.numpy(), "logit": z.numpy(),
                "harm_prob": torch.sigmoid(harm).numpy(), "present_any": present_any,
                "gain": gain, "range_share": share}


# ------------------------------------------------------------------------------------------------
# Training

def targets(tbl, cfg, drop=None):
    """Existence target given the (possibly reduced) set of members: GT-matched AND some remaining
    member within the class match radius."""
    c = tbl["cand"]
    present = c["present"] if drop is None else c["present"] & ~drop[c["frame"]]
    thr = np.where(c["cls"] == 0, cfg["matching"]["center_dist_m"]["Car"], cfg["matching"]["center_dist_m"]["Pedestrian"])
    d = np.linalg.norm(np.nan_to_num(c["xy"], nan=1e6) - np.nan_to_num(c["gt_xy"], nan=-1e6)[:, None], axis=-1)
    d = np.where(present, d, np.inf).min(1)
    return ((c["y"] > 0) & (d <= thr)).astype(np.float32), thr.astype(np.float32)


def fit_arbiter(arm, train_tbl, val_tbl, cfg, harm_labels=None, val_harm=None, weather_clf=None, log=None,
                blocks=None, context=None, seed=0, modality_dropout=None, absence_scale=1.0, learn_precision=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    acfg = cfg["arbitration"]
    blocks = list(blocks) if blocks is not None else ARM_BLOCKS[arm]
    x_train = arm_inputs(blocks, train_tbl, weather_clf)
    arb = Arbiter(arm, x_train.shape[1], cfg, weather_clf, blocks=blocks, context=context)
    arb.prior = PositionPrior().fit(train_tbl)
    arb.absence_scale, arb.learn_precision = absence_scale, learn_precision  # R2-C mechanism ablations
    if x_train.shape[1]:
        arb.std = Standardizer(np.nanmean(x_train, 0), np.nanstd(x_train, 0) + 1e-3)
    p_drop = acfg["dropout_modality_p"] if modality_dropout is None else modality_dropout
    lr = acfg["lr_constant"] if x_train.shape[1] == 0 else acfg["lr"]
    opt = torch.optim.AdamW(arb.router.parameters(), lr=lr, weight_decay=acfg["weight_decay"] if x_train.shape[1] else 0.0)
    n_frames = len(train_tbl["frames"])
    frame_of = train_tbl["cand"]["frame"]
    order_idx = np.argsort(frame_of, kind="stable")
    starts = np.searchsorted(frame_of[order_idx], np.arange(n_frames + 1))
    T_val = arb.tensors(val_tbl)
    y_val, thr_val = targets(val_tbl, cfg)
    best, best_state, patience = math.inf, None, 0
    history = []
    y_clean, thr = targets(train_tbl, cfg)
    # a fixed pool of whole-sensor dropout variants, cycled over epochs (inputs are expensive to rebuild)
    variants = [(arb.tensors(train_tbl), y_clean)]
    if p_drop > 0:
        for _ in range(acfg["dropout_variants"]):
            drop = rng.random((n_frames, 3)) < p_drop
            drop[drop.all(1)] = False
            variants.append((arb.tensors(train_tbl, drop), targets(train_tbl, cfg, drop)[0]))
    batches_per_epoch = max(1, math.ceil(n_frames / acfg["batch_frames"]))
    n_epochs = max(acfg["epochs"], math.ceil(acfg["min_steps"] / batches_per_epoch))
    patience_limit = max(acfg["early_stop_patience"], math.ceil(acfg["min_steps"] / batches_per_epoch / 8))
    for epoch in range(n_epochs):
        arb.router.train()
        T, y = variants[epoch % len(variants)]
        perm = rng.permutation(n_frames)
        losses = []
        for bstart in range(0, n_frames, acfg["batch_frames"]):
            fr = perm[bstart:bstart + acfg["batch_frames"]]
            idx = np.concatenate([order_idx[starts[f]:starts[f + 1]] for f in fr])
            if len(idx) == 0:
                continue
            local = np.full(n_frames, -1, np.int64)
            local[fr] = np.arange(len(fr))
            sel = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in T.items()}
            sel["frame"] = torch.as_tensor(local[frame_of[idx]], dtype=torch.long)
            sel["n_frames"] = len(fr)
            loss = _loss(arb, sel, y[idx], thr[idx], train_tbl["cand"]["gt_xy"][idx],
                         None if harm_labels is None else harm_labels[idx], acfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(arb.router.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        arb.router.eval()
        with torch.no_grad():
            vl = float(_loss(arb, T_val, y_val, thr_val, val_tbl["cand"]["gt_xy"],
                             val_harm, acfg))
        history.append({"epoch": epoch, "train": float(np.mean(losses)) if losses else float("nan"), "val": vl})
        if vl < best - 1e-4:
            best, best_state, patience = vl, copy.deepcopy(arb.router.state_dict()), 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is not None:
        arb.router.load_state_dict(best_state)
    arb.history = history
    if log:
        log.info(f"[{arm}] params={arb.n_params()} epochs={len(history)} best_val={best:.4f}")
    return arb


def _loss(arb, T, y, thr, gt_xy, harm, acfg):
    z, fxy, _, harm_logits = arb.forward(T)
    valid = T["present"].any(1)
    y_t = torch.as_tensor(y)
    bce = F.binary_cross_entropy_with_logits(z[valid], y_t[valid])
    pos = valid & (y_t > 0)
    loss = bce
    if pos.any():
        err = torch.linalg.norm(fxy[pos] - torch.as_tensor(np.nan_to_num(gt_xy))[pos], dim=1) / torch.as_tensor(thr)[pos]
        loss = loss + acfg["loss_loc_weight"] * F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.5)
    if harm is not None:
        h = torch.as_tensor(harm)
        m = T["present"] & (h >= 0)
        if m.any():
            loss = loss + acfg["loss_harm_weight"] * F.binary_cross_entropy_with_logits(harm_logits[m], h[m].float())
    return loss
