"""Registration 3 router variants (notes/registration3.md). Lives beside arbitrate.py instead of editing it, so the
frozen original router -- and every cache fingerprint that hashes arbitrate.py -- stays byte-identical.

Modes (the complete list registered in advance):
  V0  context_mode="mean", residual=False   the original router (reproduced here only for the sanity test)
  V1  context_mode="attn", residual=False   frame context by 2-layer 4-head attention over the frame's candidates
  V2  context_mode="mean", residual=True    + bounded correction: logit += 2 tanh(r), xy += 2 tanh(o) metres
  V3  context_mode="attn", residual=True
The fusion rule (per-sensor gains, absence terms, precision-weighted positions) is untouched in every mode; the
residual, when on, is added to its output and starts at exactly zero.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .arbitrate import (ALL_BLOCKS, N_PARAMS, Arbiter, PositionPrior, Standardizer, _loss, arm_inputs, fusion_rule,
                        targets)

MODES = {"V0": dict(context_mode="mean", residual=False), "V1": dict(context_mode="attn", residual=False),
         "V2": dict(context_mode="mean", residual=True), "V3": dict(context_mode="attn", residual=True)}
CHUNK = 64  # frames per attention block (memory only; does not change results)


class Router2(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=2, context_mode="mean", residual=False):
        super().__init__()
        self.context_mode, self.residual, self.hidden = context_mode, residual, hidden
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.enc = nn.Sequential(*layers)
        if context_mode == "attn":
            layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.0, batch_first=True, norm_first=True)
            self.attn = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_PARAMS + 3 + (3 if residual else 0)))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _mean_context(self, h, frame, n_frames):
        s = torch.zeros(n_frames, h.shape[1], device=h.device).index_add_(0, frame, h)
        cnt = torch.zeros(n_frames, device=h.device).index_add_(0, frame, torch.ones_like(frame, dtype=h.dtype))
        return (s / cnt.clamp(min=1).unsqueeze(1))[frame]

    def _attend(self, h, frame, n_frames):
        """Self-attention among the candidates of each frame (padded blocks of CHUNK frames)."""
        order = torch.sort(frame, stable=True).indices
        counts = torch.bincount(frame, minlength=n_frames)
        starts = torch.cumsum(counts, 0) - counts
        rows, vals = [], []
        nonempty = torch.nonzero(counts).squeeze(1)
        for chunk in torch.split(nonempty, CHUNK):
            c = counts[chunk]
            L = int(c.max())
            ar = torch.arange(L, device=h.device)
            mask = ar[None] < c[:, None]
            pos = torch.where(mask, starts[chunk][:, None] + ar[None], torch.zeros_like(mask, dtype=torch.long))
            src = order[pos]
            hb = h[src] * mask.unsqueeze(-1)
            ob = self.attn(hb, src_key_padding_mask=~mask)
            rows.append(src[mask])
            vals.append(ob[mask])
        return torch.zeros_like(h).index_copy(0, torch.cat(rows), torch.cat(vals))

    def forward(self, x, frame, n_frames):
        h = self.enc(x)
        ctx = self._attend(h, frame, n_frames) if self.context_mode == "attn" else self._mean_context(h, frame, n_frames)
        return self.out(torch.cat([h, ctx], 1))


def fusion_rule2(raw, present, fov, logit_s, xy, lam_r, lam_a, absence_scale=1.0):
    z, fused, logprec, harm = fusion_rule(raw[:, :N_PARAMS + 3], present, fov, logit_s, xy, lam_r, lam_a, absence_scale)
    if raw.shape[1] > N_PARAMS + 3:
        z = z + 2.0 * torch.tanh(raw[:, N_PARAMS + 3])
        fused = fused + 2.0 * torch.tanh(raw[:, N_PARAMS + 4:N_PARAMS + 6])
    return z, fused, logprec, harm


class Arbiter2(Arbiter):
    def __init__(self, arm, in_dim, cfg, weather_clf=None, blocks=None, context=None, mode="V1"):
        super().__init__(arm, in_dim, cfg, weather_clf, blocks=blocks, context=True)
        m = MODES[mode] if isinstance(mode, str) else mode
        a = cfg["arbitration"]
        self.mode = m
        self.router = Router2(in_dim, a["hidden"], a["depth"], m["context_mode"], m["residual"])

    def forward(self, T, idx=None):
        sel = dict(T) if idx is None else {k: (v[idx] if torch.is_tensor(v) else v) for k, v in T.items()}
        raw = self.router(sel["x"], sel["frame"], sel["n_frames"])
        return fusion_rule2(raw, sel["present"], sel["fov"], sel["logit"], sel["xy"], sel["lam_r"], sel["lam_a"],
                            getattr(self, "absence_scale", 1.0))

    @torch.no_grad()
    def predict(self, tbl, drop=None, chunk_frames=512):
        """Chunked over frames so attention memory stays bounded on a whole test table."""
        self.router.eval()
        T = self.tensors(tbl, drop)
        n = T["n_frames"]
        frame = T["frame"]
        out = {}
        pieces = []
        for a in range(0, n, chunk_frames):
            b = min(a + chunk_frames, n)
            sel = ((frame >= a) & (frame < b)).nonzero().squeeze(1)
            if len(sel) == 0:
                continue
            sub = {k: (v[sel] if torch.is_tensor(v) else v) for k, v in T.items()}
            sub["frame"] = frame[sel] - a
            sub["n_frames"] = b - a
            z, xy, logprec, harm = self.forward(sub)
            raw = self.router(sub["x"], sub["frame"], sub["n_frames"])
            P = raw[:, :15].view(-1, 3, 5)
            p = sub["present"].float()
            w_r = p * torch.exp((sub["lam_r"] + P[..., 3].clamp(-6, 6)).clamp(-20, 20))
            pieces.append((sel, z, xy, logprec, harm, torch.nn.functional.softplus(P[..., 0] + 1.0) * p,
                           w_r / w_r.sum(1, keepdim=True).clamp(min=1e-12)))
        N = len(frame)
        cat = lambda i, shape: np.concatenate([q[i].numpy() for q in pieces]) if pieces else np.zeros(shape)
        order = np.argsort(np.concatenate([q[0].numpy() for q in pieces]), kind="stable")
        g = lambda i: cat(i, (0,))[order]
        present_any = T["present"].any(1).numpy()
        score = torch.sigmoid(torch.as_tensor(g(1))).numpy()
        score[~present_any] = 0.0
        return {"score": score, "xy": g(2), "log_precision": g(3), "logit": g(1),
                "harm_prob": torch.sigmoid(torch.as_tensor(g(4))).numpy(), "present_any": present_any,
                "gain": g(5), "range_share": g(6)}


def fit_arbiter2(train_tbl, val_tbl, cfg, mode, harm_labels=None, val_harm=None, log=None, seed=0):
    """fit_arbiter (arbitrate.py) for the 'object' arm with a Router2; identical data handling, loss, dropout
    variants, early stopping and hyperparameters -- only the router differs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    acfg = cfg["arbitration"]
    blocks = list(ALL_BLOCKS)
    x_train = arm_inputs(blocks, train_tbl, None)
    arb = Arbiter2("object", x_train.shape[1], cfg, None, blocks=blocks, mode=mode)
    arb.prior = PositionPrior().fit(train_tbl)
    arb.std = Standardizer(np.nanmean(x_train, 0), np.nanstd(x_train, 0) + 1e-3)
    p_drop = acfg["dropout_modality_p"]
    opt = torch.optim.AdamW(arb.router.parameters(), lr=acfg["lr"], weight_decay=acfg["weight_decay"])
    n_frames = len(train_tbl["frames"])
    frame_of = train_tbl["cand"]["frame"]
    order_idx = np.argsort(frame_of, kind="stable")
    starts = np.searchsorted(frame_of[order_idx], np.arange(n_frames + 1))
    T_val = arb.tensors(val_tbl)
    y_val, thr_val = targets(val_tbl, cfg)
    best, best_state, patience = math.inf, None, 0
    history = []
    y_clean, thr = targets(train_tbl, cfg)
    variants = [(arb.tensors(train_tbl), y_clean)]
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
            torch.nn.utils.clip_grad_norm_(arb.router.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        arb.router.eval()
        with torch.no_grad():
            vl = _val_loss(arb, T_val, y_val, thr_val, val_tbl["cand"]["gt_xy"], val_harm, acfg)
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
        log.info(f"[object/{mode}] params={arb.n_params()} epochs={len(history)} best_val={best:.4f}")
    return arb


def _val_loss(arb, T, y, thr, gt_xy, harm, acfg, chunk_frames=512):
    """Validation loss over the whole validation table, in frame chunks (attention memory)."""
    n = T["n_frames"]
    frame = T["frame"]
    tot, cnt = 0.0, 0
    for a in range(0, n, chunk_frames):
        b = min(a + chunk_frames, n)
        sel = ((frame >= a) & (frame < b)).nonzero().squeeze(1)
        if len(sel) == 0:
            continue
        sub = {k: (v[sel] if torch.is_tensor(v) else v) for k, v in T.items()}
        sub["frame"] = frame[sel] - a
        sub["n_frames"] = b - a
        si = sel.numpy()
        l = float(_loss(arb, sub, y[si], thr[si], gt_xy[si], None if harm is None else harm[si], acfg))
        tot += l * len(si)
        cnt += len(si)
    return tot / max(cnt, 1)
