"""Registration 3 (notes/registration3.md): can the router match the black-box set-transformer?

    python -m src.r3_experiments dev                 # diagnosis + variants V1-V3 on the DEVELOPMENT split only
    python -m src.r3_experiments final --variant V3  # frozen variant: TruckScenes official val + Boreas test (once)

Development never sees the 75 official TruckScenes val scenes or any Boreas test scene.
"""
import argparse
import copy
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import baselines_ext as BE
from .arbitrate import merge_tables, subset_table
from .baselines_ext import SetFusion
from .common import load_config, save_json, seed_everything
from .evaluate import choose_tau_f1, eval_map, map_of_units, per_frame_units
from .r2_experiments import holm
from .r2a_truckscenes import ts_ctx
from .router_v2 import MODES, fit_arbiter2
from .run_experiments import Ctx, _md_table, fmt
from .supplementary import scene_paired, scene_units

NONINF = 0.01   # R3-1 margin
CLOSE = 0.005   # selection tolerance to the set-transformer


# ---------------------------------------------------------------------------------------------- diagnostics model
class _NoAttnNet(nn.Module):
    """The set-transformer with attention removed: identical width, FFN blocks and free heads, no cross-object mixing."""

    def __init__(self, d, h=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
                                     for _ in range(layers)])
        self.head = nn.Linear(h, 3)

    def forward(self, x, pad):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


class SetFusionNoAttn(SetFusion):
    arm = "set_noattn"

    def fit(self, train, val, cfg, **kw):
        orig = BE._SetNet
        BE._SetNet = _NoAttnNet
        try:
            return super().fit(train, val, cfg, **kw)
        finally:
            BE._SetNet = orig


# ---------------------------------------------------------------------------------------------- development data
def dev_ctx(seed=0, frac=0.2):
    """Official TruckScenes *train* scenes only; a stratified 20 % of them are dev-test."""
    base = ts_ctx()
    T = subset_table(base.T, base.split == "train")
    frames = [dict(f) for f in T["frames"]]
    scenes = np.array([f["scene"] for f in frames])
    cond = {s: next(f["condition"] for f in frames if f["scene"] == s) for s in np.unique(scenes)}
    rng = np.random.default_rng(seed)
    dev_test = set()
    for c in sorted(set(cond.values())):
        members = sorted(s for s, k in cond.items() if k == c)
        k = max(1, int(round(frac * len(members))))
        dev_test |= set(rng.choice(members, k, replace=False))
    for f in frames:
        f["split"] = "test" if f["scene"] in dev_test else "train"
    T["frames"] = frames
    cfg = copy.deepcopy(base.cfg)
    cfg["paths"]["results_root"] = str(Path(cfg["paths"]["results_root"]).parent / "truckscenes_dev")
    ctx = Ctx(cfg, dataset="truckscenes", table=T)
    ctx.dev_test_scenes = sorted(dev_test)
    return ctx


def _fp(ctx, *extra_files):
    c = ctx.T["cand"]
    h = hashlib.sha256(str((len(c["y"]), len(ctx.T["frames"]), float(c["score"].sum()), int(c["y"].sum()),
                            int(ctx.is_fit.sum()), int(ctx.is_val.sum()))).encode())
    for f in extra_files:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def cached(ctx, key, fit_fn, *src_files):
    fp = _fp(ctx, *src_files)
    path = ctx.out / "r3" / "models" / f"{key}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if getattr(obj, "_fp", None) == fp:
            ctx.log.info(f"[r3 {key}] loaded (fingerprint ok)")
            return obj
    t0 = time.time()
    obj = fit_fn()
    obj.fit_seconds = time.time() - t0
    train_clear = ctx.sub((ctx.split == "train") & (ctx.tag == "clear0"))
    obj.tau, obj.f1 = choose_tau_f1(train_clear, obj.predict(train_clear), ctx.cfg)
    obj._fp = fp
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    ctx.log.info(f"[r3 {key}] fitted in {obj.fit_seconds:.0f}s, params={obj.n_params()}")
    return obj


def fit_all(ctx):
    h, _, _ = ctx.harm()
    fm, vm = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cf, cv = fm[ctx.T["cand"]["frame"]], vm[ctx.T["cand"]["frame"]]
    arms = {"original": ctx.arm("object")}
    for m in MODES:
        arms[m] = cached(ctx, f"router_{m}", lambda m=m: fit_arbiter2(
            ctx.sub(fm), ctx.sub(vm), ctx.cfg, m, harm_labels=h[cf], val_harm=h[cv], log=ctx.log, seed=ctx.cfg["seed"]),
            "router_v2.py", "arbitrate.py")
    for name, cls in (("set_transformer", SetFusion), ("set_noattn", SetFusionNoAttn)):
        arms[name] = cached(ctx, name, lambda cls=cls: cls().fit(ctx.sub(fm), ctx.sub(vm), ctx.cfg), "baselines_ext.py",
                            "r3_experiments.py")
    return arms


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate(ctx, arms, pools, cfg):
    maps, units = {}, {}
    for pname, tags in pools.items():
        tt = ctx.test_table(tags=tags)
        if not tt["frames"]:
            continue
        for a, m in arms.items():
            pr = m.predict(tt)
            maps.setdefault(pname, {})[a] = eval_map(tt, pr, cfg)[0]
            units.setdefault(pname, {})[a] = scene_units(tt, per_frame_units(tt, pr, cfg))
    return maps, units


def paired(units, cfg, ref, others):
    out = {}
    for a in others:
        out[a] = scene_paired(units[ref], units[a], cfg)
    return out


def select_variant(maps_all, units_all, cfg):
    """Pre-registered selection rule (registration3.md)."""
    st = maps_all["set_transformer"]
    for v in ("V1", "V2", "V3"):
        if maps_all[v] >= st - CLOSE:
            return v, f"simplest variant within {CLOSE} of the set-transformer ({maps_all[v]:.4f} vs {st:.4f})"
    best = max(("V1", "V2", "V3"), key=lambda v: maps_all[v])
    r = scene_paired(units_all[best], units_all["V0"], cfg)
    if r["diff"] > 0 and r["p_one_sided"] < 0.05:
        return best, f"best dev-test mAP and improves on V0 (diff {r['diff']:.4f}, p {r['p_one_sided']:.3f})"
    return None, "no variant closes the gap or improves on V0 with p < 0.05 -> keep the original router"


def dev(args):
    cfg = None
    ctx = dev_ctx()
    cfg = ctx.cfg
    seed_everything(cfg["seed"])
    ctx.log.info(f"[r3 dev] dev-test scenes: {len(ctx.dev_test_scenes)}")
    arms = fit_all(ctx)
    pools = {"ALL": ctx.tags, "ADVERSE": ctx.adverse_tags, "clear0": ["clear0"]}
    maps, units = evaluate(ctx, arms, pools, cfg)
    order = list(arms)
    res = {"n_dev_test_scenes": len(ctx.dev_test_scenes), "params": {a: int(m.n_params()) for a, m in arms.items()},
           "map": maps, "vs_set_transformer": {}, "vs_V0": {}}
    for p in ("ALL", "ADVERSE"):
        res["vs_set_transformer"][p] = paired(units[p], cfg, "set_transformer", [a for a in order if a != "set_transformer"])
        res["vs_V0"][p] = paired(units[p], cfg, "V0", [a for a in order if a != "V0"])
    sel, why = select_variant(maps["ALL"], units["ALL"], cfg)
    res["selected"], res["selection_reason"] = sel, why
    save_json(res, ctx.out / "r3" / "r3_dev.json")
    md = ("## R3 development (TruckScenes official-train scenes only; dev-test = %d held-out scenes)\n\n" %
          res["n_dev_test_scenes"])
    md += _md_table(["model", "params", "mAP all", "mAP adverse", "mAP clear", "(set-transformer − model), adverse [95 % CI]"],
                    [[a, str(res["params"][a]), fmt(maps["ALL"][a]), fmt(maps["ADVERSE"][a]), fmt(maps["clear0"][a]),
                      ("--" if a == "set_transformer" else
                       f"{fmt(-res['vs_set_transformer']['ADVERSE'][a]['diff'], 4)} "
                       f"[{fmt(-res['vs_set_transformer']['ADVERSE'][a]['ci95'][1], 4)}, "
                       f"{fmt(-res['vs_set_transformer']['ADVERSE'][a]['ci95'][0], 4)}]")] for a in order])
    md += f"\n\nSelected: **{sel}** -- {why}\n"
    (ctx.out / "r3" / "R3dev.md").write_text(md, encoding="utf-8")
    print(md)


class RuleOnly:
    """The frozen variant with its bounded correction switched off (final layer's residual rows zeroed)."""

    def __init__(self, arb):
        self.arb = copy.deepcopy(arb)
        with torch.no_grad():
            self.arb.router.out[-1].weight[-3:] = 0.0
            self.arb.router.out[-1].bias[-3:] = 0.0
        self.tau = arb.tau

    def n_params(self):
        return self.arb.n_params()

    def predict(self, tbl, drop=None):
        return self.arb.predict(tbl, drop)


def correction_size(arb, tbl):
    """Mean |logit correction| and |position correction| (m) the variant applies on a table (interpretability report)."""
    T = arb.tensors(tbl)
    with torch.no_grad():
        raw = arb.router.eval()(T["x"], T["frame"], T["n_frames"]) if T["n_frames"] < 2000 else None
    if raw is None:  # chunk long tables
        outs = []
        for a in range(0, T["n_frames"], 500):
            b = min(a + 500, T["n_frames"])
            sel = ((T["frame"] >= a) & (T["frame"] < b)).nonzero().squeeze(1)
            with torch.no_grad():
                outs.append(arb.router(T["x"][sel], T["frame"][sel] - a, b - a))
        raw = torch.cat(outs)
    n = raw.shape[1] - 3
    logit = 2 * torch.tanh(raw[:, n]).abs().mean().item()
    xy = 2 * torch.tanh(raw[:, n + 1:n + 3]).norm(dim=1).mean().item()
    return {"mean_abs_logit_correction": logit, "mean_position_correction_m": xy}


def latency_ms(arb, tbl, n=60):
    torch.set_num_threads(1)
    ts = []
    for f in range(min(n, len(tbl["frames"]))):
        one = subset_table(tbl, np.arange(len(tbl["frames"])) == f)
        t0 = time.perf_counter()
        arb.predict(one)
        ts.append(1000 * (time.perf_counter() - t0))
    return {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95))}


def _report_block(name, maps, units, cfg, variant, refs):
    rows = []
    for p in maps:
        rows.append([p] + [fmt(maps[p].get(a)) for a in [variant, "variant_rule_only", *refs]])
    return rows


def _frozen_variant(args):
    import json
    var = args.variant or json.loads(Path("results/truckscenes_dev/r3/r3_dev.json").read_text(encoding="utf-8"))["selected"]
    assert var in MODES, "no frozen variant"
    return var


def final_truckscenes(var):
    """(i) TruckScenes official train -> val: 75 scenes that no development step ever touched."""
    from .r2_experiments import fit_baseline
    ctx = ts_ctx()
    cfg = ctx.cfg
    seed_everything(cfg["seed"])
    h, _, _ = ctx.harm()
    fm, vm = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cf, cv = fm[ctx.T["cand"]["frame"]], vm[ctx.T["cand"]["frame"]]
    arms = {"original": ctx.arm("object"),
            "set_transformer": fit_baseline(ctx, "set_transformer", ctx.model_dir / "r2b")}
    arms[var] = cached(ctx, f"final_{var}", lambda: fit_arbiter2(ctx.sub(fm), ctx.sub(vm), cfg, var, harm_labels=h[cf],
                       val_harm=h[cv], log=ctx.log, seed=cfg["seed"]), "router_v2.py", "arbitrate.py")
    arms["variant_rule_only"] = RuleOnly(arms[var])
    maps, units = evaluate(ctx, arms, {"clear0": ["clear0"], "ADVERSE": ctx.adverse_tags, "ALL": ctx.tags}, cfg)
    tt = ctx.test_table(tags=ctx.adverse_tags)
    res = {"variant": var, "noninferiority_margin": NONINF, "n_test_scenes": len(set(ctx.scene[ctx.split == "test"])),
           "n_adverse_scenes": len(set(ctx.scene[(ctx.split == "test") & np.isin(ctx.tag, ctx.adverse_tags)])),
           "params": {a: int(m.n_params()) for a, m in arms.items()}, "map": maps,
           "vs": paired(units["ADVERSE"], cfg, var, ["set_transformer", "original", "variant_rule_only"]),
           "correction": correction_size(arms[var], tt),
           "latency_ms": {"variant": latency_ms(arms[var], tt), "original": latency_ms(arms["original"], tt)}}
    lo = res["vs"]["set_transformer"]["ci95"][0]
    res["R3_1_lower_bound"], res["R3_1_refuted"] = lo, bool(lo < -NONINF)
    save_json(res, ctx.out / "r3" / "r3_final_truckscenes.json")
    nl = "\n"
    order = ["original", "set_transformer", var, "variant_rule_only"]
    md = f"## R3 final -- frozen variant {var}: TruckScenes official train -> val ({res['n_test_scenes']} scenes, " \
         f"{res['n_adverse_scenes']} adverse)" + nl + nl
    md += _md_table(["model", "params", "clear", "adverse", "all"],
                    [[a, str(res["params"][a])] + [fmt(maps[p][a]) for p in ("clear0", "ADVERSE", "ALL")] for a in order])
    md += nl + nl + f"R3-1 lower bound of (variant - set-transformer), adverse: {lo:.4f} (margin -{NONINF}); " \
                    f"refuted: {res['R3_1_refuted']}" + nl
    (ctx.out / "r3" / "R3final_truckscenes.md").write_text(md, encoding="utf-8")
    print(md)


def final_boreas(var):
    """(ii) Boreas test scenes, never used in any TruckScenes development step; no Boreas-side tuning."""
    from .r2_experiments import fit_baseline
    from .supplementary import pools as boreas_pools
    b = Ctx(load_config())
    seed_everything(b.cfg["seed"])
    bh, _, _ = b.harm()
    bfm, bvm = b.is_fit & b.train_cond, b.is_val & b.train_cond
    bcf, bcv = bfm[b.T["cand"]["frame"]], bvm[b.T["cand"]["frame"]]
    barms = {"original": b.arm("object"), "set_transformer": fit_baseline(b, "set_transformer", b.out / "models_r2")}
    barms[var] = cached(b, f"final_{var}", lambda: fit_arbiter2(b.sub(bfm), b.sub(bvm), b.cfg, var, harm_labels=bh[bcf],
                        val_harm=bh[bcv], log=b.log, seed=b.cfg["seed"]), "router_v2.py", "arbitrate.py")
    barms["variant_rule_only"] = RuleOnly(barms[var])
    bmaps, bunits = evaluate(b, barms, {k: v for k, v in boreas_pools(b).items() if v}, b.cfg)
    res = {"variant": var, "params": {a: int(m.n_params()) for a, m in barms.items()}, "map": bmaps,
           "vs": paired(bunits["ADVERSE_POOLED"], b.cfg, var, ["original", "set_transformer", "variant_rule_only"])}
    d = res["vs"]["original"]["diff"]
    res["R3_2_refuted"] = bool(d < -0.005)
    save_json(res, b.out / "r3" / "r3_final_boreas.json")
    nl = "\n"
    order = ["original", "set_transformer", var, "variant_rule_only"]
    md = f"## R3 final -- frozen variant {var}: Boreas test" + nl + nl
    md += _md_table(["model", "params", *bmaps.keys()], [[a, str(res["params"][a])] + [fmt(bmaps[p][a]) for p in bmaps] for a in order])
    md += nl + nl + f"R3-2 (variant - original, adverse pooled) = {d:.4f} (margin -0.005); refuted: {res['R3_2_refuted']}" + nl
    (b.out / "r3" / "R3final_boreas.md").write_text(md, encoding="utf-8")
    print(md)


def final_cv(var, k=5):
    """5-fold scene CV on TruckScenes with the frozen variant. NOT independent of development (folds contain scenes
    that served as dev-fit/dev-test); reported as a robustness view only. Reuses the R2-A fold caches for the
    original router and the set-transformer."""
    from .r2_experiments import fit_baseline, fold_ctx, scene_folds
    ctx = ts_ctx()
    cfg = ctx.cfg
    oof = {"original": [], "set_transformer": [], var: []}
    per_fold = []
    for i, fs in enumerate(scene_folds(ctx, k, seed=0, stratify=True)):
        d = fold_ctx(ctx, fs, f"ts_fold{i}")
        h, _, _ = d.harm()
        fm, vm = d.is_fit & d.train_cond, d.is_val & d.train_cond
        cf, cv = fm[d.T["cand"]["frame"]], vm[d.T["cand"]["frame"]]
        arms = {"original": d.arm("object"), "set_transformer": fit_baseline(d, "set_transformer", d.model_dir / "r2b")}
        arms[var] = cached(d, f"cv{i}_{var}", lambda d=d, fm=fm, vm=vm, cf=cf, cv=cv, h=h: fit_arbiter2(
            d.sub(fm), d.sub(vm), cfg, var, harm_labels=h[cf], val_harm=h[cv], log=d.log, seed=cfg["seed"]),
            "router_v2.py", "arbitrate.py")
        tt = d.test_table(tags=d.adverse_tags)
        row = {}
        for a, m in arms.items():
            pr = m.predict(tt)
            row[a] = eval_map(tt, pr, cfg)[0]
            oof[a].extend(scene_units(tt, per_frame_units(tt, pr, cfg)))
        per_fold.append(row)
        ctx.log.info(f"[r3 cv] fold {i}: {row}")
    res = {"variant": var, "per_fold": per_fold, "oof_map": {a: map_of_units(u, cfg) for a, u in oof.items()},
           "vs": {a: scene_paired(oof[var], oof[a], cfg) for a in ("set_transformer", "original")},
           "folds_variant_ge_set_transformer": int(sum(r[var] >= r["set_transformer"] for r in per_fold)),
           "note": "not independent of development (folds contain development scenes)"}
    save_json(res, ctx.out / "r3" / "r3_final_cv.json")
    print(res["oof_map"], {a: (v["diff"], v["ci95"]) for a, v in res["vs"].items()})


def final(args):
    var = _frozen_variant(args)
    if args.part in ("all", "truckscenes"):
        final_truckscenes(var)
    if args.part in ("all", "boreas"):
        final_boreas(var)
    if args.part == "cv":
        final_cv(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "final"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--part", default="all", choices=["all", "truckscenes", "boreas", "cv"])
    a = ap.parse_args()
    {"dev": dev, "final": final}[a.stage](a)


if __name__ == "__main__":
    # re-import under the real module name so pickled classes (SetFusionNoAttn) are importable later
    import src.r3_experiments as _m
    _m.main()
