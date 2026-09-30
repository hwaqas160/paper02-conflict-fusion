"""Registration-2 baselines + runner on the synthetic cache (numbers are never results)."""
import numpy as np
import pytest

from src.common import load_config


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_r2")
    from src.synthetic import make_synthetic_cache
    from src.common import iter_conditions, condition_tag, resolve
    from src.candidates import build_table, save_table
    from src import run_experiments
    make_synthetic_cache(root / "cache" / "boreas", n_frames=48, seed=0)
    cfg = load_config(overrides={
        "paths.cache_root": str(root / "cache"), "paths.candidates_root": str(root / "candidates"),
        "paths.results_root": str(root / "results"), "arbitration.epochs": 8, "arbitration.early_stop_patience": 3,
        "arbitration.min_steps": 50, "bootstrap.n": 30})
    for c, s in iter_conditions(cfg):
        tag = condition_tag(c, s)
        save_table(build_table(resolve(cfg, "cache_root") / "boreas" / tag, cfg),
                   resolve(cfg, "candidates_root") / "boreas" / f"{tag}.npz")
    return run_experiments.Ctx(cfg)


@pytest.mark.parametrize("name", ["proben", "dempster_shafer", "clocs", "set_transformer", "moe_frame", "oracle"])
def test_baseline_fits_and_predicts(ctx, name, tmp_path):
    from src.r2_experiments import fit_baseline
    m = fit_baseline(ctx, name, tmp_path)
    tt = ctx.test_table(tags=ctx.adverse_tags)
    pr = m.predict(tt)
    n = len(tt["cand"]["y"])
    assert pr["score"].shape == (n,) and pr["xy"].shape == (n, 2)
    assert np.all((pr["score"] >= 0) & (pr["score"] <= 1))
    assert np.isfinite(pr["xy"][pr["present_any"]]).all()
    d = np.zeros((len(tt["frames"]), 3), bool)
    d[:, 2] = True  # radar removed must still work
    pr2 = m.predict(tt, d)
    assert pr2["score"].shape == (n,)
    # cache round-trip honours the fingerprint
    m2 = fit_baseline(ctx, name, tmp_path)
    assert getattr(m2, "_fp", None) == getattr(m, "_fp", None)


def test_oracle_is_upper_bound(ctx, tmp_path):
    from src.evaluate import eval_map
    from src.r2_experiments import fit_baseline
    tt = ctx.test_table(tags=ctx.adverse_tags)
    orc = eval_map(tt, fit_baseline(ctx, "oracle", tmp_path).predict(tt), ctx.cfg)[0]
    ours = eval_map(tt, ctx.arm("object").predict(tt), ctx.cfg)[0]
    assert orc >= ours - 1e-6


def test_runner_sections(ctx):
    from src.r2_experiments import r2b, r2g
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    res, md = r2b(ctx, out)
    assert "headroom_captured" in res and "ProbEn" in md
    res, md = r2g(ctx, out)
    assert "object" in res["latency_ms"]


def test_r2c_ablations(ctx):
    from src.r2_experiments import r2c
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    res, md = r2c(ctx, out)
    assert set(res["vs_full"]) == {"no_absence_term", "no_learned_precision", "no_harm_aux_loss",
                                   "no_localisation_loss", "no_sensor_dropout", "no_frame_context"}
    assert "contributes?" in md


def test_r2f_cv(ctx):
    from src.r2_experiments import r2f, scene_folds
    folds = scene_folds(ctx, 3)
    allsc = sorted(s for f in folds for s in f)
    assert allsc == sorted(set(ctx.scene)) and len(allsc) == len(set(allsc))  # each scene tested exactly once
    import src.r2_experiments as R
    res = R.run_cv(ctx, k=3)
    assert len(res["per_fold"]) == 3 and "object" in res["oof_adverse_map"]


def test_r2a_runner(ctx):
    from src.r2a_truckscenes import report, run
    res = run(ctx, k=3, baselines=True)
    assert {"A1", "A2", "A3", "official", "B_vs_ours", "C_vs_full", "E"} <= set(res)
    assert all(0 <= v["cov_degradation_normalised"] <= 1 for v in res["E"].values())
    assert 0.0 <= res["A2"]["harm_rate"] <= 1.0
    assert res["A1"]["n_scenes"] == len(set(ctx.scene[np.isin(ctx.tag, ctx.adverse_tags)]))  # every scene once
    assert "R2-A1" in report(res)


def test_r2d_transfer(ctx, monkeypatch):
    import copy
    import src.r2d_cadc as D
    fake = copy.deepcopy(ctx.test_table(tag="clear0"))
    for i, f in enumerate(fake["frames"]):
        f.update(frame_id=f"2018_03_06_{i % 4:04d}_{i:010d}", scene=f"cadc_2018_03_06_{i % 4:04d}",
                 condition="snow", severity=2, split="test")
    monkeypatch.setattr(D, "load_cadc_table", lambda cfg, name: copy.deepcopy(fake))
    monkeypatch.setattr(D, "KS", (1, 2))
    res = D.r2d(ctx, draws=1)
    assert res["n_drives"] == 4 and set(res["D2"]) == {1, 2}
    assert all(len(res["D2"][k]["per_draw"][0]["train_drives"]) == k for k in (1, 2))
    assert "D1" in D.report({**res, "D2": {10 if k == 2 else k: v for k, v in res["D2"].items()}})
    assert abs(D.rescale_log_count(np.log1p(3.0), 0.25) - np.log1p(12.0)) < 1e-9


def test_r2e_conformal(ctx):
    from src.r2_experiments import r2e
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    res, md = r2e(ctx, out)
    assert "fog3" in res["loc"] and len(res["lam_bins"]) == 3
    cov = res["loc"]["clear0"]["degradation_normalised"]["coverage"]
    assert 0.0 <= cov <= 1.0


def test_router_v2_attention_is_order_invariant_and_starts_as_rule():
    import torch
    from src.router_v2 import Router2
    torch.manual_seed(0)
    r = Router2(10, 16, 2, "attn", residual=True).eval()
    x = torch.randn(40, 10)
    frame = torch.as_tensor(np.random.default_rng(0).integers(0, 6, 40), dtype=torch.long)
    out = r(x, frame, 6)
    perm = torch.randperm(40)
    out_p = r(x[perm], frame[perm], 6)
    assert torch.allclose(out[perm], out_p, atol=1e-5)  # candidate order inside a frame must not matter
    assert out.shape == (40, 22) and torch.all(out == 0)  # final layer zero-initialised -> rule starts unmodified


@pytest.mark.parametrize("mode", ["V0", "V1", "V2", "V3"])
def test_router_v2_fit_predict_and_chunking(ctx, mode):
    from src.router_v2 import fit_arbiter2
    h, _, _ = ctx.harm()
    fit_m, val_m = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cf, cv = fit_m[ctx.T["cand"]["frame"]], val_m[ctx.T["cand"]["frame"]]
    arb = fit_arbiter2(ctx.sub(fit_m), ctx.sub(val_m), ctx.cfg, mode, harm_labels=h[cf], val_harm=h[cv])
    tt = ctx.test_table(tags=ctx.adverse_tags)
    a, b = arb.predict(tt), arb.predict(tt, chunk_frames=7)
    n = len(tt["cand"]["y"])
    assert a["score"].shape == (n,) and a["xy"].shape == (n, 2) and np.isfinite(a["logit"]).all()
    assert np.allclose(a["logit"], b["logit"], atol=1e-4) and np.allclose(a["xy"], b["xy"], atol=1e-4)
    d = np.zeros((len(tt["frames"]), 3), bool)
    d[:, 2] = True
    assert arb.predict(tt, d)["score"].shape == (n,)  # radar removed still works


def test_r3_pipeline_on_synthetic(ctx):
    from src import r3_experiments as R3
    from src.router_v2 import MODES, fit_arbiter2
    h, _, _ = ctx.harm()
    fm, vm = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cf, cv = fm[ctx.T["cand"]["frame"]], vm[ctx.T["cand"]["frame"]]
    arms = {"original": ctx.arm("object")}
    for m in MODES:
        arms[m] = R3.cached(ctx, f"router_{m}", lambda m=m: fit_arbiter2(ctx.sub(fm), ctx.sub(vm), ctx.cfg, m,
                            harm_labels=h[cf], val_harm=h[cv]), "router_v2.py", "arbitrate.py")
    for name, cls in (("set_transformer", R3.SetFusion), ("set_noattn", R3.SetFusionNoAttn)):
        arms[name] = R3.cached(ctx, name, lambda cls=cls: cls().fit(ctx.sub(fm), ctx.sub(vm), ctx.cfg), "baselines_ext.py",
                               "r3_experiments.py")
    assert arms["set_noattn"].n_params() < arms["set_transformer"].n_params()  # attention removed
    maps, units = R3.evaluate(ctx, arms, {"ALL": ctx.tags, "ADVERSE": ctx.adverse_tags}, ctx.cfg)
    assert set(maps["ALL"]) == set(arms)
    sel, why = R3.select_variant(maps["ALL"], units["ALL"], ctx.cfg)
    assert sel in (None, "V1", "V2", "V3") and why
    again = R3.cached(ctx, "router_V1", lambda: 1 / 0, "router_v2.py", "arbitrate.py")  # must load from cache, not refit
    assert again.n_params() == arms["V1"].n_params()


def test_r3_rule_only_and_correction_size(ctx):
    from src import r3_experiments as R3
    from src.router_v2 import fit_arbiter2
    h, _, _ = ctx.harm()
    fm, vm = ctx.is_fit & ctx.train_cond, ctx.is_val & ctx.train_cond
    cf, cv = fm[ctx.T["cand"]["frame"]], vm[ctx.T["cand"]["frame"]]
    arb = fit_arbiter2(ctx.sub(fm), ctx.sub(vm), ctx.cfg, "V2", harm_labels=h[cf], val_harm=h[cv])
    arb.tau = 0.5
    tt = ctx.test_table(tags=ctx.adverse_tags)
    ro = R3.RuleOnly(arb)
    a, b = arb.predict(tt), ro.predict(tt)
    cs = R3.correction_size(arb, tt)
    assert 0 <= cs["mean_abs_logit_correction"] <= 2.0 and 0 <= cs["mean_position_correction_m"] <= 2 * 2 ** 0.5
    # the switched-off variant differs from the full one exactly by the bounded correction
    assert np.abs(a["logit"] - b["logit"]).max() <= 2.0 + 1e-4
    d = np.linalg.norm(a["xy"] - b["xy"], axis=1)
    assert d.max() <= 2 * 2 ** 0.5 + 1e-4
    lat = R3.latency_ms(arb, tt, n=5)
    assert lat["median"] > 0
