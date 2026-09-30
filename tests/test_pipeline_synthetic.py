"""End-to-end: synthetic cache -> candidates -> all experiments. Asserts the pipeline runs and that
the designed-in effects are recoverable. Synthetic numbers are never results."""
import numpy as np
import pytest

from src.common import load_config


@pytest.fixture(scope="module")
def synth_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    from src.synthetic import make_synthetic_cache
    make_synthetic_cache(root / "cache" / "boreas", n_frames=48, seed=0)
    cfg = load_config(overrides={
        "paths.cache_root": str(root / "cache"), "paths.candidates_root": str(root / "candidates"),
        "paths.results_root": str(root / "results"), "arbitration.epochs": 12, "arbitration.early_stop_patience": 4,
        "bootstrap.n": 50, "conformal.alpha": [0.1]})
    return cfg


def test_candidates_and_experiments(synth_cfg):
    from src import candidates, run_experiments
    from src.common import iter_conditions, condition_tag, resolve
    from src.candidates import build_table, save_table
    cfg = synth_cfg
    for c, s in iter_conditions(cfg):
        tag = condition_tag(c, s)
        tbl = build_table(resolve(cfg, "cache_root") / "boreas" / tag, cfg)
        save_table(tbl, resolve(cfg, "candidates_root") / "boreas" / f"{tag}.npz")
    ctx = run_experiments.Ctx(cfg)
    r1 = run_experiments.exp1(ctx)
    assert 0.0 <= r1["H1"]["pooled_adverse_harm_rate"] <= 1.0
    r2 = run_experiments.exp2(ctx)
    assert set(r2["map"]) >= {"fixed_clear", "object", "frame_cls"}
    assert all(0 <= v <= 1 for v in r2["map"]["object"].values() if isinstance(v, float) and not np.isnan(v))
    r4 = run_experiments.exp4(ctx)
    cov = r4["object"]["loc"]["0.1"]["clear0"]["vanilla"]["coverage"]
    assert 0.5 <= cov <= 1.0
    run_experiments.exp6(ctx)
    r7 = run_experiments.exp7(ctx)
    assert "no_lidar" in r7["object"]["clear0"]
    run_experiments.exp8(ctx)
    run_experiments.exp5(ctx)
    run_experiments.write_report(ctx, {})
    assert (resolve(cfg, "results_root") / "REPORT.md").exists()
