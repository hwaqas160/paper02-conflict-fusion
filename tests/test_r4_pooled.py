"""The information-equal per-frame arm: candidates in one frame must receive identical inputs."""
import numpy as np
import pytest

from src.common import load_config


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    root = tmp_path_factory.mktemp("r4")
    from src.synthetic import make_synthetic_cache
    from src.candidates import build_table
    from src.common import iter_conditions, condition_tag, resolve
    make_synthetic_cache(root / "cache" / "boreas", n_frames=24, seed=1)
    cfg = load_config(overrides={"paths.cache_root": str(root / "cache")})
    out = {}
    for c, s in iter_conditions(cfg):
        tag = condition_tag(c, s)
        path = resolve(cfg, "cache_root") / "boreas" / tag
        if path.exists() and any(path.glob("*.npz")):
            out[tag] = build_table(path, cfg)
    return out


def test_pooled_inputs_are_constant_within_frame(tables):
    from src import r4_experiments as R
    tbl = next(t for t in tables.values() if len(t["cand"]["frame"]) > 0)
    X = R.pooled_inputs(tbl)
    frames = tbl["cand"]["frame"]
    assert X.shape[0] == len(frames)
    for f in np.unique(frames):
        rows = X[frames == f]
        assert np.allclose(rows, rows[0]), f"frame {f} candidates received different inputs"


def test_pooled_width_is_fixed_and_finite(tables):
    from src import r4_experiments as R
    from src import arbitrate as A
    tbl = next(t for t in tables.values() if len(t["cand"]["frame"]) > 0)
    X = R.pooled_inputs(tbl)
    d = A.object_inputs(tbl["cand"], A.ALL_BLOCKS).shape[1]
    assert X.shape[1] == len(R.POOL_STATS) * d + A._frame_feat_matrix(tbl).shape[1]
    assert np.isfinite(X).all()


def test_register_routes_only_the_new_arm(tables):
    from src import r4_experiments as R
    from src import arbitrate as A
    R.register()
    tbl = next(t for t in tables.values() if len(t["cand"]["frame"]) > 0)
    x_new = A.arm_inputs(["frame_pooled"], tbl)
    assert x_new.shape == R.pooled_inputs(tbl).shape
    x_old = A.arm_inputs(["frame_feats"], tbl)
    assert x_old.shape[1] == tbl["cand"]["frame_feats"].shape[1]
