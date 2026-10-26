"""E20 (exploratory, not registered): label-free quantile alignment for sim-to-real transfer.

E13 showed Boreas-trained routers fail on real TruckScenes and that re-centring five radar features made it worse.
Here EVERY router input is mapped feature by feature so that its unlabeled TruckScenes (adverse-frame) distribution equals
the Boreas training distribution (quantile mapping: target empirical CDF -> source quantile function), then the Boreas
standardiser is applied. The per-frame classifier gets the same mapping on its frame features. No TruckScenes label,
detection quality or outcome is used.

    python -m src.e20_quantile [n_boot]
"""
import copy
import json
import sys

import numpy as np

from .e10_minimal import compare


class QuantileStd:
    def __init__(self, src_std, src_q, tgt_q, grid):
        self.src_std, self.src_q, self.tgt_q, self.grid = src_std, src_q, tgt_q, grid

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        out = np.array(x, copy=True)
        for j in range(x.shape[1]):
            ok = np.isfinite(x[:, j])
            if ok.any() and np.ptp(self.tgt_q[:, j]) > 0:
                u = np.interp(x[ok, j], self.tgt_q[:, j], self.grid)       # target CDF
                out[ok, j] = np.interp(u, self.grid, self.src_q[:, j])      # source quantile function
        return self.src_std(out)


def quantiles(X, grid):
    return np.stack([np.nanquantile(X[:, j], grid) if np.isfinite(X[:, j]).any() else np.zeros(len(grid))
                     for j in range(X.shape[1])], 1)


def run(n_boot=1000):
    from .arbitrate import ALL_BLOCKS, _frame_feat_matrix, object_inputs
    from .common import load_config, resolve
    from .evaluate import map_of_units, per_frame_units
    from .r2a_truckscenes import ts_ctx
    from .run_experiments import Ctx
    from .supplementary import scene_units

    bo = Ctx(load_config())
    ts = ts_ctx()
    cfg = ts.cfg
    grid = np.linspace(0.005, 0.995, 199)
    src_tbl = bo.sub(bo.is_fit & bo.train_cond)
    tbl = ts.sub(np.isin(ts.tag, ts.adverse_tags))
    Xs, Xt = object_inputs(src_tbl["cand"], ALL_BLOCKS), object_inputs(tbl["cand"], ALL_BLOCKS)
    obj = copy.deepcopy(bo.arm("object"))
    obj.std = QuantileStd(obj.std, quantiles(Xs, grid), quantiles(Xt, grid), grid)
    fc = copy.deepcopy(bo.arm("frame_cls"))
    wc = copy.deepcopy(fc.weather_clf)
    Fs, Ft = _frame_feat_matrix(src_tbl), _frame_feat_matrix(tbl)
    wc.std = QuantileStd(wc.std, quantiles(Fs, grid), quantiles(Ft, grid), grid)
    fc.weather_clf = wc
    models = {"fixed_clear": bo.status_quo(), "frame_cls": fc, "object": obj}
    units = {n: scene_units(tbl, per_frame_units(tbl, m.predict(tbl), cfg)) for n, m in models.items()}
    res = {"method": "unlabeled quantile alignment of all inputs to Boreas training distribution", "n_boot": n_boot,
           "map": {n: map_of_units(u, cfg) for n, u in units.items()},
           "object_vs_frame_cls": compare(units["object"], units["frame_cls"], cfg, n_boot),
           "object_vs_fixed_clear": compare(units["object"], units["fixed_clear"], cfg, n_boot)}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e20_quantile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    print({k: round(v, 3) for k, v in r["map"].items()})
    for k in ("object_vs_frame_cls", "object_vs_fixed_clear"):
        c = r[k]
        print(k, f"{c['diff']:+.3f}", [round(x, 3) for x in c["ci95"]], f"{c['scenes_ahead']}/{c['n_scenes']}")
