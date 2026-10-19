"""E13 (exploratory, not registered): zero-shot transfer from simulated Boreas weather to real TruckScenes weather.

A reviewer can ask whether the router merely learned to invert the weather simulator. Here every arm is trained only on
Boreas (simulated weather, its radar, its LiDAR, its camera) and applied unchanged, with Boreas' standardiser, position
prior and weather classifier, to every real adverse TruckScenes scene. No TruckScenes frame, label or statistic is used
for fitting or selection. Cached Boreas models are loaded; nothing is retrained.

    python -m src.e13_sim2real [n_boot]
"""
import json
import sys

import numpy as np

from .e10_minimal import ARMS, compare


def run(n_boot=1000):
    from .common import load_config, resolve
    from .evaluate import map_of_units, per_frame_units
    from .r2a_truckscenes import ts_ctx
    from .run_experiments import Ctx
    from .supplementary import scene_units

    bo = Ctx(load_config())
    models = {"fixed_clear": bo.status_quo(), "frame_cls": bo.arm("frame_cls"), "object": bo.arm("object")}
    for name, blocks in ARMS.items():
        models[name] = bo.arm("object", blocks=blocks, context=False, tag=f"e10_{name}")

    ts = ts_ctx()
    cfg = ts.cfg
    tbl = ts.sub(np.isin(ts.tag, ts.adverse_tags))
    ts.log.info(f"[e13] zero-shot on {len(tbl['frames'])} real adverse TruckScenes frames, "
                f"{len(set(f['scene'] for f in tbl['frames']))} scenes")
    units = {n: scene_units(tbl, per_frame_units(tbl, m.predict(tbl), cfg)) for n, m in models.items()}
    ts.log.info("[e13] predictions done")
    res = {"train": "Boreas simulated weather only", "test": "all real adverse TruckScenes scenes", "n_boot": n_boot,
           "map": {n: map_of_units(u, cfg) for n, u in units.items()},
           "object_vs_frame_cls": compare(units["object"], units["frame_cls"], cfg, n_boot),
           "object_vs_fixed_clear": compare(units["object"], units["fixed_clear"], cfg, n_boot),
           "minimal_vs_frame_cls": {n: compare(units[n], units["frame_cls"], cfg, n_boot) for n in ARMS},
           "frame_cls_vs_fixed_clear": compare(units["frame_cls"], units["fixed_clear"], cfg, n_boot)}
    out = resolve(cfg, "results_root") / "r4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e13_sim2real.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


SHIFTED_RADAR = [("radar_probes", "r_bg_pow"), ("radar_probes", "r_mean_pow"), ("frame_feats", "f_r_log_hits"),
                 ("frame_feats", "f_r_hit_pow"), ("frame_feats", "f_r_noise")]


def _object_input_names():
    from .arbitrate import PROBE_GROUPS
    from .candidates import CONFLICT_FEATURES, DET_FEATURES
    from .degradation import FRAME_FEATURES
    names = []
    for g in ("geometry", "lidar_probes", "camera_probes", "radar_probes"):
        names += [(g, n) for n in PROBE_GROUPS[g]]
        if g in ("camera_probes", "radar_probes"):
            names.append((g, "missing_flag"))
    names += [("det_feats", n) for n in DET_FEATURES] + [("conflict", n) for n in CONFLICT_FEATURES]
    names += [("confidence", f"c{i}") for i in range(11)]
    names += [("frame_feats", n) for n in FRAME_FEATURES]
    return names


def run_recentred(n_boot=1000, mode="recentre"):
    """Same zero-shot transfer, but the five most input-shifted radar features (chosen by shift size only, from the
    unlabelled target inputs) are standardised with the unlabelled TruckScenes mean and sd instead of Boreas'. Applied to
    the per-object arm and, for fairness, to the per-frame classifier's weather features. No labels used."""
    import copy

    from .arbitrate import ALL_BLOCKS, Standardizer, _frame_feat_matrix, object_inputs
    from .common import load_config, resolve
    from .degradation import FRAME_FEATURES
    from .evaluate import map_of_units, per_frame_units
    from .r2a_truckscenes import ts_ctx
    from .run_experiments import Ctx
    from .supplementary import scene_units

    bo = Ctx(load_config())
    ts = ts_ctx()
    cfg = ts.cfg
    tbl = ts.sub(np.isin(ts.tag, ts.adverse_tags))
    names = _object_input_names()
    X = object_inputs(tbl["cand"], ALL_BLOCKS)
    assert X.shape[1] == len(names)
    idx = [names.index(k) for k in SHIFTED_RADAR]
    obj = copy.deepcopy(bo.arm("object"))
    mean, std = obj.std.mean.copy(), obj.std.std.copy()
    if mode == "neutral":
        mean[idx], std[idx] = np.nanmean(X[:, idx], 0), 1e9
    else:
        mean[idx], std[idx] = np.nanmean(X[:, idx], 0), np.nanstd(X[:, idx], 0) + 1e-3
    obj.std = Standardizer(mean, std)
    fc = copy.deepcopy(bo.arm("frame_cls"))
    ff = _frame_feat_matrix(tbl)
    fidx = [FRAME_FEATURES.index(n) for g, n in SHIFTED_RADAR if g == "frame_feats"]
    wc = copy.deepcopy(fc.weather_clf)
    m2, s2 = wc.std.mean.copy(), wc.std.std.copy()
    m2[fidx], s2[fidx] = ff[:, fidx].mean(0), (1e9 if mode == "neutral" else ff[:, fidx].std(0) + 1e-6)
    wc.std = Standardizer(m2, s2)
    fc.weather_clf = wc
    models = {"fixed_clear": bo.status_quo(), "frame_cls": fc, "object": obj}
    units = {n: scene_units(tbl, per_frame_units(tbl, m.predict(tbl), cfg)) for n, m in models.items()}
    res = {"mode": mode, "train": "Boreas simulated weather only", "recentred_features": [f"{g}.{n}" for g, n in SHIFTED_RADAR],
           "n_boot": n_boot, "map": {n: map_of_units(u, cfg) for n, u in units.items()},
           "object_vs_frame_cls": compare(units["object"], units["frame_cls"], cfg, n_boot),
           "object_vs_fixed_clear": compare(units["object"], units["fixed_clear"], cfg, n_boot)}
    out = resolve(cfg, "results_root") / "r4"
    fname = "e13_sim2real_recentred.json" if mode == "recentre" else "e13_sim2real_neutral.json"
    (out / fname).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] in ("recentred", "neutral"):
        r = run_recentred(int(sys.argv[1]), "neutral" if sys.argv[2] == "neutral" else "recentre")
        print({k: round(v, 3) for k, v in r["map"].items()})
        for k in ("object_vs_frame_cls", "object_vs_fixed_clear"):
            c = r[k]
            print(k, f"{c['diff']:+.3f}", [round(x, 3) for x in c["ci95"]], f"{c['scenes_ahead']}/{c['n_scenes']}")
        sys.exit(0)
    r = run(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    print({k: round(v, 3) for k, v in r["map"].items()})
    for k in ("object_vs_frame_cls", "object_vs_fixed_clear", "frame_cls_vs_fixed_clear"):
        c = r[k]
        print(k, f"{c['diff']:+.3f}", [round(x, 3) for x in c["ci95"]], f"{c['scenes_ahead']}/{c['n_scenes']}",
              f"sign_p={c['sign_test_p_two_sided']:.2g}")
    for n, c in r["minimal_vs_frame_cls"].items():
        print("minimal", n, f"{c['diff']:+.3f}", f"{c['scenes_ahead']}/{c['n_scenes']}")
