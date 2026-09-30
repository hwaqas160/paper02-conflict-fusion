"""Registration 2, R2-D: real-data transfer to CADC (real snow, VLP-32C LiDAR, no radar).

R2-D1 (resolution confound). Boreas' LiDAR is a 128-beam Alpha Prime, CADC's a 32-beam VLP-32C; both 40 deg vertical
FOV and ~0.2 deg horizontal step at 10 Hz. The LiDAR probes are mapped into Boreas (reference-sensor) units using
ONLY the published sensor specifications -- no CADC labels, no fitting:
  * point counts (l_log_pts, l_density, f_l_log_pts, f_l_log_far_pts): n -> n / rho, rho = 32 / 128 (beam density);
  * l_ring_ratio: recomputed from the TRUE beam index, recovered as the nearest of the VLP-32C's 32 published beam
    elevations (median residual 0.001 deg on real CADC sweeps), instead of the constant 1/32 the original cache
    carried (no ring field in CADC's .bin; notes/s4_fov_bug.md).
  Boreas-trained heads and tau are then applied zero-shot, exactly as in Exp 5.
R2-D2 (few-shot). k in {1, 5, 10} labelled CADC drives are added to the Boreas training scenes; every other CADC
drive is test. Both routers (per-object, per-frame classifier) get identical data. Repeated over R random drive
draws per k; out-of-draw units are merged per test drive and compared by a paired drive-cluster bootstrap.

    python -m src.r2d_cadc --build       # normalised cache -> cache/cadc_norm, then candidates
    python -m src.r2d_cadc [--draws 3]
"""
import argparse
import copy
import multiprocessing as mp
import shutil
import time

import numpy as np

from .arbitrate import merge_tables
from .candidates import load_table
from .common import get_logger, load_config, resolve, save_json, seed_everything
from .degradation import FRAME_FEATURES, PROBE_NAMES, ImageStats, compute_probes
from .evaluate import eval_map, map_of_units, per_frame_units
from .run_experiments import Ctx, NoRadar, _md_table, fmt
from .supplementary import merge_units, scene_paired

VLP32C_ELEV = np.array([-25, -15.639, -11.31, -8.843, -7.254, -6.148, -5.333, -4.667, -4, -3.667, -3.333, -3, -2.667,
                        -2.333, -2, -1.667, -1.333, -1, -0.667, -0.333, 0, 0.333, 0.667, 1, 1.333, 1.667, 2.333,
                        3.333, 4.667, 7, 10.333, 15])  # Velodyne VLP-32C manual, Table 9-1
RHO = 32 / 128
COUNT_PROBES = ["l_log_pts", "l_density"]
COUNT_FRAME = ["f_l_log_pts", "f_l_log_far_pts"]
KS = (1, 5, 10)


def rescale_log_count(v, rho=RHO):
    return np.log1p(np.expm1(v) / rho)


def _lidar_points(frame_id, cfg):
    date, drive, idx = frame_id.rsplit("_", 2)
    f = resolve(cfg, "cadc_root") / date / drive / "labeled" / "lidar_points" / "data" / f"{idx}.bin"
    p = np.fromfile(f, dtype=np.float32).reshape(-1, 4).astype(np.float64)
    if p[:, 3].max() <= 1.5:
        p[:, 3] *= 255.0
    elev = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    ring = np.abs(elev[:, None] - VLP32C_ELEV[None]).argmin(1).astype(np.float64)
    return np.column_stack([p, ring, np.zeros(len(p))])


class _EmptyGrid:
    x_min = y_min = -1e9
    res, nx, ny = 1.0, 0, 0


def _normalise_one(args):
    src, dst, cfg = args
    d = dict(np.load(src, allow_pickle=True))
    meta = eval(str(d["meta"]))
    pts = _lidar_points(meta["frame_id"], cfg)
    dummy = ImageStats(np.zeros((8, 8, 3), np.uint8))
    ci = {n: PROBE_NAMES.index(n) for n in COUNT_PROBES + ["l_ring_ratio"]}
    for key in ("camera", "lidar", "radar", "gt"):
        boxes, pr = d[f"{key}_boxes"], d[f"{key}_probes"].copy()
        if len(boxes):
            fresh = compute_probes(boxes.astype(np.float64), pts, dummy, np.eye(4), np.eye(4), np.zeros((0, 0), np.float32),
                                   _EmptyGrid(), np.zeros((0, 0)), n_rings=len(VLP32C_ELEV))
            pr[:, ci["l_ring_ratio"]] = fresh[:, ci["l_ring_ratio"]]
            for n in COUNT_PROBES:
                pr[:, ci[n]] = rescale_log_count(pr[:, ci[n]])
        d[f"{key}_probes"] = pr
    ff = d["frame_feats"].copy()
    for n in COUNT_FRAME:
        ff[FRAME_FEATURES.index(n)] = rescale_log_count(ff[FRAME_FEATURES.index(n)])
    d["frame_feats"] = ff
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(dst)
    return meta["frame_id"]


def build(cfg, workers=3):
    log = get_logger("r2d", resolve(cfg, "results_root") / "logs" / "r2d.log")
    src_root, dst_root = resolve(cfg, "cache_root") / "cadc", resolve(cfg, "cache_root") / "cadc_norm"
    items = [(p, dst_root / p.parent.name / p.name, cfg) for p in sorted((src_root).glob("*/*.npz"))
             if not (dst_root / p.parent.name / p.name).exists()]
    log.info(f"normalising {len(items)} CADC frames -> {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)
    if (src_root / "manifest.json").exists():
        shutil.copy(src_root / "manifest.json", dst_root / "manifest.json")
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_normalise_one, items, chunksize=8)):
            if (i + 1) % 500 == 0:
                log.info(f"{i + 1}/{len(items)} ({(time.time() - t0) / (i + 1):.2f}s/frame)")
    log.info("done; now: python -m src.candidates --dataset cadc_norm")


def load_cadc_table(cfg, name):
    root = resolve(cfg, "candidates_root") / name
    T = merge_tables([load_table(p) for p in sorted(root.glob("*.npz"))])
    for f in T["frames"]:
        f["scene"] = "cadc_" + str(f["frame_id"]).rsplit("_", 1)[0]  # date_drive: drive numbers repeat across dates
        f["condition"], f["severity"], f["split"] = "snow", 2, "test"  # as r2_experiments.load_cadc
    return T


def drive_units(T, pred, cfg):
    drives = np.array([f["scene"] for f in T["frames"]])
    u = per_frame_units(T, pred, cfg)
    return {d: merge_units([u[i] for i in np.nonzero(drives == d)[0]]) for d in np.unique(drives)}


ARMS = ("fixed_clear", "frame_cls", "object")


def zero_shot(ctx, T, cfg):
    arms = {"fixed_clear": NoRadar(ctx.status_quo()), "frame_cls": NoRadar(ctx.arm("frame_cls")),
            "object": NoRadar(ctx.arm("object"))}
    out, units = {}, {}
    for a, m in arms.items():
        pr = m.predict(T)
        out[a] = eval_map(T, pr, cfg)[0]
        units[a] = drive_units(T, pr, cfg)
    ds = sorted(units["object"])
    cmp = scene_paired([units["object"][d] for d in ds], [units["frame_cls"][d] for d in ds], cfg)
    return {"map": out, "object_vs_frame_cls": cmp}


def few_shot_ctx(boreas, T_cadc, train_drives, tag):
    """Boreas training scenes + k labelled CADC drives; CADC drives are always in the FIT set (never validation)."""
    Tc = copy.deepcopy(T_cadc)
    for f in Tc["frames"]:
        f["split"] = "train" if f["scene"] in train_drives else "test"
    cfg = copy.deepcopy(boreas.cfg)
    cfg["paths"]["results_root"] = str(boreas.out / "r2d" / tag)
    d = Ctx(cfg, table=merge_tables([boreas.T, Tc]))
    is_cadc = np.char.startswith(d.scene.astype(str), "cadc_")
    d.train_cond = d.train_cond | is_cadc  # Boreas snow stays held out; the k labelled CADC drives are training data
    d.is_val = d.is_val & ~is_cadc
    d.is_fit = (d.split == "train") & ~d.is_val
    return d, is_cadc


def r2d(boreas, draws=3):
    cfg = boreas.cfg
    log = boreas.log
    res = {"rho": RHO, "draws": draws}
    T_raw, T_norm = load_cadc_table(cfg, "cadc"), load_cadc_table(cfg, "cadc_norm")
    res["D1_raw"] = zero_shot(boreas, T_raw, cfg)
    res["D1_normalised"] = zero_shot(boreas, T_norm, cfg)
    o = res["D1_normalised"]["object_vs_frame_cls"]
    res["D1_refuted"] = bool(o["diff"] <= 0 or o["p_one_sided"] >= 0.05)
    log.info(f"[r2d] D1 raw {res['D1_raw']['map']} | normalised {res['D1_normalised']['map']}")
    all_drives = sorted({f["scene"] for f in T_norm["frames"]})
    res["n_drives"] = len(all_drives)
    res["D2"] = {}
    for k in KS:
        pooled = {a: {} for a in ARMS}
        per_draw = []
        for r in range(draws):
            rng = np.random.default_rng(1000 + 10 * k + r)
            train = set(rng.choice(all_drives, k, replace=False))
            d, is_cadc = few_shot_ctx(boreas, T_norm, train, f"k{k}_r{r}")
            test = d.sub(is_cadc & (d.split == "test"))
            arms = {"fixed_clear": NoRadar(d.status_quo()), "frame_cls": NoRadar(d.arm("frame_cls")),
                    "object": NoRadar(d.arm("object"))}
            row = {"train_drives": sorted(train)}
            for a, m in arms.items():
                pr = m.predict(test)
                row[a] = eval_map(test, pr, cfg)[0]
                for dv, u in drive_units(test, pr, cfg).items():
                    pooled[a].setdefault(dv, []).append(u)
            per_draw.append(row)
            log.info(f"[r2d] k={k} draw {r}: {row}")
        ds = sorted(pooled["object"])
        U = {a: [merge_units(pooled[a][dv]) for dv in ds] for a in ARMS}
        res["D2"][k] = {"per_draw": per_draw, "pooled_map": {a: map_of_units(U[a], cfg) for a in ARMS},
                        "object_vs_frame_cls": scene_paired(U["object"], U["frame_cls"], cfg),
                        "object_vs_fixed": scene_paired(U["object"], U["fixed_clear"], cfg),
                        "mean_delta": float(np.mean([x["object"] - x["frame_cls"] for x in per_draw]))}
    k10 = res["D2"][max(KS)]["object_vs_frame_cls"]  # registered rule at k = 10
    deltas = [res["D2"][k]["object_vs_frame_cls"]["diff"] for k in KS]
    res["D2_monotone"] = bool(all(b >= a for a, b in zip(deltas, deltas[1:])))
    res["D2_refuted"] = bool(k10["diff"] <= 0 or k10["p_one_sided"] >= 0.05)
    return res


def report(res):
    L = {"fixed_clear": "Fixed (status quo)", "frame_cls": "Per-frame classifier", "object": "Per-object (ours)"}
    md = (f"## R2-D -- Transfer to CADC real snow ({res['n_drives']} drives, radar absent)\n\n"
          "### D1: zero-shot, LiDAR probes in reference-sensor units (spec-only normalisation, no CADC labels)\n\n")
    md += _md_table(["arm", "raw probes (Exp 5)", "resolution-normalised"],
                    [[L[a], fmt(res["D1_raw"]["map"][a]), fmt(res["D1_normalised"]["map"][a])] for a in ARMS])
    for key in ("D1_raw", "D1_normalised"):
        o = res[key]["object_vs_frame_cls"]
        md += (f"\n\n{key}: ours − per-frame Δ {fmt(o['diff'], 4)} [{fmt(o['ci95'][0], 4)}, {fmt(o['ci95'][1], 4)}], "
               f"drive p {fmt(o['p_one_sided'], 3)}")
    md += f"\n\nR2-D1 refuted: {res['D1_refuted']}\n\n### D2: few-shot (k labelled CADC drives, {res['draws']} draws each)\n\n"
    rows = []
    for k, v in res["D2"].items():
        o = v["object_vs_frame_cls"]
        rows.append([str(k)] + [fmt(v["pooled_map"][a]) for a in ARMS] +
                    [f"{fmt(o['diff'], 4)} [{fmt(o['ci95'][0], 4)}, {fmt(o['ci95'][1], 4)}]", fmt(o["p_one_sided"], 3)])
    md += _md_table(["k", *[L[a] for a in ARMS], "Δ ours − per-frame [95 % CI]", "drive p"], rows)
    md += f"\n\nΔ non-decreasing in k: {res['D2_monotone']}. R2-D2 refuted: {res['D2_refuted']}\n"
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    if args.build:
        build(cfg, args.workers)
        return
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    out = ctx.out / "r2"
    out.mkdir(parents=True, exist_ok=True)
    res = r2d(ctx, args.draws)
    save_json(res, out / "r2d_cadc.json")
    (out / "R2D.md").write_text(report(res), encoding="utf-8")


if __name__ == "__main__":
    main()
