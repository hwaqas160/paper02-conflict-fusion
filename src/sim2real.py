"""Sim-to-real check of the weather simulator on UNLABELLED real Boreas adverse-weather sequences.

Boreas' snow sequences have no 3D labels, but frame-level degradation statistics need none. For a
real snowy sequence this script downloads an evenly spaced subset of LiDAR / camera / radar frames,
computes the same FRAME_FEATURES the cache stores, and compares their distribution with the labelled
clear frames (clear0) and the simulated severities. Reported per feature: median and IQR per group,
plus the simulated severity whose feature vector is nearest (standardised) to the real sequence.

    python -m src.sim2real --sequence boreas-2021-01-26-11-22 --n 60
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .boreas_data import Frame, IMG_H, IMG_W, _nearest, _read_poses, read_camera, read_lidar, read_radar
from .common import PROJECT_ROOT, get_logger, load_config, load_json, resolve, save_json
from .degradation import FRAME_FEATURES, ImageStats, lidar_frame_features
from .detectors.radar_cfar import RadarDetector

S5CMD = PROJECT_ROOT.parent / "shared" / "tools" / "s5cmd.exe"


def fetch(sequence, root, n):
    seq = root / sequence
    meta_cmds = [f'cp "s3://boreas/{sequence}/applanix/*" "{(seq / "applanix").as_posix()}/"',
                 f'cp "s3://boreas/{sequence}/calib/*" "{(seq / "calib").as_posix()}/"']
    _run(meta_cmds, root / f"_sim2real_meta_{sequence}.txt")
    lt, _ = _read_poses(seq / "applanix" / "lidar_poses.csv")
    ct, _ = _read_poses(seq / "applanix" / "camera_poses.csv")
    rt, _ = _read_poses(seq / "applanix" / "radar_poses.csv")
    pick = lt[np.linspace(len(lt) * 0.05, len(lt) * 0.95, n).astype(int)]
    cmds = []
    for t in pick:
        ri = _nearest(rt, t)
        for sensor, ts, ext in (("lidar", t, ".bin"), ("camera", ct[_nearest(ct, t)], ".png"),
                                ("radar", rt[ri], ".png"), ("radar", rt[max(ri - 1, 0)], ".png")):
            dst = seq / sensor / f"{ts}{ext}"
            if not dst.exists():
                cmds.append(f'cp "s3://boreas/{sequence}/{sensor}/{ts}{ext}" "{dst.as_posix()}"')
    _run(cmds, root / f"_sim2real_{sequence}.txt")
    return pick


def _run(cmds, path):
    if not cmds:
        return
    path.write_text("\n".join(cmds) + "\n")
    subprocess.run([str(S5CMD), "--no-sign-request", "--numworkers", "16", "run", str(path)], check=True,
                   stdout=subprocess.DEVNULL)


def real_frame_features(sequence, root, lidar_times, cfg, ground_z):
    seq = root / sequence
    lt, lp = _read_poses(seq / "applanix" / "lidar_poses.csv")
    ct, cp = _read_poses(seq / "applanix" / "camera_poses.csv")
    rt, rp = _read_poses(seq / "applanix" / "radar_poses.csv")
    rdet = RadarDetector(cfg["detectors"]["radar"], cfg["roi"], ground_z)
    feats = []
    for t in lidar_times:
        li, ci, ri = int(np.searchsorted(lt, t)), _nearest(ct, t), _nearest(rt, t)
        fr = Frame("boreas", -1, int(t), int(ct[ci]), int(rt[ri]), int(rt[max(ri - 1, 0)]), lp[li], cp[ci], rp[ri],
                   rp[max(ri - 1, 0)], seq)
        if not fr.has_data():
            continue
        pts = read_lidar(fr)
        img = cv2.resize(read_camera(fr), (IMG_W // 2, IMG_H // 2), interpolation=cv2.INTER_AREA)
        det = rdet(read_radar(fr), fr.T_lidar_radar(), read_radar(fr, prev=True), fr.T_lidar_radar(prev=True))
        feats.append(lidar_frame_features(pts, ground_z) + ImageStats(img).global_feats +
                     [det["frame"]["noise_floor_median"], np.log1p(det["frame"]["n_hits_roi"]), det["frame"]["hit_mean_power"]])
    return np.asarray(feats, np.float32)


def cached_frame_features(cache_dir):
    out = []
    for p in sorted(Path(cache_dir).glob("*.npz")):
        out.append(np.load(p)["frame_feats"])
    return np.asarray(out, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", nargs="+", default=["boreas-2021-01-26-11-22"])
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("sim2real")
    root = resolve(cfg, "boreas_root")
    cache = resolve(cfg, "cache_root") / "boreas"
    ground_z = load_json(cache / "ground_z.json")["ground_z"]
    groups = {}
    for tag in ("clear0", "snow1", "snow2", "snow3", "rain2", "fog2"):
        if (cache / tag).exists():
            f = cached_frame_features(cache / tag)
            if len(f):
                groups[f"sim:{tag}"] = f
    for seq in args.sequence:
        try:
            times = fetch(seq, root, args.n)
        except FileNotFoundError as e:
            # e.g. boreas-2021-01-26-10-59 ships only raw IMU/DMI, no post-processed sensor poses
            log.warning(f"{seq}: skipped, no post-processed poses ({e.filename})")
            continue
        groups[f"real:{seq}"] = real_frame_features(seq, root, times, cfg, ground_z)
        log.info(f"{seq}: {len(groups[f'real:{seq}'])} frames")
    ref = groups["sim:clear0"]
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    table = {g: {n: [float(np.percentile(v[:, i], 25)), float(np.median(v[:, i])), float(np.percentile(v[:, i], 75))]
                 for i, n in enumerate(FRAME_FEATURES)} for g, v in groups.items()}
    nearest = {}
    sims = {g: np.median((v - mu) / sd, 0) for g, v in groups.items() if g.startswith("sim:")}
    for g, v in groups.items():
        if g.startswith("real:"):
            z = np.median((v - mu) / sd, 0)
            d = {s: float(np.linalg.norm(z - zs)) for s, zs in sims.items()}
            per_modality = {}
            for mod, prefix in (("lidar", "f_l_"), ("camera", "f_c_"), ("radar", "f_r_")):
                idx = [i for i, n in enumerate(FRAME_FEATURES) if n.startswith(prefix)]
                per_modality[mod] = min(sims, key=lambda s: np.linalg.norm(z[idx] - sims[s][idx]))
            nearest[g] = {"distances": d, "nearest_overall": min(d, key=d.get), "nearest_per_modality": per_modality}
    out = resolve(cfg, "results_root") / "sim2real"
    save_json({"percentiles_25_50_75": table, "nearest_simulated": nearest}, out / "sim2real.json")
    header = "| feature | " + " | ".join(groups) + " |\n|" + "---|" * (len(groups) + 1) + "\n"
    rows = "".join(f"| {n} | " + " | ".join(f"{table[g][n][1]:.3f} [{table[g][n][0]:.3f}, {table[g][n][2]:.3f}]" for g in groups) + " |\n"
                   for n in FRAME_FEATURES)
    md = "# Sim-to-real check — frame-level degradation features (median [IQR])\n\n" + header + rows
    md += "\n## Nearest simulated condition to each real sequence (standardised by clear0)\n\n"
    for g, v in nearest.items():
        md += f"- **{g}**: overall → {v['nearest_overall']}; per modality → {v['nearest_per_modality']}\n"
    (out / "sim2real.md").write_text(md, encoding="utf-8")
    log.info(f"wrote {out / 'sim2real.md'}")


if __name__ == "__main__":
    main()
