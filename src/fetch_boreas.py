"""Selective Boreas download for Paper 02.

Only the labelled lidar frames of ``boreas-objects-v1`` carry 3D ground truth, so there is no
reason to pull all 142 GB of that sequence. For every labelled lidar frame this script selects
    * the lidar scan itself,
    * the camera image closest in time,
    * the radar scan closest in time and the one before it (temporal-persistence feature),
writes an s5cmd command file, and runs it. Downloads are phased: phase A takes every
``--stride``-th labelled frame so the whole pipeline can run end-to-end while phase B continues.

Usage:
    python src/fetch_boreas.py --phase A          # ~1/5 of frames, ~8 GB
    python src/fetch_boreas.py --phase B          # the rest, ~33 GB
    python src/fetch_boreas.py --phase A --dry-run
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S5CMD = ROOT.parent / "shared" / "tools" / "s5cmd.exe"
SEQ = "boreas-objects-v1"


def read_pose_times(csv_path):
    with open(csv_path) as f:
        f.readline()
        return np.array([int(line.split(",", 1)[0]) for line in f if line.strip()], dtype=np.int64)


def nearest(sorted_times, query):
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left, right = sorted_times[idx - 1], sorted_times[idx]
    return np.where(query - left <= right - query, idx - 1, idx)


def build_manifest(data_root, stride, phase):
    seq_root = data_root / SEQ
    label_ts = np.array(sorted(int(p.stem) for p in (seq_root / "labels_detection").glob("*.txt")), dtype=np.int64)
    cam_ts = np.sort(read_pose_times(seq_root / "applanix" / "camera_poses.csv"))
    rad_ts = np.sort(read_pose_times(seq_root / "applanix" / "radar_poses.csv"))

    order = np.arange(len(label_ts))
    chosen = order[order % stride == 0] if phase == "A" else order[order % stride != 0]
    lid = label_ts[chosen]
    cam = cam_ts[nearest(cam_ts, lid)]
    r_idx = nearest(rad_ts, lid)
    rad = np.unique(np.concatenate([rad_ts[r_idx], rad_ts[np.maximum(r_idx - 1, 0)]]))

    items = [("lidar", t, ".bin") for t in lid] + [("camera", t, ".png") for t in np.unique(cam)] + \
            [("radar", t, ".png") for t in rad]
    lines = []
    for sensor, t, ext in items:
        dst = seq_root / sensor / f"{t}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        lines.append(f'cp "s3://boreas/{SEQ}/{sensor}/{t}{ext}" "{dst.as_posix()}"')
    return lines, len(lid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "boreas"))
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    lines, n_frames = build_manifest(data_root, args.stride, args.phase)
    for sensor in ("lidar", "camera", "radar"):
        (data_root / SEQ / sensor).mkdir(parents=True, exist_ok=True)
    manifest = data_root / f"manifest_phase{args.phase}.txt"
    manifest.write_text("\n".join(lines) + "\n")
    print(f"phase {args.phase}: {n_frames} labelled frames, {len(lines)} files still to fetch -> {manifest}")
    if args.dry_run or not lines:
        return
    subprocess.run([str(S5CMD), "--no-sign-request", "--numworkers", str(args.workers), "run", str(manifest)],
                   check=True)


if __name__ == "__main__":
    main()
