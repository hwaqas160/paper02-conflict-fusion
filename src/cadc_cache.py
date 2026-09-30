"""CADC (Canadian Adverse Driving Conditions, Pitropov et al., IJRR 2021) -> detector cache.

External validation in REAL snow (PLAN.md Exp 5). CADC has LiDAR (VLP-32C) and 8 cameras but NO
radar, so the radar stream is cached empty and the heads are evaluated with radar removed.
No weather simulation is applied — the snow is real; every frame is tagged `cadc0`.

Expected layout (cadc_devkit, github.com/mpitropov/cadc_devkit):
    data/cadc/<date>/calib/{00.yaml, extrinsics.yaml}
    data/cadc/<date>/<drive>/3d_ann.json
    data/cadc/<date>/<drive>/labeled/lidar_points/data/<10-digit>.bin     float32 (x, y, z, intensity)
    data/cadc/<date>/<drive>/labeled/image_00/data/<10-digit>.png         front camera

UNVERIFIED until CADC is downloaded (this workspace has no CADC data):
  * cuboid dimension order — devkit convention assumed: dimensions.x = width, .y = length, .z = height;
  * extrinsics key `T_LIDAR_CAM00` assumed to map camera-00 points into the LiDAR frame;
  * image_00 in `labeled/` assumed already undistorted (else apply 00.yaml distortion first).
Check them first on a small slice: `python -m src.cadc_cache --limit 50`, then
`python -m src.candidates --dataset cadc` and `python -m src.run_experiments --exp 5`. The LiDAR-only
status-quo mAP is near zero if the dimension order or the extrinsic direction is wrong.

    python -m src.cadc_cache --workers 2 --threads 6
"""
import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import MODALITIES, CLASS_TO_ID, env_threads, get_logger, load_config, resolve, save_json
from .degradation import FRAME_FEATURES, PROBE_NAMES, ImageStats, compute_probes, lidar_frame_features
from .geometry import in_roi, inv_tf

CAM_SCALE = 0.5
CADC_LABEL_MAP = {"Car": "Car", "Pickup_Truck": "Car", "Pedestrian": "Pedestrian", "Pedestrian_With_Object": "Pedestrian",
                  "Bicycle": "Cyclist"}


@dataclass
class CadcFrame:
    root: Path
    date: str
    drive: str
    index: int
    cuboids: list

    @property
    def frame_id(self):
        return f"{self.date}_{self.drive}_{self.index:010d}"

    def lidar_path(self):
        return self.root / self.date / self.drive / "labeled" / "lidar_points" / "data" / f"{self.index:010d}.bin"

    def cam_path(self):
        return self.root / self.date / self.drive / "labeled" / "image_00" / "data" / f"{self.index:010d}.png"


def build_cadc_index(root):
    root = Path(root)
    frames = []
    for ann in sorted(root.glob("*/*/3d_ann.json")):
        date, drive = ann.parts[-3], ann.parts[-2]
        with open(ann) as f:
            data = json.load(f)
        for i, fr in enumerate(data):
            frames.append(CadcFrame(root, date, drive, i, fr.get("cuboids", [])))
    return frames


def cadc_calib(root, date):
    calib = Path(root) / date / "calib"
    with open(calib / "extrinsics.yaml") as f:
        ext = yaml.safe_load(f)
    with open(calib / "00.yaml") as f:
        intr = yaml.safe_load(f)
    K = np.array(intr["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    P = np.eye(4)
    P[:3, :3] = K
    # Key verified on the real calib (2026-09-21): T_LIDAR_00CAMERA maps camera-00 points into the LiDAR frame
    # (its 3rd rotation column, the optical axis, is LiDAR +x).
    T_lidar_cam = np.array(ext["T_LIDAR_00CAMERA"], dtype=np.float64).reshape(4, 4)
    return {"P": P, "T_cam_lidar": inv_tf(T_lidar_cam), "width": int(intr["image_width"]), "height": int(intr["image_height"])}


def cadc_ground_truth(frame):
    boxes, cls, npts = [], [], []
    dontcare = []
    for c in frame.cuboids:
        p, d = c["position"], c["dimensions"]
        box = [p["x"], p["y"], p["z"], d["y"], d["x"], d["z"], c["yaw"]]
        label = CADC_LABEL_MAP.get(c["label"])
        n = int(c.get("points_count", 0))
        if label is None or n < 25:
            dontcare.append(box)
            continue
        boxes.append(box)
        cls.append(CLASS_TO_ID[label])
        npts.append(n)
    return {"boxes": np.asarray(boxes, float).reshape(-1, 7), "cls": np.asarray(cls, int), "npts": np.asarray(npts, int),
            "dontcare": np.asarray(dontcare, float).reshape(-1, 7)}


_W = {}


def _init(cfg, ground_z, threads):
    env_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from .common import pick_device
    from .detectors.camera_frcnn import CameraDetector
    from .detectors.pointpillars import PointPillarsDetector
    dcfg = cfg["detectors"]
    device = pick_device(dcfg["device"])
    _W["lidar"] = PointPillarsDetector(resolve(cfg, "checkpoints") / dcfg["lidar"]["checkpoint"], device=device,
                                       z_shift=-dcfg["lidar"]["sensor_height_kitti"] - ground_z, intensity_scale=1 / 255.0,
                                       score_thr=dcfg["lidar"]["score_thr"], legacy=True)
    _W["camera"] = CameraDetector(device=device, score_thr=dcfg["camera"]["score_thr"], ground_z=ground_z,
                                  bottom_sigma_px=dcfg["camera"]["lift_ground_sigma_px"] * CAM_SCALE)
    _W["cfg"], _W["ground_z"] = cfg, ground_z


def process(item):
    frame, out_path = item
    cfg, ground_z = _W["cfg"], _W["ground_z"]
    cal = cadc_calib(frame.root, frame.date)
    P = cal["P"].copy()
    P[:2, :3] *= CAM_SCALE
    w, h = int(cal["width"] * CAM_SCALE), int(cal["height"] * CAM_SCALE)
    pts4 = np.fromfile(frame.lidar_path(), dtype=np.float32).reshape(-1, 4).astype(np.float64)
    # CADC intensity is in [0, 1]; Boreas (and the detector + probes built on it) use 0-255.
    if pts4[:, 3].max() <= 1.5:
        pts4[:, 3] *= 255.0
    pts = np.concatenate([pts4, np.zeros((len(pts4), 2))], 1)  # ring/time unavailable -> 0
    img = cv2.resize(cv2.cvtColor(cv2.imread(str(frame.cam_path())), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
    det_l = _W["lidar"](pts)
    det_l["labels"] = np.array([2, 1, 0])[det_l["labels"]] if len(det_l["labels"]) else det_l["labels"]
    det_c = _W["camera"](img, cal["T_cam_lidar"], P)
    gt = cadc_ground_truth(frame)
    stats = ImageStats(img)

    class _EmptyGrid:  # radar absent: probes stay NaN
        x_min = y_min = -1e9
        res, nx, ny = 1.0, 0, 0

    probe = lambda boxes: compute_probes(boxes, pts, stats, cal["T_cam_lidar"], P, np.zeros((0, 0), np.float32),
                                         _EmptyGrid(), np.zeros((0, 0)), n_rings=32)
    ff = np.array(lidar_frame_features(pts, ground_z) + stats.global_feats + [np.nan, np.nan, np.nan], np.float32)
    arrays = {"frame_feats": np.nan_to_num(ff), "gt_boxes": gt["boxes"], "gt_cls": gt["cls"], "gt_npts": gt["npts"],
              "gt_dontcare": gt["dontcare"], "gt_probes": probe(gt["boxes"]), "gt_in_roi": in_roi(gt["boxes"], cfg["roi"])}
    for name, d in (("camera", det_c), ("lidar", det_l)):
        arrays[f"{name}_boxes"] = d["boxes"].astype(np.float32)
        arrays[f"{name}_scores"] = d["scores"].astype(np.float32)
        arrays[f"{name}_labels"] = d["labels"].astype(np.int8)
        arrays[f"{name}_probes"] = probe(d["boxes"])
    arrays.update({"radar_boxes": np.zeros((0, 7), np.float32), "radar_scores": np.zeros(0, np.float32),
                   "radar_labels": np.zeros(0, np.int8), "radar_probes": np.zeros((0, len(PROBE_NAMES)), np.float32),
                   "radar_feats": np.zeros((0, 7), np.float32)})
    for k in ("range_ground", "range_size", "range_sigma", "truncated", "boxes2d"):
        arrays[f"camera_{k}"] = np.asarray(det_c[k], np.float32)
    # CADC is external validation only: every frame is 'test'
    meta = {"frame_id": frame.frame_id, "scene": int(frame.drive), "split": "test", "condition": "cadc", "severity": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=np.array(repr(meta)), **arrays)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="override detectors.device (cpu | cuda | auto)")
    args = ap.parse_args()
    cfg = load_config(None, {"detectors.device": args.device} if args.device else None)
    log = get_logger("cadc", resolve(cfg, "results_root") / "logs" / "cadc_cache.log")
    frames = build_cadc_index(resolve(cfg, "cadc_root"))
    if not frames:
        raise SystemExit(f"no CADC 3d_ann.json under {resolve(cfg, 'cadc_root')} — see data/DOWNLOAD.md §2")
    out_root = resolve(cfg, "cache_root") / "cadc" / "cadc0"
    manifest = out_root.parent / "manifest.json"
    if manifest.exists():  # frozen on first run so every cached frame shares one ground height
        from .common import load_json
        ground_z = load_json(manifest)["ground_z"]
    else:
        bottoms = [cadc_ground_truth(f)["boxes"] for f in frames[:: max(1, len(frames) // 300)]]
        ground_z = float(np.median(np.concatenate([b[:, 2] - b[:, 5] / 2 for b in bottoms if len(b)])))
    items = [(f, out_root / f"{f.frame_id}.npz") for f in frames if not (out_root / f"{f.frame_id}.npz").exists()]
    items = items[: args.limit] if args.limit else items
    save_json({"ground_z": ground_z, "n_frames": len(frames)}, out_root.parent / "manifest.json")
    log.info(f"CADC: {len(frames)} frames, {len(items)} to run, ground_z={ground_z:.3f}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg, ground_z, args.threads)) as pool:
        for k, meta in enumerate(pool.imap_unordered(process, items), 1):
            if k % 20 == 0:
                log.info(f"{k}/{len(items)} {(time.time() - t0) / k:.1f}s/frame")


if __name__ == "__main__":
    main()
