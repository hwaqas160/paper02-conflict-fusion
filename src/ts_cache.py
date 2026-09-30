"""MAN TruckScenes -> frozen-detector cache (Registration 2, R2-A). Same .npz schema as cache_detections.py,
so candidates.py / run_experiments.py / baselines_ext.py run unchanged with --dataset truckscenes.

Real weather, no simulation: each keyframe is filed under its scene's tag (clear0, rain1, snow1, fog1,
night1; see truckscenes_data.condition_of). Detectors are the same frozen checkpoints as for Boreas; radar is
the training-free 4D point detector (detectors/radar_points4d.py).

    python -m src.ts_cache --workers 2 --threads 6 [--device auto] [--limit 50]
"""
import argparse
import multiprocessing as mp
import time
import traceback

import cv2
import numpy as np

from .common import MODALITIES, env_threads, get_logger, load_config, load_json, resolve, save_json
from .degradation import FRAME_FEATURES, ImageStats, compute_probes, lidar_frame_features
from .geometry import in_roi

CAM_SCALE = 0.5
_W = {}


def _init(cfg, ground_z, threads):
    env_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from .common import pick_device
    from .detectors.camera_frcnn import CameraDetector
    from .detectors.pointpillars import PointPillarsDetector
    from .detectors.radar_points4d import RadarPoints4D
    from .truckscenes_data import annotations_by_sample
    d = cfg["detectors"]
    device = pick_device(d["device"])
    _W["lidar"] = PointPillarsDetector(resolve(cfg, "checkpoints") / d["lidar"]["checkpoint"], device=device,
                                       z_shift=-d["lidar"]["sensor_height_kitti"] - ground_z, intensity_scale=1 / 255.0,
                                       score_thr=d["lidar"]["score_thr"], nms_iou=d["lidar"]["nms_iou"],
                                       max_det=d["lidar"]["max_det"], legacy=True)
    _W["camera"] = CameraDetector(device=device, score_thr=d["camera"]["score_thr"], max_det=d["camera"]["max_det"],
                                  ground_z=ground_z, bottom_sigma_px=d["camera"]["lift_ground_sigma_px"] * CAM_SCALE)
    _W["radar"] = RadarPoints4D(d["radar"], cfg["roi"], ground_z)
    _W["ann"] = annotations_by_sample()
    _W["cfg"], _W["ground_z"], _W["device"] = cfg, ground_z, str(device)


def process(item):
    from .truckscenes_data import (camera_geometry, n_pseudo_rings, read_camera, read_ground_truth, read_lidar,
                                   read_radar)
    fr, out_path, split = item
    cfg, ground_z = _W["cfg"], _W["ground_z"]
    t = {}
    t0 = time.time()
    pts = read_lidar(fr)
    img_full = read_camera(fr)
    h, w = int(img_full.shape[0] * CAM_SCALE), int(img_full.shape[1] * CAM_SCALE)
    img = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
    T_cam, P = camera_geometry(fr)
    P[:2, :3] *= CAM_SCALE
    rad, rad_prev = read_radar(fr), read_radar(fr, prev=True)
    t["load"] = time.time() - t0
    t0 = time.time()
    det_l = _W["lidar"](pts)
    det_l["labels"] = np.array([2, 1, 0])[det_l["labels"]] if len(det_l["labels"]) else det_l["labels"]
    t["lidar"] = time.time() - t0
    t0 = time.time()
    det_c = _W["camera"](img, T_cam, P)
    t["camera"] = time.time() - t0
    det_r = _W["radar"](rad, rad_prev)
    t0 = time.time()
    stats = ImageStats(img)
    rings = n_pseudo_rings(pts)
    probe = lambda b: compute_probes(b, pts, stats, T_cam, P, det_r["power_map"], _W["radar"].grid,
                                     det_r["occupancy"], n_rings=rings)
    gt = read_ground_truth(fr, _W["ann"])
    ff = np.array(lidar_frame_features(pts, ground_z) + stats.global_feats +
                  [det_r["frame"]["noise_floor_median"], np.log1p(det_r["frame"]["n_hits_roi"]),
                   det_r["frame"]["hit_mean_power"]], np.float32)
    assert len(ff) == len(FRAME_FEATURES)
    arrays = {"frame_feats": ff, "gt_boxes": gt["boxes"], "gt_cls": gt["cls"], "gt_npts": gt["npts"],
              "gt_dontcare": gt["dontcare"], "gt_probes": probe(gt["boxes"]), "gt_in_roi": in_roi(gt["boxes"], cfg["roi"])}
    for name, d in zip(MODALITIES, (det_c, det_l, det_r)):
        arrays[f"{name}_boxes"] = d["boxes"].astype(np.float32)
        arrays[f"{name}_scores"] = d["scores"].astype(np.float32)
        arrays[f"{name}_labels"] = d["labels"].astype(np.int8)
        arrays[f"{name}_probes"] = probe(d["boxes"])
    for k in ("range_ground", "range_size", "range_sigma", "truncated", "boxes2d"):
        arrays[f"camera_{k}"] = np.asarray(det_c[k], np.float32)
    arrays["radar_feats"] = det_r["feats"].astype(np.float32)
    t["probes"] = time.time() - t0
    meta = {"frame_id": fr.token, "scene": fr.scene, "split": split, "condition": fr.condition,
            "severity": fr.severity, "tags": fr.tags, "timings": t, "n_points": int(len(pts)), "device": _W["device"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(repr(meta)), **arrays)
    tmp.replace(out_path)
    return meta


def _safe(item):
    try:
        return process(item), None
    except Exception:
        return None, f"{item[0].token}: {traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="every k-th frame (quick representative checks)")
    args = ap.parse_args()
    cfg = load_config(None, {"detectors.device": args.device} if args.device else None)
    log = get_logger("ts_cache", resolve(cfg, "results_root") / "logs" / "ts_cache.log")
    from . import ts_splits
    from .truckscenes_data import annotations_by_sample, build_ts_index, read_ground_truth
    frames = [f for f in build_ts_index() if f.has_data()]
    split_of = {s: "train" for s in ts_splits.train_detect}
    split_of.update({s: "test" for s in ts_splits.val})
    out_root = resolve(cfg, "cache_root") / "truckscenes"
    man = out_root / "manifest.json"
    if man.exists():  # frozen on first run: one ground height for every frame
        ground_z = load_json(man)["ground_z"]
    else:
        ann = annotations_by_sample()
        bottoms = [read_ground_truth(f, ann)["boxes"] for f in frames[::20] if split_of.get(f.scene) == "train"]
        bottoms = np.concatenate([b[:, 2] - b[:, 5] / 2 for b in bottoms if len(b)])
        ground_z = float(np.median(bottoms))
        save_json({"ground_z": ground_z, "estimated_from": "train-split GT box bottoms (every 20th frame)"}, man)
    items = []
    for f in frames[::args.stride]:
        p = out_root / f"{f.condition}{f.severity}" / f"{f.token}.npz"
        if not p.exists():
            items.append((f, p, split_of.get(f.scene, "train")))
    items = items[: args.limit] if args.limit else items
    log.info(f"TruckScenes: {len(frames)} frames with data, {len(items)} to run, ground_z={ground_z:.3f}")
    if not items:
        return
    t0, done, failed = time.time(), 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg, ground_z, args.threads)) as pool:
        for meta, err in pool.imap_unordered(_safe, items, chunksize=1):
            if err:
                failed += 1
                log.error(err)
                continue
            done += 1
            if done % 20 == 0 or done == len(items):
                r = (time.time() - t0) / done
                log.info(f"{done}/{len(items)} ({failed} failed) {r:.1f}s/frame ETA {(len(items) - done) * r / 3600:.1f}h "
                         f"| {meta['condition']} " + " ".join(f"{k}={v:.1f}" for k, v in meta["timings"].items()))


if __name__ == "__main__":
    main()
