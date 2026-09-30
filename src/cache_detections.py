"""THE offline inference pass (PLAN.md §8): run frozen detectors ONCE per (frame, weather), cache all.

For every labelled frame and every (condition, severity):
    raw sensors -> seeded weather degradation -> PointPillars / Faster R-CNN / CFAR
               -> degradation probes at every detection and every GT box -> one .npz

Resumable (existing outputs are skipped), multi-process, ordered so that a partial run is already
usable: all frames for the priority conditions first.

    python -m src.cache_detections --workers 3 --threads 6
    python -m src.cache_detections --stride 3 --conditions clear0 snow2 fog2   # dev subset
"""
import argparse
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .boreas_data import IMG_H, IMG_W, assign_splits, build_index, load_calib, read_camera, read_ground_truth, \
    read_lidar, read_radar
from .common import MODALITIES, condition_tag, env_threads, get_logger, iter_conditions, load_config, resolve, \
    save_json, source_tree_hash
from .degradation import FRAME_FEATURES, ImageStats, compute_probes, lidar_frame_features
from .geometry import in_roi

PRIORITY = ["clear0", "snow2", "fog2", "rain2", "night2", "snow3", "fog3", "rain3", "snow1", "fog1", "rain1"]
CAM_SCALE = 0.5  # detector input is half resolution for every condition (clear included)

_W = {}  # per-process worker state


def weather_params(cfg):
    return {k: {int(s): float(v) for s, v in cfg["weather"][k].items()}
            for k in ("fog_visibility_m", "rain_rate_mmh", "snow_rate_mmh")}


def estimate_ground(frames):
    bottoms = []
    for f in frames:
        g = read_ground_truth(f)
        if len(g["boxes"]):
            bottoms.append(g["boxes"][:, 2] - g["boxes"][:, 5] / 2)
    return float(np.median(np.concatenate(bottoms)))


def _init_worker(cfg, ground_z, threads):
    env_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from .common import pick_device
    from .detectors.camera_frcnn import CameraDetector
    from .detectors.pointpillars import PointPillarsDetector
    from .detectors.radar_cfar import RadarDetector
    dcfg = cfg["detectors"]
    device = pick_device(dcfg["device"])
    ck = resolve(cfg, "checkpoints") / dcfg["lidar"]["checkpoint"]
    _W["lidar"] = PointPillarsDetector(ck, device=device, z_shift=dcfg["lidar"]["sensor_height_kitti"] * -1 - ground_z,
                                       intensity_scale=1 / 255.0, score_thr=dcfg["lidar"]["score_thr"],
                                       nms_iou=dcfg["lidar"]["nms_iou"], max_det=dcfg["lidar"]["max_det"], legacy=True)
    _W["camera"] = CameraDetector(device=device, score_thr=dcfg["camera"]["score_thr"], max_det=dcfg["camera"]["max_det"],
                                  ground_z=ground_z, bottom_sigma_px=dcfg["camera"]["lift_ground_sigma_px"] * CAM_SCALE)
    _W["radar"] = RadarDetector(dcfg["radar"], cfg["roi"], ground_z)
    _W["cfg"], _W["ground_z"], _W["device"] = cfg, ground_z, str(device)


def process(item):
    frame, cond, sev, out_path = item
    from . import weather as W
    cfg, ground_z = _W["cfg"], _W["ground_z"]
    wcfg = weather_params(cfg)
    timings = {}
    t0 = time.time()
    calib = load_calib(frame.root)
    P_half = calib["P"].copy()
    P_half[:2, :3] *= CAM_SCALE
    w_half, h_half = int(IMG_W * CAM_SCALE), int(IMG_H * CAM_SCALE)
    T_cam_lidar = frame.T_cam_lidar()

    pts_clean = read_lidar(frame)
    img = cv2.resize(read_camera(frame), (w_half, h_half), interpolation=cv2.INTER_AREA)
    rad, rad_prev = read_radar(frame), read_radar(frame, prev=True)
    timings["load"] = time.time() - t0

    t0 = time.time()
    pts = W.degrade_lidar(pts_clean, cond, sev, wcfg, W.rng_for(frame.frame_id, cond, sev, "lidar"))
    img_d = W.degrade_camera(img, cond, sev, wcfg, W.rng_for(frame.frame_id, cond, sev, "camera"),
                             points_clean=pts_clean, T_cam_lidar=T_cam_lidar, P=P_half)
    rad_d = W.degrade_radar(rad, cond, sev, wcfg, W.rng_for(frame.frame_id, cond, sev, "radar"))
    rad_prev_d = W.degrade_radar(rad_prev, cond, sev, wcfg, W.rng_for(frame.frame_id, cond, sev, "radar_prev"))
    timings["weather"] = time.time() - t0

    t0 = time.time()
    det_l = _W["lidar"](pts)
    timings["lidar"] = time.time() - t0
    t0 = time.time()
    det_c = _W["camera"](img_d, T_cam_lidar, P_half)
    timings["camera"] = time.time() - t0
    t0 = time.time()
    det_r = _W["radar"](rad_d, frame.T_lidar_radar(), rad_prev_d, frame.T_lidar_radar(prev=True))
    timings["radar"] = time.time() - t0

    # PointPillars class order (Ped, Cyc, Car) -> project ids (Car 0, Ped 1, Cyc 2)
    det_l["labels"] = np.array([2, 1, 0])[det_l["labels"]] if len(det_l["labels"]) else det_l["labels"]

    t0 = time.time()
    from .detectors.radar_cfar import power_map_lidar_frame
    grid = _W["radar"].grid
    rmap = power_map_lidar_frame(rad_d, frame.T_lidar_radar(), grid, ground_z)
    stats = ImageStats(img_d)
    gt = read_ground_truth(frame, cfg["roi"]["min_gt_points"])
    probe = lambda boxes: compute_probes(boxes, pts, stats, T_cam_lidar, P_half, rmap, grid, det_r["occupancy"])
    frame_feats = np.array(lidar_frame_features(pts, ground_z) + stats.global_feats +
                           [det_r["frame"]["noise_floor_median"], np.log1p(det_r["frame"]["n_hits_roi"]),
                            det_r["frame"]["hit_mean_power"]], np.float32)
    assert len(frame_feats) == len(FRAME_FEATURES)
    timings["probes"] = time.time() - t0

    arrays = {"frame_feats": frame_feats,
              "gt_boxes": gt["boxes"], "gt_cls": gt["cls"], "gt_npts": gt["npts"], "gt_dontcare": gt["dontcare"],
              "gt_probes": probe(gt["boxes"]), "gt_in_roi": in_roi(gt["boxes"], cfg["roi"])}
    for name, d in zip(MODALITIES, (det_c, det_l, det_r)):
        arrays[f"{name}_boxes"] = d["boxes"].astype(np.float32)
        arrays[f"{name}_scores"] = d["scores"].astype(np.float32)
        arrays[f"{name}_labels"] = d["labels"].astype(np.int8)
        arrays[f"{name}_probes"] = probe(d["boxes"])
    for k in ("range_ground", "range_size", "range_sigma", "truncated", "boxes2d"):
        arrays[f"camera_{k}"] = np.asarray(det_c[k], np.float32)
    arrays["radar_feats"] = det_r["feats"].astype(np.float32)
    meta = {"frame_id": frame.frame_id, "scene": frame.scene, "split": frame.split, "condition": cond,
            "severity": sev, "cam_ts": frame.cam_ts, "radar_ts": frame.radar_ts, "timings": timings,
            "n_points": int(len(pts)), "device": _W["device"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(repr(meta)), **arrays)
    tmp.replace(out_path)
    return meta


def _safe_process(item, retries=3):
    """This machine is shared; host RAM fluctuates and torch allocations fail transiently.
    Back off and retry instead of dropping the item."""
    import gc
    for attempt in range(retries):
        try:
            return process(item), None
        except (MemoryError, np.core._exceptions._ArrayMemoryError, RuntimeError) as e:
            if attempt == retries - 1:
                return None, f"{item[0].frame_id} {item[1]}{item[2]} (after {retries} tries): {e}"
            gc.collect()
            time.sleep(20 * (attempt + 1))
        except Exception:
            return None, f"{item[0].frame_id} {item[1]}{item[2]}: {traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--stride", type=int, default=None, help="override splits.frame_stride")
    ap.add_argument("--conditions", nargs="*", default=None, help="e.g. clear0 snow2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="override detectors.device (cpu | cuda | auto)")
    ap.add_argument("--refresh-frames", action="store_true", help="re-derive the frame list instead of reusing the manifest")
    args = ap.parse_args()

    cfg = load_config(args.config, {"detectors.device": args.device} if args.device else None)
    cache_root = resolve(cfg, "cache_root") / "boreas"
    log = get_logger("cache", resolve(cfg, "results_root") / "logs" / "cache_detections.log")
    all_frames = build_index(resolve(cfg, "boreas_root"), require_data=True)
    stride = args.stride or cfg["splits"]["frame_stride"]
    frames = assign_splits(all_frames, cfg["splits"]["scene_fractions"], cfg["seed"], stride)
    # The frame list is FROZEN in the manifest on first run. Without this, a resume after more data has
    # been downloaded silently caches a different frame set per condition, and conditions stop being
    # comparable. --refresh-frames deliberately re-derives it (then re-cache every condition).
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists() and not args.refresh_frames:
        from .common import load_json
        frozen = load_json(manifest_path).get("frame_ids")
        if frozen:
            by_id = {f.frame_id: f for f in assign_splits(all_frames, cfg["splits"]["scene_fractions"], cfg["seed"], 1)}
            missing = [i for i in frozen if i not in by_id]
            if missing:
                log.warning(f"{len(missing)} frozen frames are no longer available; keeping the rest")
            frames = [by_id[i] for i in frozen if i in by_id]
            log.info(f"using frozen frame list from manifest ({len(frames)} frames)")
    ground_path = cache_root / "ground_z.json"
    if ground_path.exists():
        from .common import load_json
        ground_z = load_json(ground_path)["ground_z"]
    else:
        ground_z = estimate_ground([f for f in frames if f.split == "train"])
        save_json({"ground_z": ground_z, "estimated_from": "train-split GT box bottoms (median)"}, ground_path)
    tags = [condition_tag(c, s) for c, s in iter_conditions(cfg)]
    wanted = args.conditions or tags
    order = [t for t in PRIORITY if t in wanted] + [t for t in wanted if t not in PRIORITY]
    parse = lambda t: (t.rstrip("0123456789"), int(t[len(t.rstrip("0123456789")):]))
    items = []
    for tag in order:
        cond, sev = parse(tag)
        for f in frames:
            out = cache_root / tag / f"{f.frame_id}.npz"
            if not out.exists():
                items.append((f, cond, sev, out))
    if args.limit:
        items = items[: args.limit]
    save_json({"source_hash": source_tree_hash(), "n_frames": len(frames), "stride": stride, "conditions": order,
               "frame_ids": [f.frame_id for f in frames],
               "splits": {s: sum(f.split == s for f in frames) for s in ("train", "cal", "test")},
               "scenes": {s: sorted({f.scene for f in frames if f.split == s}) for s in ("train", "cal", "test")},
               "ground_z": ground_z}, cache_root / "manifest.json")
    log.info(f"{len(frames)} frames x {len(order)} conditions -> {len(items)} items to run "
             f"(workers={args.workers}, threads={args.threads}, ground_z={ground_z:.3f})")
    if not items:
        return
    t_start, done, failed = time.time(), 0, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg, ground_z, args.threads)) as pool:
        for meta, err in pool.imap_unordered(_safe_process, items, chunksize=1):
            if err:
                failed += 1
                log.error(err)
                continue
            done += 1
            if done % 10 == 0 or done == len(items):
                rate = (time.time() - t_start) / done
                tm = meta["timings"]
                log.info(f"{done}/{len(items)} ({failed} failed) {rate:.1f}s/item ETA {(len(items) - done) * rate / 3600:.1f}h "
                         f"| last {meta['condition']}{meta['severity']} " +
                         " ".join(f"{k}={v:.1f}" for k, v in tm.items()))


if __name__ == "__main__":
    main()
