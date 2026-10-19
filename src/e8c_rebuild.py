"""E8-C step 2: swap the cross-fitted in-domain LiDAR Car detections into the TruckScenes cache.

Three phases, so the GPU is used by ONE process (several processes on one shared 5 GB card time-slice badly):
  detect      single GPU process; CPU workers prefetch pillars; writes results/ts_detector/dets/<token>.npz (resumable)
  build       CPU-only workers: merge detections into the cached sample, recompute LiDAR probes, write the new cache
  candidates  build candidate tables from the new cache

For each frame the LiDAR Car boxes/scores come from the detector trained on the OTHER detector fold (ts_detector.py);
LiDAR Pedestrian/Cyclist boxes stay as cached; probes are recomputed at the merged boxes with the same compute_probes call
and inputs as ts_cache.py. Camera, radar, frame features and ground truth are copied unchanged. The original cache and
candidates are never touched.

    python -m src.e8c_rebuild check [--n 12]          # recomputing probes for the ORIGINAL boxes must match the cache
    python -m src.e8c_rebuild detect [--workers 6] [--limit N]
    python -m src.e8c_rebuild build [--workers 10]
    python -m src.e8c_rebuild candidates
"""
import argparse
import multiprocessing as mp
import time
import traceback

import cv2
import numpy as np

from .common import env_threads, get_logger, load_config, load_json, resolve, save_json

CAM_SCALE = 0.5
_W = {}


def paths(cfg):
    base = resolve(cfg, "results_root") / "ts_detector"
    return resolve(cfg, "cache_root") / "truckscenes", base / "cache" / "truckscenes", base / "dets"


# ---------------------------------------------------------------------------------------------- CPU workers
def _init(cfg, ground_z, threads):
    env_threads(threads)
    from .detectors.radar_points4d import RadarPoints4D
    from .truckscenes_data import build_ts_index
    from .ts_detector import load_prep
    _W["fold"] = load_prep(cfg)["fold"]
    _W["by_id"] = {f.token: f for f in build_ts_index()}
    _W["radar"] = RadarPoints4D(cfg["detectors"]["radar"], cfg["roi"], ground_z)
    _W["dets"] = paths(cfg)[2]


def probes_for(fr, pts, boxes):
    """compute_probes exactly as ts_cache.process, for arbitrary boxes of one frame."""
    from .degradation import ImageStats, compute_probes
    from .truckscenes_data import camera_geometry, n_pseudo_rings, read_camera, read_radar
    img_full = read_camera(fr)
    h, w = int(img_full.shape[0] * CAM_SCALE), int(img_full.shape[1] * CAM_SCALE)
    img = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
    T_cam, P = camera_geometry(fr)
    P[:2, :3] *= CAM_SCALE
    det_r = _W["radar"](read_radar(fr), read_radar(fr, prev=True))
    stats = ImageStats(img)
    return compute_probes(boxes, pts, stats, T_cam, P, det_r["power_map"], _W["radar"].grid, det_r["occupancy"],
                          n_rings=n_pseudo_rings(pts))


def _check(path):
    from .candidates import load_sample
    from .truckscenes_data import read_lidar
    meta, s = load_sample(path)
    fr = _W["by_id"][meta["frame_id"]]
    new = probes_for(fr, read_lidar(fr), s["lidar_boxes"].astype(np.float64))
    old = s["lidar_probes"]
    same = np.array_equal(np.isnan(new), np.isnan(old)) and np.allclose(np.nan_to_num(new), np.nan_to_num(old), rtol=1e-3,
                                                                       atol=1e-3)
    return meta["frame_id"], bool(same), float(np.nanmax(np.abs(new - old))) if len(old) else 0.0


def _process(item):
    from .candidates import load_sample
    from .truckscenes_data import read_lidar
    path, out_path = item
    meta, s = load_sample(path)
    fr = _W["by_id"][meta["frame_id"]]
    det = np.load(_W["dets"] / f"{fr.token}.npz")
    keep = s["lidar_labels"] != 0                         # project id 0 = Car; keep Pedestrian/Cyclist as cached
    boxes = np.concatenate([s["lidar_boxes"][keep], det["boxes"]]).astype(np.float64).reshape(-1, 7)
    s = dict(s)
    s["lidar_boxes"] = boxes.astype(np.float32)
    s["lidar_scores"] = np.concatenate([s["lidar_scores"][keep], det["scores"]]).astype(np.float32)
    s["lidar_labels"] = np.concatenate([s["lidar_labels"][keep], np.zeros(len(det["scores"]))]).astype(np.int8)
    s["lidar_probes"] = probes_for(fr, read_lidar(fr), boxes)
    meta = dict(meta)
    meta["lidar_detector"] = "in_domain_crossfit"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(repr(meta)), **s)
    tmp.replace(out_path)
    return meta["frame_id"]


def _safe(fn, item):
    try:
        return fn(item), None
    except Exception:
        return None, traceback.format_exc()


def _safe_check(item):
    return _safe(_check, item)


def _safe_process(item):
    return _safe(_process, item)


# ---------------------------------------------------------------------------------------------- GPU phase
class _VoxData:
    def __init__(self, frames, ground_z):
        self.frames, self.ground_z = frames, ground_z

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from .detectors.pointpillars import voxelize
        from .truckscenes_data import read_lidar
        from .ts_detector import preprocess
        fr = self.frames[i]
        return fr.token, voxelize(preprocess(read_lidar(fr), self.ground_z))


def detect(cfg, ground_z, workers, limit, log):
    import torch

    from .truckscenes_data import build_ts_index
    from .ts_detector import InDomainCarDetector, load_prep, out_dir
    prepd = load_prep(cfg)
    dets_dir = paths(cfg)[2]
    dets_dir.mkdir(parents=True, exist_ok=True)
    frames = [f for f in build_ts_index() if f.has_data() and not (dets_dir / f"{f.token}.npz").exists()]
    frames = frames[:limit] if limit else frames
    log.info(f"detect: {len(frames)} frames to run")
    t0, done, wait, comp = time.time(), 0, 0.0, 0.0
    for k in (0, 1):                                    # frames of fold k are scored by the detector trained on fold 1-k
        part = [f for f in frames if prepd["fold"][f.scene] == k]
        if not part:
            continue
        det = InDomainCarDetector(out_dir(cfg) / f"fold{1 - k}_final.pt", ground_z, "cuda")
        dl = torch.utils.data.DataLoader(_VoxData(part, ground_z), batch_size=None, shuffle=False, num_workers=workers,
                                         prefetch_factor=4)
        t_prev = time.time()
        for token, (feats, num, coors) in dl:
            t_got = time.time()
            wait += t_got - t_prev
            out = det.from_voxels(feats, num, coors)
            comp += time.time() - t_got
            tmp = dets_dir / f"{token}.tmp.npz"
            np.savez_compressed(tmp, boxes=out["boxes"].astype(np.float32), scores=out["scores"].astype(np.float32))
            tmp.replace(dets_dir / f"{token}.npz")
            done += 1
            t_prev = time.time()
            if done % 100 == 0:
                r = (time.time() - t0) / done
                log.info(f"detect {done}/{len(frames)} {r:.2f}s/frame (wait {wait / done:.2f}s, compute {comp / done:.2f}s) ETA {(len(frames) - done) * r / 3600:.2f}h")
        del det
    log.info("detect: done")


def _init_det(cfg, ground_z, threads):
    """CPU detection worker. On the shared 5 GB GPU (another job at ~80% utilisation) a forward pass took 2.7 s; on 6 CPU
    threads it takes ~0.9 s, and the machine has 36 cores, so detection runs as several CPU processes."""
    env_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from .truckscenes_data import build_ts_index
    from .ts_detector import InDomainCarDetector, load_prep, out_dir
    _W["fold"] = load_prep(cfg)["fold"]
    _W["by_id"] = {f.token: f for f in build_ts_index()}
    _W["det"] = {k: InDomainCarDetector(out_dir(cfg) / f"fold{k}_final.pt", ground_z, "cpu") for k in (0, 1)}
    _W["dets"] = paths(cfg)[2]


def _detect_one(token):
    from .truckscenes_data import read_lidar
    fr = _W["by_id"][token]
    out = _W["det"][1 - _W["fold"][fr.scene]](read_lidar(fr))   # trained on the other fold, never on this scene
    tmp = _W["dets"] / f"{token}.tmp.npz"
    np.savez_compressed(tmp, boxes=out["boxes"].astype(np.float32), scores=out["scores"].astype(np.float32))
    tmp.replace(_W["dets"] / f"{token}.npz")
    return token


def _safe_detect(token):
    return _safe(_detect_one, token)


def detect_cpu(cfg, ground_z, workers, threads, limit, log):
    from .truckscenes_data import build_ts_index
    dets_dir = paths(cfg)[2]
    dets_dir.mkdir(parents=True, exist_ok=True)
    tokens = [f.token for f in build_ts_index() if f.has_data() and not (dets_dir / f"{f.token}.npz").exists()]
    tokens = tokens[:limit] if limit else tokens
    log.info(f"detect (cpu): {len(tokens)} frames, {workers} workers x {threads} threads")
    t0, done, failed = time.time(), 0, 0
    with mp.get_context("spawn").Pool(workers, initializer=_init_det, initargs=(cfg, ground_z, threads)) as pool:
        for res, err in pool.imap_unordered(_safe_detect, tokens, chunksize=4):
            if err:
                failed += 1
                log.error(err)
                continue
            done += 1
            if done % 200 == 0 or done == len(tokens):
                r = (time.time() - t0) / done
                log.info(f"detect {done}/{len(tokens)} ({failed} failed) {r:.2f}s/frame ETA {(len(tokens) - done) * r / 3600:.2f}h")
    log.info("detect: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "detect", "build", "candidates"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    cfg = load_config()
    src, dst, dets = paths(cfg)
    log = get_logger("e8c_rebuild", resolve(cfg, "results_root") / "logs" / "e8c_rebuild.log")
    ground_z = load_json(src / "manifest.json")["ground_z"]
    if a.cmd == "detect":
        if a.device == "cuda":
            detect(cfg, ground_z, a.workers, a.limit, log)
        else:
            detect_cpu(cfg, ground_z, a.workers, a.threads, a.limit, log)
        return
    if a.cmd == "check":
        files = sorted(src.glob("*/*.npz"))
        files = files[:: max(1, len(files) // a.n)][: a.n]
        with mp.get_context("spawn").Pool(min(a.workers, len(files)), initializer=_init,
                                          initargs=(cfg, ground_z, a.threads)) as pool:
            res = [r for r, e in pool.imap_unordered(_safe_check, files) if r]
        print(f"probe equivalence: {sum(r[1] for r in res)}/{len(res)} frames reproduce the cached LiDAR probes; max abs "
              f"diff {max(r[2] for r in res):.2e}")
        return
    if a.cmd == "build":
        files = sorted(src.glob("*/*.npz"))
        items = [(p, dst / p.parent.name / p.name) for p in files
                 if not (dst / p.parent.name / p.name).exists() and (dets / f"{p.stem}.npz").exists()]
        items = items[: a.limit] if a.limit else items
        dst.mkdir(parents=True, exist_ok=True)
        save_json({"ground_z": ground_z, "note": "cross-fitted in-domain LiDAR Car detections (E8-C)"}, dst / "manifest.json")
        log.info(f"build: {len(items)} frames to write ({len(files)} total)")
        t0, done, failed = time.time(), 0, 0
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg, ground_z, a.threads)) as pool:
            for res, err in pool.imap_unordered(_safe_process, items, chunksize=2):
                if err:
                    failed += 1
                    log.error(err)
                    continue
                done += 1
                if done % 200 == 0 or done == len(items):
                    r = (time.time() - t0) / done
                    log.info(f"build {done}/{len(items)} ({failed} failed) {r:.2f}s/frame ETA {(len(items) - done) * r / 3600:.2f}h")
        return
    from .candidates import build_table, save_table
    out_root = resolve(cfg, "results_root") / "ts_detector" / "candidates" / "truckscenes"
    summary = {}
    for tag in sorted(p.name for p in dst.iterdir() if p.is_dir()):
        tbl = build_table(dst / tag, cfg)
        if tbl is None:
            continue
        save_table(tbl, out_root / f"{tag}.npz")
        c = tbl["cand"]
        summary[tag] = {"frames": len(tbl["frames"]), "candidates": int(len(c["y"])), "positives": int(c["y"].sum())}
        log.info(f"{tag}: {summary[tag]}")
    save_json(summary, out_root / "summary.json")


if __name__ == "__main__":
    main()
