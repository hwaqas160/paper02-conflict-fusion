"""E21 (exploratory): retrain the CAMERA detector in domain (Faster R-CNN R50-FPN v2 fine-tuned on TruckScenes Car boxes).

Cross-fitted exactly like ts_detector.py: scenes are split into the two detector folds stored in prep.pkl; detector k is
trained on fold k and applied only to the other fold. 2D training boxes are the projections of the 3D Car labels.
Everything resumes after interruption (checkpoint every 100 iterations, one npz per detected frame).

    python -m src.e21_camera_train train --fold 0 [--threads 24] [--stride 6] [--epochs 2]
    python -m src.e21_camera_train detect [--workers 6] [--threads 4]
    python -m src.e21_camera_train build  [--workers 8]       # lift, recompute probes -> cache_cam
"""
import argparse
import multiprocessing as mp
import sys
import time
import traceback

import cv2
import numpy as np

from .common import env_threads, load_config, load_json, resolve

CAM_SCALE = 0.5
_W = {}


def out_dir(cfg):
    d = resolve(cfg, "results_root") / "ts_detector" / "camera"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corners(b):
    x, y, z, l, w, h, yaw = b
    dx, dy, dz = np.meshgrid([-l / 2, l / 2], [-w / 2, w / 2], [-h / 2, h / 2], indexing="ij")
    c = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], 1)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return c @ R.T + np.array([x, y, z])


def boxes_2d(gt, T_cam, P, W, H):
    out = []
    for b, c in zip(gt["boxes"], gt["cls"]):
        if c != 0:
            continue
        pc = (T_cam @ np.c_[corners(b), np.ones(8)].T).T[:, :3]
        if (pc[:, 2] < 1.0).any():
            continue
        uv = (P[:3, :3] @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            out.append([x0, y0, x1, y1])
    return np.asarray(out, np.float32).reshape(-1, 4)


def make_model(weights=None):
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if weights is None else None,
        weights_backbone=None, trainable_backbone_layers=2, box_score_thresh=0.05, box_detections_per_img=100)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


def sample(fr, ann):
    from .truckscenes_data import camera_geometry, read_camera, read_ground_truth
    img_full = read_camera(fr)
    h, w = int(img_full.shape[0] * CAM_SCALE), int(img_full.shape[1] * CAM_SCALE)
    img = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
    T_cam, P = camera_geometry(fr)
    P[:2, :3] *= CAM_SCALE
    return img, T_cam, P, read_ground_truth(fr, ann)


def train(cfg, fold, threads, stride, epochs):
    import torch
    env_threads(threads)
    torch.set_num_threads(threads)
    from .truckscenes_data import annotations_by_sample, build_ts_index
    from .ts_detector import load_prep
    fold_of = load_prep(cfg)["fold"]
    ann = annotations_by_sample()
    frames = [f for f in build_ts_index() if f.has_data() and fold_of[f.scene] == fold][::stride]
    ck = out_dir(cfg) / f"cam_fold{fold}.pt"
    m = make_model()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)
    it0 = 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu")
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        it0 = s["it"]
    total = len(frames) * epochs
    rng = np.random.default_rng(fold)
    order = np.concatenate([rng.permutation(len(frames)) for _ in range(epochs)])
    m.train()
    t0 = time.time()
    for it in range(it0, total):
        fr = frames[order[it]]
        img, T_cam, P, gt = sample(fr, ann)
        bx = boxes_2d(gt, T_cam, P, img.shape[1], img.shape[0])
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        tgt = {"boxes": torch.from_numpy(bx), "labels": torch.ones(len(bx), dtype=torch.int64)}
        for g in opt.param_groups:
            g["lr"] = 0.01 * (0.1 if it > 0.7 * total else 1.0)
        loss = sum(m([t], [tgt]).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 20 == 0:
            print(f"fold{fold} it {it + 1}/{total} loss {float(loss):.3f} {(time.time() - t0) / (it + 1 - it0):.1f}s/it", flush=True)
        if (it + 1) % 100 == 0 or it + 1 == total:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "it": it + 1}, ck)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "it": total}, out_dir(cfg) / f"cam_fold{fold}_final.pt")
    print("train done", fold, flush=True)


def _init_det(cfg, threads):
    import torch
    env_threads(threads)
    torch.set_num_threads(threads)
    from .truckscenes_data import build_ts_index
    from .ts_detector import load_prep
    _W["fold"] = load_prep(cfg)["fold"]
    _W["by_id"] = {f.token: f for f in build_ts_index()}
    _W["m"] = {}
    for k in (0, 1):
        m = make_model()
        m.load_state_dict(torch.load(out_dir(cfg) / f"cam_fold{k}_final.pt", map_location="cpu")["model"])
        _W["m"][k] = m.eval()
    _W["out"] = out_dir(cfg) / "dets"
    _W["out"].mkdir(exist_ok=True)


def _detect_one(token):
    import torch
    from .truckscenes_data import read_camera
    fr = _W["by_id"][token]
    img_full = read_camera(fr)
    img = cv2.resize(img_full, (int(img_full.shape[1] * CAM_SCALE), int(img_full.shape[0] * CAM_SCALE)), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        o = _W["m"][1 - _W["fold"][fr.scene]]([torch.from_numpy(img).permute(2, 0, 1).float().div(255)])[0]
    tmp = _W["out"] / f"{token}.tmp.npz"
    np.savez_compressed(tmp, boxes=o["boxes"].numpy(), scores=o["scores"].numpy())
    tmp.replace(_W["out"] / f"{token}.npz")
    return token


def _safe(fn, x):
    try:
        return fn(x), None
    except Exception:
        return None, traceback.format_exc()


def _safe_detect(x):
    return _safe(_detect_one, x)


def detect(cfg, workers, threads):
    from .truckscenes_data import build_ts_index
    d = out_dir(cfg) / "dets"
    d.mkdir(exist_ok=True)
    tokens = [f.token for f in build_ts_index() if f.has_data() and not (d / f"{f.token}.npz").exists()]
    print(len(tokens), "frames to detect", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init_det, initargs=(cfg, threads)) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(_safe_detect, tokens, chunksize=4), 1):
            if e:
                print(e, flush=True)
            if i % 500 == 0:
                print("detect", i, "/", len(tokens), flush=True)


def _init_build(cfg, ground_z, threads):
    from . import e8c_rebuild as R
    from .detectors.camera_frcnn import CameraDetector
    R._init(cfg, ground_z, threads)
    _W["cam"] = CameraDetector(device="cpu", ground_z=ground_z, bottom_sigma_px=cfg["detectors"]["camera"]["lift_ground_sigma_px"] * CAM_SCALE)
    _W["dets"] = out_dir(cfg) / "dets"


def _build_one(item):
    from . import e8c_rebuild as R
    from .candidates import load_sample
    from .truckscenes_data import camera_geometry, read_camera, read_lidar
    path, out = item
    meta, s = load_sample(path)
    fr = R._W["by_id"][meta["frame_id"]]
    T_cam, P = camera_geometry(fr)
    P[:2, :3] *= CAM_SCALE
    H, W = int(read_camera(fr).shape[0] * CAM_SCALE), int(read_camera(fr).shape[1] * CAM_SCALE)
    z = np.load(_W["dets"] / f"{fr.token}.npz")
    raw = {"boxes": z["boxes"], "scores": z["scores"], "labels": np.full(len(z["scores"]), 3)}
    d = _W["cam"].lift(raw, T_cam, P, W, H)
    keep = s["camera_labels"] != 0
    n_new = len(d["scores"])
    s = dict(s)
    cat = lambda old, new: np.concatenate([old[keep], np.asarray(new, old.dtype).reshape((-1,) + old.shape[1:])])
    s["camera_boxes"] = cat(s["camera_boxes"], d["boxes"])
    s["camera_scores"] = cat(s["camera_scores"], d["scores"])
    s["camera_labels"] = cat(s["camera_labels"], np.zeros(n_new))
    for k in ("range_ground", "range_size", "range_sigma", "truncated"):
        s[f"camera_{k}"] = cat(s[f"camera_{k}"], d[k])
    s["camera_boxes2d"] = cat(s["camera_boxes2d"], d["boxes2d"])
    s["camera_probes"] = R.probes_for(fr, read_lidar(fr), s["camera_boxes"].astype(np.float64))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(repr(meta)), **s)
    tmp.replace(out)
    return fr.token


def _safe_build(x):
    return _safe(_build_one, x)


def build(cfg, workers):
    from . import e8c_rebuild as R
    base = resolve(cfg, "results_root") / "ts_detector"
    src, dst = base / "cache_full2" / "truckscenes", base / "cache_cam" / "truckscenes"
    ground_z = load_json(R.paths(cfg)[0] / "manifest.json")["ground_z"]
    items = [(p, dst / p.parent.name / p.name) for p in sorted(src.glob("*/*.npz")) if not (dst / p.parent.name / p.name).exists()]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.json").write_text((src / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(len(items), "frames to build", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init_build, initargs=(cfg, ground_z, 2)) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(_safe_build, items, chunksize=2), 1):
            if e:
                print(e, flush=True)
            if i % 1000 == 0:
                print("build", i, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "detect", "build"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "train":
        train(cfg, a.fold, a.threads, a.stride, a.epochs)
    elif a.cmd == "detect":
        detect(cfg, a.workers, a.threads if a.threads < 8 else 4)
    else:
        build(cfg, a.workers)
