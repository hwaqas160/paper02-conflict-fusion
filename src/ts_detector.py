"""In-domain Car LiDAR detector for MAN TruckScenes: the KITTI PointPillars checkpoint fine-tuned on TruckScenes labels.

Purpose (E8-C, exploratory): the paper's LiDAR detector is a KITTI checkpoint applied to a truck-mounted LiDAR it was
never trained on. This module gives a strong, matched detector so the per-object advantage can be tested when the
frozen detector is no longer weak.

Leakage control. The router is cross-validated over every scene, so a detector trained on all scenes would have seen the
labels of the scenes it is later scored on. Scenes are therefore split into two detector folds (stratified by weather);
detector k is trained on fold k and applied ONLY to the other fold (cross-fitting). Every frame's detections come from a
detector that never saw its scene.

Only the Car anchors of the head are trained (assignment by centre distance, focal + smooth-L1 + sin-difference + direction
losses, the SECOND/PointPillars recipe without rotated-IoU matching). Pedestrian/Cyclist outputs are left as they were.

    python -m src.ts_detector prep
    python -m src.ts_detector train --fold 0 --epochs 4 [--batch 4] [--limit N]
    python -m src.ts_detector eval --fold 0 [--stride 10]
"""
import argparse
import json
import math
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import load_config, load_json, resolve
from .detectors.pointpillars import (DIR_LIMIT_OFFSET, DIR_OFFSET, GRID, PointPillarsNet, decode, limit_period,
                                     make_anchors, voxelize)
from .geometry import rotated_nms_bev

CAR_ROWS = slice(4, 6)         # anchors 4, 5 of the 6 per location are the Car anchors (rotations 0, 1.57)
CAR_COL = 2                    # Car logit column
ANCH_L, ANCH_W, ANCH_H = 3.9, 1.6, 1.56
ANCH_Z_BOTTOM = -1.78
ROT = np.array([0.0, 1.57])
KITTI_HEIGHT = 1.73


def out_dir(cfg=None):
    cfg = cfg or load_config()
    d = resolve(cfg, "results_root") / "ts_detector"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- data
def detector_folds(frames, seed=0):
    """scene -> detector fold (0/1), stratified by the scene's weather condition."""
    cond = {}
    for f in frames:
        cond[f.scene] = f.condition
    rng = np.random.default_rng(seed)
    fold = {}
    for c in sorted(set(cond.values())):
        scenes = sorted(s for s, v in cond.items() if v == c)
        for i, s in enumerate(rng.permutation(scenes)):
            fold[str(s)] = i % 2
    return fold


def prep(cfg=None):
    from .truckscenes_data import annotations_by_sample, build_ts_index, read_ground_truth
    cfg = cfg or load_config()
    frames = [f for f in build_ts_index() if f.has_data()]
    ann = annotations_by_sample()
    gt = {}
    for f in frames:
        g = read_ground_truth(f, ann)
        car = g["cls"] == 0
        gt[f.token] = {"car": g["boxes"][car].astype(np.float32), "dc": g["dontcare"].astype(np.float32)}
    fold = detector_folds(frames)
    ground_z = load_json(resolve(cfg, "cache_root") / "truckscenes" / "manifest.json")["ground_z"]
    with open(out_dir(cfg) / "prep.pkl", "wb") as fh:
        pickle.dump({"gt": gt, "fold": fold, "ground_z": ground_z}, fh)
    n = {k: sum(1 for f in frames if fold[f.scene] == k) for k in (0, 1)}
    print(f"prep: {len(frames)} frames, ground_z={ground_z:.3f}, frames per detector fold {n}, "
          f"{sum(len(v['car']) for v in gt.values())} Car labels")


def load_prep(cfg=None):
    with open(out_dir(cfg) / "prep.pkl", "rb") as fh:
        return pickle.load(fh)


def preprocess(pts, ground_z):
    p = np.asarray(pts[:, :4], np.float32).copy()
    p[:, 2] += -KITTI_HEIGHT - ground_z
    p[:, 3] = np.clip(p[:, 3] / 255.0, 0, 1)
    return p


class DetData(torch.utils.data.Dataset):
    def __init__(self, frames, prepd, train):
        self.frames, self.gt, self.ground_z, self.train = frames, prepd["gt"], prepd["ground_z"], train
        self.z_shift = -KITTI_HEIGHT - self.ground_z

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from .truckscenes_data import read_lidar
        fr = self.frames[i]
        p = preprocess(read_lidar(fr), self.ground_z)
        g = self.gt[fr.token]
        car, dc = g["car"].copy(), g["dc"].copy()
        car[:, 2] += self.z_shift
        if self.train:
            rng = np.random.default_rng()
            th = rng.uniform(-0.2, 0.2)
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]], np.float32)
            p[:, :2] = p[:, :2] @ R.T
            car[:, :2] = car[:, :2] @ R.T
            car[:, 6] += th
            dc[:, :2] = dc[:, :2] @ R.T
            if rng.random() < 0.5:
                p[:, 1] *= -1
                car[:, 1] *= -1
                car[:, 6] *= -1
                dc[:, 1] *= -1
        feats, num, coors = voxelize(p)
        return feats, num, coors, car, dc


def collate(batch):
    feats, num, coors = [], [], []
    for b, (f, n, c, _, _) in enumerate(batch):
        feats.append(torch.from_numpy(f))
        num.append(torch.from_numpy(n))
        cc = torch.from_numpy(c)
        cc[:, 0] = b
        coors.append(cc)
    return (torch.cat(feats), torch.cat(num), torch.cat(coors), [b[3] for b in batch], [b[4] for b in batch])


# ---------------------------------------------------------------------------------------------- model
def load_net(checkpoint=None, state=None, device="cuda"):
    net = PointPillarsNet(legacy=True)
    if state is None:
        cfg = load_config()
        ck = resolve(cfg, "checkpoints") / cfg["detectors"]["lidar"]["checkpoint"]
        state = torch.load(ck, map_location="cpu")["state_dict"]
    net.load_state_dict(state, strict=False)
    return net.to(device)


def encode_pillars(net, feats, num, coors, chunk=6000):
    """Pillar feature net in gradient-checkpointed chunks. Dense frames (three merged LiDARs) hit the 40k-pillar cap and the
    (V, 32, 64) activations alone need several GB, which thrashes a 5 GB card; chunking keeps them small."""
    if not torch.is_grad_enabled() or len(feats) <= chunk:
        return net.voxel_encoder(feats, num, coors)
    from torch.utils.checkpoint import checkpoint
    return torch.cat([checkpoint(net.voxel_encoder, feats[i:i + chunk], num[i:i + chunk], coors[i:i + chunk],
                                 use_reentrant=False) for i in range(0, len(feats), chunk)])


def forward_batch(net, feats, num, coors, B):
    pillars = encode_pillars(net, feats, num, coors)
    nx, ny = int(GRID[0]), int(GRID[1])
    canvas = pillars.new_zeros(B, ny * nx, 64)
    canvas[coors[:, 0].long(), (coors[:, 2] * nx + coors[:, 3]).long()] = pillars
    canvas = canvas.permute(0, 2, 1).reshape(B, 64, ny, nx)
    return net.bbox_head(net.neck(net.backbone(canvas)))


def car_outputs(cls, reg, dirc):
    B, _, H, W = cls.shape
    c = cls.permute(0, 2, 3, 1).reshape(B, H, W, 6, 3)[:, :, :, CAR_ROWS, CAR_COL]
    r = reg.permute(0, 2, 3, 1).reshape(B, H, W, 6, 7)[:, :, :, CAR_ROWS]
    d = dirc.permute(0, 2, 3, 1).reshape(B, H, W, 6, 2)[:, :, :, CAR_ROWS]
    return c, r, d


_ANCH = {}


def car_anchor_grid(H, W, device):
    if (H, W, str(device)) not in _ANCH:
        a = make_anchors(H, W)[:, :, 2]          # (H, W, 2, 7)
        _ANCH[(H, W, str(device))] = torch.as_tensor(a, dtype=torch.float32, device=device)
    return _ANCH[(H, W, str(device))]


def wrap_pi(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def assign(car, dc, anch):
    """Targets for one sample. car (N,7) [x,y,zc,L,W,H,yaw] in the detector frame. Returns
    pos (H,W,2) bool, valid (H,W,2) bool (False = ignored), reg (H,W,2,7), dirl (H,W,2) long."""
    H, W = anch.shape[:2]
    dev = anch.device
    pos = torch.zeros(H, W, 2, dtype=torch.bool, device=dev)
    valid = torch.ones(H, W, 2, dtype=torch.bool, device=dev)
    reg = torch.zeros(H, W, 2, 7, device=dev)
    dirl = torch.zeros(H, W, 2, dtype=torch.long, device=dev)
    cx, cy = anch[:, :, 0, 0], anch[:, :, 0, 1]
    if len(dc):
        g = torch.as_tensor(dc, device=dev)
        d = torch.sqrt((cx[None] - g[:, 0, None, None]) ** 2 + (cy[None] - g[:, 1, None, None]) ** 2).min(0)[0]
        valid[d <= 2.0] = False
    if len(car) == 0:
        return pos, valid, reg, dirl
    g = torch.as_tensor(car, device=dev)
    N = len(g)
    d = torch.sqrt((cx[None] - g[:, 0, None, None]) ** 2 + (cy[None] - g[:, 1, None, None]) ** 2)   # (N,H,W)
    dmin, idx = d.min(0)
    r_pos = (0.3 * g[:, 3]).clamp(0.9, 1.6)
    r_ign = 0.5 * g[:, 3] + 0.8
    yaw = g[:, 6]
    bin_n = (torch.abs(torch.sin(yaw)) > torch.abs(torch.cos(yaw))).long()          # 0: ~along x, 1: ~along y
    near = dmin <= r_pos[idx]
    for b in (0, 1):
        pos[:, :, b] = near & (bin_n[idx] == b)
    forced = d.view(N, -1).argmin(1)                                                  # every GT gets >= 1 positive
    fy, fx = forced // W, forced % W
    pos[fy, fx, bin_n] = True
    idx = idx.clone()
    idx[fy, fx] = torch.arange(N, device=dev)
    ign = (dmin <= r_ign[idx]).unsqueeze(-1).expand(H, W, 2) & ~pos
    valid &= ~ign
    valid |= pos
    m = g[idx]                                                                       # (H,W,7) matched GT per location
    a = anch                                                                         # (H,W,2,7)
    diag = math.sqrt(ANCH_L ** 2 + ANCH_W ** 2)
    za_c = a[..., 2] + ANCH_H / 2
    reg[..., 0] = (m[..., None, 0] - a[..., 0]) / diag
    reg[..., 1] = (m[..., None, 1] - a[..., 1]) / diag
    reg[..., 2] = (m[..., None, 2] - za_c) / ANCH_H
    reg[..., 3] = torch.log(m[..., None, 3] / ANCH_L)
    reg[..., 4] = torch.log(m[..., None, 4] / ANCH_W)
    reg[..., 5] = torch.log(m[..., None, 5] / ANCH_H)
    reg[..., 6] = wrap_pi(2 * (m[..., None, 6] - a[..., 6])) / 2                      # residual in [-pi/2, pi/2)
    dirl[...] = (torch.remainder(m[..., None, 6] + math.pi / 2, 2 * math.pi) >= math.pi).long()
    return pos, valid, reg, dirl


def focal(logit, target, valid, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
    return (w * ce * valid).sum()


def loss_fn(outs, cars, dcs, device):
    c, r, d = car_outputs(*outs)
    B, H, W = c.shape[:3]
    anch = car_anchor_grid(H, W, device)
    L_cls = L_reg = L_dir = 0.0
    npos = 0
    for b in range(B):
        pos, valid, reg_t, dirl = assign(cars[b], dcs[b], anch)
        L_cls = L_cls + focal(c[b], pos.float(), valid.float())
        if pos.any():
            rp, rt = r[b][pos], reg_t[pos]
            l1 = F.smooth_l1_loss(rp[:, :6], rt[:, :6], beta=1 / 9, reduction="sum")
            sp, st = torch.sin(rp[:, 6]) * torch.cos(rt[:, 6]), torch.cos(rp[:, 6]) * torch.sin(rt[:, 6])
            l1 = l1 + F.smooth_l1_loss(sp, st, beta=1 / 9, reduction="sum")
            L_reg = L_reg + l1
            L_dir = L_dir + F.cross_entropy(d[b][pos], dirl[pos], reduction="sum")
        npos += int(pos.sum())
    n = max(npos, 1)
    return (L_cls + 2.0 * L_reg + 0.2 * L_dir) / n, {"cls": float(L_cls) / n, "reg": float(L_reg) / n,
                                                   "dir": float(L_dir) / n, "npos": npos}


# ---------------------------------------------------------------------------------------------- train
def train(fold, epochs, batch, limit=None, lr=2e-4, workers=6, per_epoch=None):
    cfg = load_config()
    prepd = load_prep(cfg)
    from .truckscenes_data import build_ts_index
    frames = [f for f in build_ts_index() if f.has_data() and prepd["fold"][f.scene] == fold]
    if limit:
        frames = frames[:: max(1, len(frames) // limit)][:limit]
    od = out_dir(cfg)
    ckpt = od / f"fold{fold}_last.pt"
    dev = torch.device("cuda")
    net = load_net(device=dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    per_epoch = min(per_epoch or len(frames), len(frames))
    steps_per_epoch = per_epoch // batch
    torch.backends.cudnn.benchmark = True
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    start_epoch = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start_epoch = s["epoch"]
        print(f"resumed fold {fold} from epoch {start_epoch}", flush=True)
    log = open(od / f"fold{fold}_train.log", "a", buffering=1)
    ds = DetData(frames, prepd, train=True)
    for ep in range(start_epoch, epochs):
        sampler = torch.utils.data.RandomSampler(ds, replacement=False, num_samples=per_epoch,
                                                 generator=torch.Generator().manual_seed(1000 * fold + ep))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, collate_fn=collate,
                                         drop_last=True, persistent_workers=False)
        t0, run = time.time(), {"loss": 0.0, "n": 0, "wait": 0.0, "step": 0.0}
        t_prev = time.time()
        for it, (feats, num, coors, cars, dcs) in enumerate(dl):
            t_got = time.time()
            run["wait"] += t_got - t_prev
            feats, num, coors = feats.to(dev), num.to(dev), coors.to(dev)
            outs = forward_batch(net, feats, num, coors, len(cars))
            loss, parts = loss_fn(outs, cars, dcs, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            run["loss"] += float(loss)
            run["n"] += 1
            torch.cuda.synchronize()
            t_prev = time.time()
            run["step"] += t_prev - t_got
            if (it + 1) % 50 == 0:
                msg = (f"fold{fold} ep{ep} it{it + 1}/{steps_per_epoch} loss={run['loss'] / run['n']:.4f} "
                       f"cls={parts['cls']:.3f} reg={parts['reg']:.3f} dir={parts['dir']:.3f} npos={parts['npos']} "
                       f"wait={run['wait'] / run['n']:.2f}s step={run['step'] / run['n']:.2f}s/it peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB lr={sched.get_last_lr()[0]:.2e}")
                print(msg, flush=True)
                log.write(msg + "\n")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1},
                   ckpt)
        log.write(f"epoch {ep + 1} done in {time.time() - t0:.0f}s mean loss {run['loss'] / max(run['n'], 1):.4f}\n")
    torch.save(net.state_dict(), od / f"fold{fold}_final.pt")
    print("training finished", flush=True)


# ---------------------------------------------------------------------------------------------- inference
class InDomainCarDetector:
    def __init__(self, weights, ground_z, device="cuda", score_thr=0.05, nms_iou=0.01, max_det=100, nms_pre=1000):
        self.dev = torch.device(device)
        self.net = load_net(state=torch.load(weights, map_location="cpu"), device=self.dev).eval()
        self.ground_z, self.score_thr, self.nms_iou, self.max_det, self.nms_pre = ground_z, score_thr, nms_iou, max_det, nms_pre
        self.z_shift = -KITTI_HEIGHT - ground_z

    @staticmethod
    def center_prefilter(boxes, scores, dist=1.2):
        """Greedy suppression of boxes whose centre lies within `dist` m of a higher-scoring kept box. Cars are >= 3.9 x 1.6 m, so
        boxes this close always overlap and the IoU=0.01 rotated NMS would remove them too; this just spares it ~1000 polygons."""
        order = np.argsort(-scores)
        xy = boxes[order, :2]
        D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        gone = np.zeros(len(order), bool)
        keep = []
        for i in range(len(order)):
            if gone[i]:
                continue
            keep.append(order[i])
            gone |= D[i] < dist
        return np.array(keep, int)

    def prepare(self, points):
        return voxelize(preprocess(points, self.ground_z))

    @torch.no_grad()
    def from_voxels(self, feats, num, coors):
        empty = {"boxes": np.zeros((0, 7)), "scores": np.zeros(0), "labels": np.zeros(0, int)}
        if len(feats) == 0:
            return empty
        t = lambda a: (a if torch.is_tensor(a) else torch.from_numpy(a)).to(self.dev)
        outs = forward_batch(self.net, t(feats), t(num), t(coors), 1)
        c, r, d = car_outputs(*outs)
        H, W = c.shape[1:3]
        anchors = make_anchors(H, W)[:, :, 2].reshape(-1, 7)
        scores = torch.sigmoid(c[0]).reshape(-1).cpu().numpy()
        deltas = r[0].reshape(-1, 7).cpu().numpy()
        dir_lbl = d[0].reshape(-1, 2).argmax(-1).cpu().numpy()
        cand = np.nonzero(scores > self.score_thr)[0]
        cand = cand[np.argsort(-scores[cand])[: self.nms_pre]]
        if len(cand) == 0:
            return empty
        boxes = decode(anchors[cand], deltas[cand])
        rot = limit_period(boxes[:, 6] - DIR_OFFSET, DIR_LIMIT_OFFSET, np.pi)
        boxes[:, 6] = rot + DIR_OFFSET + np.pi * dir_lbl[cand]
        sc = scores[cand]
        pre = self.center_prefilter(boxes, sc)
        boxes, sc = boxes[pre], sc[pre]
        keep = rotated_nms_bev(boxes, sc, self.nms_iou, self.max_det)
        b, s = boxes[keep], sc[keep]
        b[:, 2] = b[:, 2] + b[:, 5] / 2 - self.z_shift          # project convention: centre z in the sensor frame
        return {"boxes": b, "scores": s, "labels": np.zeros(len(s), int)}

    def __call__(self, points):
        return self.from_voxels(*self.prepare(points))


def evaluate(fold, stride=10, device="cpu", threads=4):
    """Car AP in the ROI on the frames of detector fold `fold` (held out from detector 1-fold's training) for the KITTI
    checkpoint and for the detector trained on the OTHER fold."""
    torch.set_num_threads(threads)
    from .detectors.pointpillars import PointPillarsDetector
    from .geometry import in_roi
    from .metrics import average_precision_center
    from .truckscenes_data import build_ts_index, read_lidar
    cfg = load_config()
    prepd = load_prep(cfg)
    ground_z = prepd["ground_z"]
    frames = [f for f in build_ts_index() if f.has_data() and prepd["fold"][f.scene] == fold][::stride]
    ind = InDomainCarDetector(out_dir(cfg) / f"fold{1 - fold}_final.pt", ground_z, device)
    ck = resolve(cfg, "checkpoints") / cfg["detectors"]["lidar"]["checkpoint"]
    kit = PointPillarsDetector(ck, device=device, z_shift=-KITTI_HEIGHT - ground_z, intensity_scale=1 / 255.0,
                               score_thr=0.05, nms_iou=0.01, max_det=100, legacy=True)
    roi = cfg["roi"]
    res = {"fold_evaluated": fold, "n_frames": len(frames)}
    preds = {"kitti": [], "in_domain": []}
    gts = []
    by_cond = {}
    for f in frames:
        pts = read_lidar(f)
        g = prepd["gt"][f.token]["car"]
        gts.append(g[in_roi(g, roi)] if len(g) else g)
        o = kit(pts)
        car = (o["labels"] == 2) & in_roi(o["boxes"], roi)
        preds["kitti"].append((o["boxes"][car], o["scores"][car]))
        o = ind(pts)
        m = in_roi(o["boxes"], roi) if len(o["boxes"]) else np.zeros(0, bool)
        preds["in_domain"].append((o["boxes"][m], o["scores"][m]))
        by_cond.setdefault(f.condition, []).append(len(gts) - 1)
    for name, p in preds.items():
        res[name] = {str(t): average_precision_center(p, gts, thr=t) for t in (0.5, 1.0, 2.0, 4.0)}
        res[name]["mean"] = float(np.mean(list(res[name].values())))
        res[name + "_by_condition"] = {
            c: float(np.mean([average_precision_center([p[i] for i in idx], [gts[i] for i in idx], thr=t)
                              for t in (0.5, 1.0, 2.0, 4.0)])) for c, idx in by_cond.items()}
    (out_dir(cfg) / f"eval_fold{fold}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-epoch", type=int, default=None, help="random frames per epoch (default: all of the fold)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "train":
        train(a.fold, a.epochs, a.batch, a.limit, lr=a.lr, workers=a.workers, per_epoch=a.per_epoch)
    else:
        evaluate(a.fold, a.stride, device=a.device, threads=a.threads)
