"""Pure-PyTorch port of mmdetection3d v1.0.0 PointPillars (KITTI 3-class, SECFPN).

mmdet3d's CUDA ops (hard voxelisation, rotated NMS) do not build on Windows without nvcc, so the
network is re-implemented module-for-module and the *published* checkpoint is loaded unchanged:

    hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
    https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/

Voxelisation and rotated NMS are done on CPU in numpy/shapely. The port is validated against
Boreas ground truth by `python -m src.detectors.pointpillars --selfcheck`.

Domain adaptation is limited to *input* conventions, never weights: the KITTI Velodyne sits 1.73 m
above ground, Boreas' Alpha-Prime ~2 m, so points are shifted in z; intensity is rescaled to [0,1].
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import rotated_nms_bev

PC_RANGE = np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0])
VOXEL = np.array([0.16, 0.16, 4.0])
GRID = ((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL).round().astype(int)  # (432, 496, 1)
MAX_PTS, MAX_VOXELS = 32, 40000
CLASSES = ("Pedestrian", "Cyclist", "Car")  # checkpoint order
ANCHOR_RANGES = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                 [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                 [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
ANCHOR_SIZES = [[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]]
ROTATIONS = [0.0, 1.57]
DIR_OFFSET, DIR_LIMIT_OFFSET = -np.pi / 2, 0.0


def _bn2d(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class PFNLayer(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.linear = nn.Linear(cin, cout, bias=False)
        self.norm = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        return F.relu(x).max(dim=1)[0]


class PillarFeatureNet(nn.Module):
    def __init__(self, legacy=True):
        super().__init__()
        self.legacy = legacy
        self.pfn_layers = nn.ModuleList([PFNLayer(10, 64)])
        self.vx, self.vy, self.vz = VOXEL
        self.x_offset = self.vx / 2 + PC_RANGE[0]
        self.y_offset = self.vy / 2 + PC_RANGE[1]
        self.z_offset = self.vz / 2 + PC_RANGE[2]

    def forward(self, feats, num_points, coors):
        # feats (V, P, 4); coors (V, 4) = (batch, z, y, x)
        points_mean = feats[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(feats).view(-1, 1, 1)
        f_cluster = feats[:, :, :3] - points_mean
        centre = torch.stack([coors[:, 3].type_as(feats) * self.vx + self.x_offset,
                              coors[:, 2].type_as(feats) * self.vy + self.y_offset,
                              coors[:, 1].type_as(feats) * self.vz + self.z_offset], dim=1).unsqueeze(1)
        f_center = feats[:, :, :3] - centre
        # mmdet3d legacy=True writes f_center into the xyz slice of `feats` in place, so the first
        # three input channels become voxel-centre offsets too. The checkpoint was trained that way.
        base = torch.cat([f_center, feats[:, :, 3:]], dim=-1) if self.legacy else feats
        x = torch.cat([base, f_cluster, f_center], dim=-1)
        mask = torch.arange(x.shape[1], device=x.device)[None] < num_points[:, None]
        x = x * mask.unsqueeze(-1).type_as(x)
        return self.pfn_layers[0](x)


class SECOND(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(64, 64, 3), (64, 128, 5), (128, 256, 5)]
        blocks = []
        for cin, cout, n in cfg:
            layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), _bn2d(cout), nn.ReLU(inplace=True)]
            for _ in range(n):
                layers += [nn.Conv2d(cout, cout, 3, padding=1, bias=False), _bn2d(cout), nn.ReLU(inplace=True)]
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return outs


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(cin, 128, s, stride=s, bias=False), _bn2d(128), nn.ReLU(inplace=True))
            for cin, s in [(64, 1), (128, 2), (256, 4)]])

    def forward(self, xs):
        return torch.cat([d(x) for d, x in zip(self.deblocks, xs)], dim=1)


class Anchor3DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cls = nn.Conv2d(384, 18, 1)
        self.conv_reg = nn.Conv2d(384, 42, 1)
        self.conv_dir_cls = nn.Conv2d(384, 12, 1)

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)


class PointPillarsNet(nn.Module):
    def __init__(self, legacy=True):
        super().__init__()
        self.voxel_encoder = PillarFeatureNet(legacy)
        self.backbone = SECOND()
        self.neck = SECONDFPN()
        self.bbox_head = Anchor3DHead()

    def forward(self, feats, num_points, coors):
        pillars = self.voxel_encoder(feats, num_points, coors)
        nx, ny = GRID[0], GRID[1]
        canvas = pillars.new_zeros(64, ny * nx)
        canvas[:, (coors[:, 2] * nx + coors[:, 3]).long()] = pillars.t()
        canvas = canvas.view(1, 64, ny, nx)
        return self.bbox_head(self.neck(self.backbone(canvas)))


def make_anchors(feat_h, feat_w):
    """(H, W, 3 classes, 2 rotations, 7) anchors, mmdet3d AlignedAnchor3DRangeGenerator(align_corner=False)."""
    out = []
    for rng, size in zip(ANCHOR_RANGES, ANCHOR_SIZES):
        xs = np.linspace(rng[0], rng[3], feat_w + 1)
        ys = np.linspace(rng[1], rng[4], feat_h + 1)
        xs = xs[:feat_w] + (xs[1] - xs[0]) / 2
        ys = ys[:feat_h] + (ys[1] - ys[0]) / 2
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        a = np.zeros((feat_h, feat_w, len(ROTATIONS), 7))
        a[..., 0] = xx[..., None]
        a[..., 1] = yy[..., None]
        a[..., 2] = rng[2]
        a[..., 3:6] = size
        a[..., 6] = np.array(ROTATIONS)[None, None]
        out.append(a)
    return np.stack(out, axis=2)


def decode(anchors, deltas):
    """DeltaXYZWLHRBBoxCoder.decode; z is the bottom face on input and output."""
    xa, ya, za, wa, la, ha, ra = [anchors[:, i] for i in range(7)]
    xt, yt, zt, wt, lt, ht, rt = [deltas[:, i] for i in range(7)]
    za = za + ha / 2
    diag = np.sqrt(la ** 2 + wa ** 2)
    xg, yg = xt * diag + xa, yt * diag + ya
    zg = zt * ha + za
    wg, lg, hg = np.exp(wt) * wa, np.exp(lt) * la, np.exp(ht) * ha
    return np.stack([xg, yg, zg - hg / 2, wg, lg, hg, rt + ra], axis=1)


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def voxelize(points):
    """Hard voxelisation, mmcv semantics: first MAX_PTS points per pillar in input order,
    pillars ordered by first appearance, at most MAX_VOXELS pillars."""
    keep = np.all((points[:, :3] >= PC_RANGE[:3]) & (points[:, :3] < PC_RANGE[3:]), axis=1)
    pts = points[keep]
    idx = np.floor((pts[:, :3] - PC_RANGE[:3]) / VOXEL).astype(np.int64)
    lin = idx[:, 1] * GRID[0] + idx[:, 0]
    uniq, first, inverse = np.unique(lin, return_index=True, return_inverse=True)
    order_of_voxel = np.argsort(np.argsort(first))  # rank by first appearance
    vid = order_of_voxel[inverse]
    n_vox = min(len(uniq), MAX_VOXELS)
    sel = vid < n_vox
    pts, vid, idx = pts[sel], vid[sel], idx[sel]
    order = np.argsort(vid, kind="stable")
    vid_sorted = vid[order]
    starts = np.searchsorted(vid_sorted, np.arange(n_vox))
    rank = np.arange(len(order)) - starts[vid_sorted]
    ok = rank < MAX_PTS
    feats = np.zeros((n_vox, MAX_PTS, 4), dtype=np.float32)
    feats[vid_sorted[ok], rank[ok]] = pts[order][ok, :4]
    num = np.minimum(np.bincount(vid, minlength=n_vox), MAX_PTS)
    coors = np.zeros((n_vox, 4), dtype=np.int64)
    coors[vid_sorted, 1:] = np.stack([idx[order][:, 2], idx[order][:, 1], idx[order][:, 0]], axis=1)
    return feats, num, coors


class PointPillarsDetector:
    """Frozen LiDAR detector. Output boxes use this project's convention (centre z, l along yaw)."""

    def __init__(self, checkpoint, device="cpu", z_shift=0.0, intensity_scale=1.0, score_thr=0.05,
                 nms_iou=0.01, max_det=100, legacy=True, nms_pre=1000):
        self.device = torch.device(device)
        self.net = PointPillarsNet(legacy=legacy)
        sd = torch.load(checkpoint, map_location="cpu")["state_dict"]
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.net.eval().to(self.device)
        self.z_shift, self.intensity_scale = z_shift, intensity_scale
        self.score_thr, self.nms_iou, self.max_det, self.nms_pre = score_thr, nms_iou, max_det, nms_pre
        self._anchors = None

    @torch.no_grad()
    def __call__(self, points):
        """points (N, >=4) Boreas LiDAR frame. Returns dict(boxes (M,7), scores (M,), labels (M,) in CLASSES)."""
        p = np.asarray(points[:, :4], dtype=np.float32).copy()
        p[:, 2] += self.z_shift
        p[:, 3] = np.clip(p[:, 3] * self.intensity_scale, 0, 1)
        feats, num, coors = voxelize(p)
        if len(feats) == 0:
            return {"boxes": np.zeros((0, 7)), "scores": np.zeros(0), "labels": np.zeros(0, int)}
        t = lambda a: torch.from_numpy(a).to(self.device)
        cls, reg, dirc = self.net(t(feats), t(num), t(coors))
        h, w = cls.shape[-2:]
        if self._anchors is None or self._anchors.shape[:2] != (h, w):
            self._anchors = make_anchors(h, w)
        anchors = self._anchors.reshape(-1, 7)
        scores = torch.sigmoid(cls[0].permute(1, 2, 0).reshape(-1, 3)).cpu().numpy()
        deltas = reg[0].permute(1, 2, 0).reshape(-1, 7).cpu().numpy()
        dir_lbl = dirc[0].permute(1, 2, 0).reshape(-1, 2).argmax(-1).cpu().numpy()
        best = scores.max(1)
        cand = np.nonzero(best > self.score_thr)[0]
        cand = cand[np.argsort(-best[cand])[: self.nms_pre]]
        boxes = decode(anchors[cand], deltas[cand])
        rot = limit_period(boxes[:, 6] - DIR_OFFSET, DIR_LIMIT_OFFSET, np.pi)
        boxes[:, 6] = rot + DIR_OFFSET + np.pi * dir_lbl[cand]
        out_b, out_s, out_l = [], [], []
        for c in range(3):
            sc = scores[cand, c]
            m = sc > self.score_thr
            if not m.any():
                continue
            b = boxes[m]
            keep = rotated_nms_bev(b[:, [0, 1, 2, 3, 4, 5, 6]], sc[m], self.nms_iou, self.max_det)
            out_b.append(b[keep])
            out_s.append(sc[m][keep])
            out_l.append(np.full(len(keep), c))
        if not out_b:
            return {"boxes": np.zeros((0, 7)), "scores": np.zeros(0), "labels": np.zeros(0, int)}
        b, s, l = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)
        order = np.argsort(-s)[: self.max_det]
        b, s, l = b[order], s[order], l[order]
        # to project convention: centre z, undo the height shift
        b[:, 2] = b[:, 2] + b[:, 5] / 2 - self.z_shift
        return {"boxes": b, "scores": s, "labels": l}


def estimate_ground_z(frames_gt_boxes):
    bottoms = np.concatenate([b[:, 2] - b[:, 5] / 2 for b in frames_gt_boxes if len(b)])
    near = bottoms  # labels are relative to the LiDAR, so the median bottom is the road plane
    return float(np.median(near))


def _selfcheck():
    """Load real Boreas frames, run both voxel-centre variants and yaw conventions, report BEV AP."""
    from ..boreas_data import build_index, read_lidar, read_ground_truth
    from ..common import load_config, resolve
    from ..metrics import average_precision_center
    cfg = load_config()
    frames = [f for f in build_index(resolve(cfg, "boreas_root"), require_data=False) if f.lidar_path().exists()]
    frames = frames[:: max(1, len(frames) // 40)][:40]
    gts = [read_ground_truth(f) for f in frames]
    ground = estimate_ground_z([g["boxes"] for g in gts])
    z_shift = -1.73 - ground
    print(f"frames={len(frames)} ground_z={ground:.3f} -> z_shift={z_shift:+.3f}")
    pts0 = read_lidar(frames[0])
    scale = 1 / 255.0 if pts0[:, 3].max() > 1.5 else 1.0
    print(f"intensity max={pts0[:, 3].max():.1f} -> scale={scale}")
    ck = resolve(cfg, "checkpoints") / cfg["detectors"]["lidar"]["checkpoint"]
    from ..geometry import in_roi
    import time
    roi = cfg["roi"]
    clouds = [read_lidar(f) for f in frames]
    for legacy, zs in [(True, z_shift), (False, z_shift), (True, z_shift - 0.3), (True, z_shift + 0.3)]:
        det = PointPillarsDetector(ck, z_shift=zs, intensity_scale=scale, legacy=legacy)
        preds, gtl, t0 = [], [], time.time()
        for pts, g in zip(clouds, gts):
            out = det(pts)
            car = (out["labels"] == 2) & in_roi(out["boxes"], roi)
            preds.append((out["boxes"][car], out["scores"][car]))
            gtl.append(g["boxes"][(g["cls"] == 0) & in_roi(g["boxes"], roi)])
        dt = (time.time() - t0) / len(clouds)
        ap = {t: average_precision_center(preds, gtl, thr=t) for t in (0.5, 1.0, 2.0, 4.0)}
        print(f"legacy={legacy} z_shift={zs:+.2f}: Car AP in ROI " +
              " ".join(f"@{t}m={v:.3f}" for t, v in ap.items()) + f"  ({dt:.2f}s/frame)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    if ap.parse_args().selfcheck:
        _selfcheck()
