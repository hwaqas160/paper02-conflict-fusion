"""Boreas-objects-v1 frame index and raw-sensor loaders.

Boreas sensors are asynchronous. A *frame* here is anchored on a labelled LiDAR scan and pairs it
with the camera image and radar scan closest in time. Every cross-sensor transform uses the
per-sensor Applanix poses, T_a_b = inv(T_enu_a) @ T_enu_b, so ego-motion between the three
timestamps is compensated exactly (Burnett et al., IJRR 2023, Sec. 7.1).
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pyboreas.utils.radar import load_radar
from pyboreas.utils.utils import get_transform, load_lidar

from .common import CLASS_TO_ID
from .geometry import relative_tf

SEQ = "boreas-objects-v1"
IMG_W, IMG_H = 2448, 2048


@dataclass
class Frame:
    dataset: str
    scene: int
    lidar_ts: int
    cam_ts: int
    radar_ts: int
    radar_prev_ts: int
    T_enu_lidar: np.ndarray
    T_enu_cam: np.ndarray
    T_enu_radar: np.ndarray
    T_enu_radar_prev: np.ndarray
    root: Path
    split: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def frame_id(self):
        return str(self.lidar_ts)

    def T_cam_lidar(self):
        return relative_tf(self.T_enu_cam, self.T_enu_lidar)

    def T_lidar_radar(self, prev=False):
        return relative_tf(self.T_enu_lidar, self.T_enu_radar_prev if prev else self.T_enu_radar)

    def lidar_path(self):
        return self.root / "lidar" / f"{self.lidar_ts}.bin"

    def cam_path(self):
        return self.root / "camera" / f"{self.cam_ts}.png"

    def radar_path(self, prev=False):
        return self.root / "radar" / f"{self.radar_prev_ts if prev else self.radar_ts}.png"

    def has_data(self):
        paths = [self.lidar_path(), self.cam_path(), self.radar_path(), self.radar_path(prev=True)]
        return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _read_poses(csv_path):
    times, poses = [], []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            gt = [float(x) for x in parts]
            times.append(int(parts[0]))
            poses.append(get_transform(gt))
    order = np.argsort(times)
    return np.asarray(times, dtype=np.int64)[order], np.asarray(poses)[order]


def _nearest(times, q):
    i = int(np.clip(np.searchsorted(times, q), 1, len(times) - 1))
    return i - 1 if q - times[i - 1] <= times[i] - q else i


def load_calib(root):
    root = Path(root)
    P = np.loadtxt(root / "calib" / "P_camera.txt")
    T_camera_lidar = np.loadtxt(root / "calib" / "T_camera_lidar.txt")
    T_radar_lidar = np.loadtxt(root / "calib" / "T_radar_lidar.txt")
    return {"P": P, "T_camera_lidar": T_camera_lidar, "T_radar_lidar": T_radar_lidar,
            "width": IMG_W, "height": IMG_H}


def build_index(boreas_root, require_data=True):
    """All labelled Boreas-objects-v1 frames, each assigned to its scene."""
    root = Path(boreas_root) / SEQ
    lt, lp = _read_poses(root / "applanix" / "lidar_poses.csv")
    ct, cp = _read_poses(root / "applanix" / "camera_poses.csv")
    rt, rp = _read_poses(root / "applanix" / "radar_poses.csv")
    scenes = np.loadtxt(root / "applanix" / "seqtimes.txt")  # seconds, [start, end]
    label_ts = sorted(int(p.stem) for p in (root / "labels_detection").glob("*.txt"))
    lidar_lookup = {t: i for i, t in enumerate(lt)}
    frames = []
    for ts in label_ts:
        if ts not in lidar_lookup:
            continue
        t_sec = ts * 1e-6
        scene = int(np.nonzero((scenes[:, 0] - 1e-3 <= t_sec) & (t_sec <= scenes[:, 1] + 1e-3))[0][0]) \
            if np.any((scenes[:, 0] - 1e-3 <= t_sec) & (t_sec <= scenes[:, 1] + 1e-3)) else -1
        ci = _nearest(ct, ts)
        ri = _nearest(rt, ts)
        rpi = max(ri - 1, 0)
        fr = Frame("boreas", scene, ts, int(ct[ci]), int(rt[ri]), int(rt[rpi]),
                   lp[lidar_lookup[ts]], cp[ci], rp[ri], rp[rpi], root)
        if require_data and not fr.has_data():
            continue
        frames.append(fr)
    return frames


def assign_splits(frames, fractions, seed=0, stride=1):
    """Scene-level split. Deterministic given the seed and the set of scenes."""
    scenes = sorted({f.scene for f in frames})
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(scenes))
    n = len(perm)
    n_train = int(round(fractions["train"] * n))
    n_cal = int(round(fractions["cal"] * n))
    split_of = {}
    for i, s in enumerate(perm):
        split_of[s] = "train" if i < n_train else ("cal" if i < n_train + n_cal else "test")
    out, counters = [], {}
    for f in frames:
        f.split = split_of[f.scene]
        k = counters.get(f.scene, 0)
        counters[f.scene] = k + 1
        if k % stride == 0:
            out.append(f)
    return out


# --------------------------------------------------------------------------------------------
# Raw loaders

def read_lidar(frame):
    """(N,6) float64 [x, y, z, intensity, ring, t] in the LiDAR frame."""
    return load_lidar(str(frame.lidar_path()))


def read_camera(frame):
    img = cv2.imread(str(frame.cam_path()), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_radar(frame, prev=False):
    """Returns dict(azimuths (A,), polar (A,R) float32 in [0,1], resolution m/bin, timestamps)."""
    ts, az, _, polar, res = load_radar(str(frame.radar_path(prev)))
    return {"azimuths": az.astype(np.float64).ravel(), "polar": polar.astype(np.float32), "resolution": float(res),
            "timestamps": ts.squeeze()}


def read_labels(frame, folder="labels_detection"):
    """GT boxes (N,7) [x,y,z,l,w,h,yaw], class ids (N,), num_points (N,), uuids."""
    path = frame.root / folder / f"{frame.lidar_ts}.txt"
    boxes, cls, npts, uuids = [], [], [], []
    if path.exists():
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 9:
                    continue
                label = p[1]
                cid = CLASS_TO_ID.get(label, -1)  # Misc -> -1 (ignored / don't-care)
                boxes.append([float(p[5]), float(p[6]), float(p[7]), float(p[2]), float(p[3]), float(p[4]), float(p[8])])
                cls.append(cid)
                npts.append(0 if p[9] == "None" else int(p[9]))
                uuids.append(p[0])
    return (np.asarray(boxes, dtype=np.float64).reshape(-1, 7), np.asarray(cls, dtype=np.int64),
            np.asarray(npts, dtype=np.int64), uuids)


def read_ground_truth(frame, min_points=25):
    """Evaluation GT (labels_detection) plus don't-care boxes: tracking labels with too few points
    and 'Misc' vehicles — detections on those are neither TP nor FP."""
    boxes, cls, npts, uuids = read_labels(frame, "labels_detection")
    keep = (cls >= 0) & (npts >= min_points)
    t_boxes, t_cls, t_npts, t_uuids = read_labels(frame, "labels_tracking")
    kept = {u for u, k in zip(uuids, keep) if k}
    dc = np.array([u not in kept for u in t_uuids], dtype=bool)
    return {"boxes": boxes[keep], "cls": cls[keep], "npts": npts[keep],
            "dontcare": t_boxes[dc] if len(t_boxes) else np.zeros((0, 7))}
