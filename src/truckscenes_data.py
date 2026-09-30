"""MAN TruckScenes v1.2 frame index, loaders and ground truth, in the conventions of boreas_data.py.

Reference frame per keyframe: the truck's ego frame at the LIDAR_LEFT keyframe timestamp, shifted forward by
REF_X so the origin sits at the front of the cab (where a front-mounted LiDAR would be); x forward, y left, z up.
Every sensor is motion-compensated into it through its own timestamped ego pose (nuScenes convention:
sensor -> ego(t_sensor) -> global -> ego(t_ref) -> ref).

No LiDAR ring index is stored, so a pseudo-ring is derived from each point's elevation angle in its own
sensor frame (0.3 deg bins, offset per sensor) -- the physical definition of a beam channel -- which keeps the
l_ring_ratio probe meaningful and comparable to Boreas (see notes/s4_fov_bug.md on the degenerate CADC ring).
"""
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .common import CLASS_TO_ID, PROJECT_ROOT
from .fetch_truckscenes import CAMERA, LIDARS, META, RADARS, ROOT
from .pcd import read_pcd

REF_X = 5.0
RING_BIN_DEG = 0.3
CLASS_MAP = {"vehicle.car": "Car", "human.pedestrian.adult": "Pedestrian", "human.pedestrian.child": "Pedestrian",
             "human.pedestrian.construction_worker": "Pedestrian", "human.pedestrian.police_officer": "Pedestrian",
             "vehicle.bicycle": "Cyclist", "vehicle.motorcycle": "Cyclist"}
MIN_LIDAR_PTS = 5  # Registration 2 (fixed before data)


def tf(translation, quat_wxyz):
    T = np.eye(4)
    w, x, y, z = quat_wxyz
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = translation
    return T


def condition_of(description):
    d = dict(t.split(".", 1) for t in description.split(";") if "." in t)
    w, l = d.get("weather", ""), d.get("lighting", "")
    if w in ("rain", "snow", "fog"):
        return w, 1, d
    if l in ("dark", "twilight"):
        return "night", 1, d
    return "clear", 0, d


@dataclass
class TSFrame:
    token: str
    scene: str
    condition: str
    severity: int
    tags: dict
    t_ref: int
    T_global_ego_ref: np.ndarray
    sensors: dict = field(default_factory=dict)  # channel -> dict(file, T_ego_sensor, T_global_ego, K, prev_*)

    @property
    def frame_id(self):
        return self.token

    def T_ref_global(self):
        T_ref_ego = np.eye(4)
        T_ref_ego[0, 3] = -REF_X
        return T_ref_ego @ np.linalg.inv(self.T_global_ego_ref)

    def T_ref_sensor(self, ch, prev=False):
        s = self.sensors[ch]
        Tg = s["T_global_ego_prev"] if prev else s["T_global_ego"]
        return self.T_ref_global() @ Tg @ s["T_ego_sensor"]

    def has_data(self):
        return all((ROOT / s["file"]).exists() for s in self.sensors.values())


def build_ts_index(cache=ROOT / "frames.pkl"):
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    J = lambda n: json.load(open(META / f"{n}.json"))
    sensors = {s["token"]: s["channel"] for s in J("sensor")}
    calib = {c["token"]: c for c in J("calibrated_sensor")}
    poses = {p["token"]: p for p in J("ego_pose")}
    scenes = {s["token"]: s for s in J("scene")}
    samples = J("sample")
    sd = J("sample_data")
    by_tok = {r["token"]: r for r in sd}
    key = {}
    for r in sd:
        if r["is_key_frame"]:
            ch = sensors[calib[r["calibrated_sensor_token"]]["sensor_token"]]
            if ch in LIDARS + [CAMERA] + RADARS:
                key.setdefault(r["sample_token"], {})[ch] = r
    pose_T = lambda tok: tf(poses[tok]["translation"], poses[tok]["rotation"])
    frames = []
    for s in samples:
        chans = key.get(s["token"], {})
        if len(chans) < len(LIDARS) + 1 + len(RADARS):
            continue
        sc = scenes[s["scene_token"]]
        cond, sev, tags = condition_of(sc["description"])
        ref = chans["LIDAR_LEFT"]
        fr = TSFrame(s["token"], sc["name"], cond, sev, tags, ref["timestamp"], pose_T(ref["ego_pose_token"]))
        for ch, r in chans.items():
            c = calib[r["calibrated_sensor_token"]]
            e = {"file": r["filename"], "T_ego_sensor": tf(c["translation"], c["rotation"]),
                 "T_global_ego": pose_T(r["ego_pose_token"]), "timestamp": r["timestamp"]}
            if c.get("camera_intrinsic"):
                e["K"] = np.array(c["camera_intrinsic"], float)
            if ch in RADARS and r["prev"]:
                p = by_tok[r["prev"]]
                e["prev_file"] = p["filename"]
                e["T_global_ego_prev"] = pose_T(p["ego_pose_token"])
            fr.sensors[ch] = e
        frames.append(fr)
    with open(cache, "wb") as f:
        pickle.dump(frames, f)
    return frames


def _to_ref(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def read_lidar(fr):
    """Merged LIDAR_TOP_FRONT/LEFT/RIGHT in the ref frame -> (N, 6) [x, y, z, intensity 0-255, pseudo_ring, 0]."""
    out = []
    for k, ch in enumerate(LIDARS):
        d = read_pcd(ROOT / fr.sensors[ch]["file"])
        xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
        ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.0)
        xyz = xyz[ok]
        elev = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
        ring = np.round(elev / RING_BIN_DEG) + 1000.0 * k
        inten = d["intensity"][ok].astype(np.float64)
        inten = inten * 255.0 if inten.max(initial=0) <= 1.5 else inten
        out.append(np.column_stack([_to_ref(fr.T_ref_sensor(ch), xyz), inten, ring, np.zeros(len(xyz))]))
    pts = np.concatenate(out)
    # forward crop: nothing outside x in [-10, 80] m, |y| <= 50 m can reach the ROI or its probes (speed only)
    keep = (pts[:, 0] > -10) & (pts[:, 0] < 80) & (np.abs(pts[:, 1]) < 50)
    return pts[keep]


def n_pseudo_rings(points):
    return max(len(np.unique(points[:, 4])), 1)


def read_camera(fr):
    img = cv2.imread(str(ROOT / fr.sensors[CAMERA]["file"]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def camera_geometry(fr):
    """T_cam_ref (4x4) and P (4x4 with K) for the ref frame."""
    T_ref_cam = fr.T_ref_sensor(CAMERA)
    P = np.eye(4)
    P[:3, :3] = fr.sensors[CAMERA]["K"]
    return np.linalg.inv(T_ref_cam), P


def read_radar(fr, prev=False):
    """Both front 4D radars in the ref frame -> (N, 5) [x, y, z, v_radial (m/s, ego-compensated), rcs]."""
    out = []
    for ch in RADARS:
        s = fr.sensors[ch]
        f = s.get("prev_file") if prev else s["file"]
        if f is None or not (ROOT / f).exists():
            continue
        d = read_pcd(ROOT / f)
        xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
        v = np.stack([d["vrel_x"], d["vrel_y"], d["vrel_z"]], 1).astype(np.float64)
        u = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-6)
        vr = (v * u).sum(1)  # radial (Doppler) component in the sensor frame
        out.append(np.column_stack([_to_ref(fr.T_ref_sensor(ch, prev=prev), xyz), vr, d["rcs"]]))
    return np.concatenate(out) if out else np.zeros((0, 5))


def read_ground_truth(fr, ann_by_sample):
    """GT in the ref frame -> dict(boxes (N,7) [x,y,z,l,w,h,yaw], cls, npts, dontcare (M,7))."""
    T = fr.T_ref_global()
    yaw_ref = np.arctan2(T[1, 0], T[0, 0])
    boxes, cls, npts, dc = [], [], [], []
    for a in ann_by_sample.get(fr.token, []):
        c = _to_ref(T, np.asarray(a["translation"], float)[None])[0]
        w, l, h = a["size"]
        qw, qx, qy, qz = a["rotation"]
        yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("zyx")[0] + yaw_ref
        box = [c[0], c[1], c[2], l, w, h, np.arctan2(np.sin(yaw), np.cos(yaw))]
        name = CLASS_MAP.get(a["category"])
        if name is None or a["num_lidar_pts"] < MIN_LIDAR_PTS:
            dc.append(box)
            continue
        boxes.append(box)
        cls.append(CLASS_TO_ID[name])
        npts.append(a["num_lidar_pts"])
    return {"boxes": np.asarray(boxes, float).reshape(-1, 7), "cls": np.asarray(cls, int),
            "npts": np.asarray(npts, int), "dontcare": np.asarray(dc, float).reshape(-1, 7)}


def annotations_by_sample():
    cats = {c["token"]: c["name"] for c in json.load(open(META / "category.json"))}
    inst = {i["token"]: cats[i["category_token"]] for i in json.load(open(META / "instance.json"))}
    out = {}
    for a in json.load(open(META / "sample_annotation.json")):
        a["category"] = inst[a["instance_token"]]
        out.setdefault(a["sample_token"], []).append(a)
    return out
