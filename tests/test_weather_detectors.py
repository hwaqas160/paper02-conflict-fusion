import numpy as np
import torch

from src import weather as W
from src.detectors.pointpillars import GRID, decode, make_anchors, voxelize
from src.detectors.radar_cfar import ca_cfar

WCFG = {"fog_visibility_m": {1: 150.0, 2: 70.0, 3: 35.0}, "rain_rate_mmh": {1: 5.0, 2: 20.0, 3: 50.0},
        "snow_rate_mmh": {1: 0.5, 2: 1.5, 3: 3.0}}


def _cloud(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(2, 80, n)
    az = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-2, 1, n), rng.uniform(1, 60, n),
                     rng.integers(0, 128, n), np.zeros(n)], 1)


def test_weather_deterministic():
    pts = _cloud()
    a = W.degrade_lidar(pts, "snow", 2, WCFG, W.rng_for("f", "snow", 2, "lidar"))
    b = W.degrade_lidar(pts, "snow", 2, WCFG, W.rng_for("f", "snow", 2, "lidar"))
    assert np.array_equal(a, b)


def test_fog_severity_monotone_lidar_far_returns():
    pts = _cloud()
    far = []
    for s in (1, 2, 3):
        out = W.degrade_lidar(pts, "fog", s, WCFG, W.rng_for("f", "fog", s))
        far.append(np.sum(np.linalg.norm(out[:, :2], axis=1) > 40))
    assert far[0] > far[1] > far[2]


def test_snow_adds_near_clutter():
    pts = _cloud()
    out = W.degrade_lidar(pts, "snow", 3, WCFG, W.rng_for("f", "snow", 3))
    near = lambda p: np.mean(np.linalg.norm(p[:, :3], axis=1) < 5)
    assert near(out) > near(pts)


def test_camera_fog_reduces_contrast_and_clear_is_identity():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    assert W.degrade_camera(img, "clear", 0, WCFG, rng) is img
    fog = W.degrade_camera(img, "fog", 3, WCFG, W.rng_for("f", "fog", 3))
    assert fog.std() < img.std()


def test_radar_nearly_immune_to_fog():
    rng = np.random.default_rng(0)
    radar = {"polar": rng.random((400, 1000)).astype(np.float32), "resolution": 0.0596, "azimuths": np.linspace(0, 2 * np.pi, 400)}
    out = W.degrade_radar(radar, "fog", 3, WCFG, rng)
    assert np.abs(out["polar"][:, 100:] - radar["polar"][:, 100:]).mean() < 0.01


def test_voxelize_limits_and_coords():
    pts = np.array([[1.0, 0.0, -1.0, 0.5]] * 50 + [[10.0, 5.0, 0.0, 0.1]], np.float32)
    feats, num, coors = voxelize(pts)
    assert feats.shape[0] == 2 and num.max() == 32
    assert np.all(coors[:, 3] < GRID[0]) and np.all(coors[:, 2] < GRID[1])


def test_anchor_decode_zero_delta_is_anchor():
    a = make_anchors(248, 216).reshape(-1, 7)
    out = decode(a[:10], np.zeros((10, 7)))
    assert np.allclose(out, a[:10])


def test_cfar_finds_peak():
    polar = np.full((4, 200), 0.1, np.float32)
    polar[2, 120] = 0.9
    hits, _ = ca_cfar(polar, 2, 10, 0.1, 0.2, 200)
    assert hits[2, 120] and hits.sum() == 1
