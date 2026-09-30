"""Graded, physically-motivated weather degradation of raw camera / LiDAR / radar data.

Why simulate: every labelled Boreas frame was recorded in sunny weather (Burnett et al. 2023, §8),
so real-snow Boreas sequences have no 3D ground truth. Degrading the labelled frames keeps GT
exact while producing *modality-specific* degradation — the precondition for the harm case.
Real adverse-weather data enters in two places: CADC (labelled real snow, camera+LiDAR) and
unlabelled Boreas snow sequences for a sim-to-real check of the degradation statistics.

Physics (simplified, parameters in configs/default.yaml):
  * Fog, LiDAR   — Beer–Lambert two-way attenuation exp(-2βr), β = 3.912/V (Koschmieder); returns
                   under the noise floor are lost; near-range backscatter returns
                   (after Hahner et al., ICCV 2021).
  * Snow, LiDAR  — per-beam probability of hitting a flake before the target, 1-exp(-λr), λ ∝
                   snowfall rate; flake returns replace the target at short range with low
                   intensity (after Hahner et al., CVPR 2022; cf. Boreas Fig. 3).
  * Rain, LiDAR  — weak extinction plus sparse droplet returns (after Kilic et al., LISA 2021).
  * Camera       — Koschmieder airlight model I = J·t + A(1-t), t = exp(-β d) with a densified depth
                   map from the *clean* LiDAR sweep (scene geometry, not the detector input;
                   cf. Sakaridis et al., IJCV 2018); snowflakes, rain streaks, lens droplets and
                   lens-occluding snow patches give spatially varying degradation; night = low
                   exposure with Poisson–Gaussian sensor noise.
  * Radar 77 GHz — rain specific attenuation γ = k R^α dB/km (ITU-R P.838 order of magnitude),
                   weaker for snow, negligible for fog; small precipitation clutter near range.
"""
import hashlib

import cv2
import numpy as np

from .geometry import apply_tf

INTENSITY_FLOOR = 2.0          # Boreas intensity units (0..255)
LIDAR_GAIN_HEADROOM = 30.0     # receiver detects returns ~30x weaker than the reported intensity quantum
RADAR_DB_PER_UNIT = 50.0       # assumed dynamic range of the normalised polar log-power image


def _survives(intensity, attenuation, rng):
    """A return is detected if its received power (∝ intensity x attenuation) clears the receiver
    threshold with log-normal fading. Returns (keep mask, reported attenuated intensity)."""
    margin = np.maximum(intensity, 0.5) * attenuation * LIDAR_GAIN_HEADROOM
    keep = margin * np.exp(rng.normal(0, 0.5, len(intensity))) > 1.0
    return keep, intensity * attenuation


def rng_for(frame_id, condition, severity, salt="weather"):
    h = hashlib.sha256(f"{salt}|{frame_id}|{condition}|{severity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# --------------------------------------------------------------------------------------------
# LiDAR

def degrade_lidar(points, condition, severity, wcfg, rng):
    if condition in ("clear", "night") or severity == 0:
        return points
    p = points.copy()
    r = np.linalg.norm(p[:, :3], axis=1)
    unit = p[:, :3] / np.maximum(r, 1e-6)[:, None]
    n = len(p)
    if condition == "fog":
        beta = 3.912 / wcfg["fog_visibility_m"][severity]
        keep, p[:, 3] = _survives(p[:, 3], np.exp(-2 * beta * r), rng)
        p_scatter = 1 - np.exp(-beta * 6.0)
        scatter = rng.random(n) < p_scatter
        r_s = rng.gamma(2.0, 0.8 + 6.0 / (beta * 150.0), n)
        scatter &= (r_s < r) & (r_s > 0.8)
        out = p.copy()
        out[scatter, :3] = unit[scatter] * r_s[scatter, None]
        out[scatter, 3] = rng.uniform(0, 6, scatter.sum())
        return out[keep | scatter]
    if condition == "rain":
        R = wcfg["rain_rate_mmh"][severity]
        beta = 2e-3 * R ** 0.6
        keep, p[:, 3] = _survives(p[:, 3], np.exp(-2 * beta * r), rng)
        lam = 4e-4 * R ** 0.6
        drop_hit = (rng.random(n) < 1 - np.exp(-lam * r))
        r_d = rng.uniform(0.5, 6.0, n)
        drop_hit &= r_d < r
        out = p.copy()
        out[drop_hit, :3] = unit[drop_hit] * r_d[drop_hit, None]
        out[drop_hit, 3] = rng.uniform(0, 4, drop_hit.sum())
        return out[keep | drop_hit]
    if condition == "snow":
        S = wcfg["snow_rate_mmh"][severity]
        vis = 1300.0 / (S ** 0.7)
        beta = 3.912 / vis
        lam = 4e-3 * S
        hit = rng.random(n) < 1 - np.exp(-lam * r)
        # flake range: truncated exponential on (0.5, r) — near flakes are the likely *and* bright ones
        u = rng.random(n)
        upper = np.maximum(r - 0.5, 1e-3)
        r_f = 0.5 - np.log(1 - u * (1 - np.exp(-lam * upper))) / max(lam, 1e-6)
        flake_detected = hit & (rng.random(n) < np.exp(-r_f / 15.0))
        survived, p[:, 3] = _survives(p[:, 3], np.exp(-2 * beta * r), rng)
        target_kept = survived & ~(hit & (rng.random(n) < 0.5))
        out = p.copy()
        out[flake_detected, :3] = unit[flake_detected] * r_f[flake_detected, None]
        out[flake_detected, 3] = rng.uniform(0, 12, flake_detected.sum())
        keep = np.where(flake_detected, True, target_kept)
        return out[keep]
    raise ValueError(condition)


# --------------------------------------------------------------------------------------------
# Camera

def depth_map(points_clean, T_cam_lidar, P, width, height, block=8, far=300.0):
    """Dense depth (m) by projecting the clean sweep, block-min pooling and nearest fill."""
    cam = apply_tf(T_cam_lidar, points_clean[:, :3])
    front = cam[:, 2] > 0.5
    cam = cam[front]
    uv = cam @ P[:3, :3].T
    uv = uv[:, :2] / uv[:, 2:3]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, d = uv[ok], np.linalg.norm(cam[ok], axis=1)
    gh, gw = height // block + 1, width // block + 1
    grid = np.full((gh, gw), np.inf, np.float32)
    np.minimum.at(grid, ((uv[:, 1] // block).astype(int), (uv[:, 0] // block).astype(int)), d)
    valid = np.isfinite(grid)
    if valid.any():
        # nearest-valid fill by distance transform indices
        _, (iy, ix) = _edt_indices(~valid)
        grid = grid[iy, ix]
    else:
        grid[:] = far
    # rows above the highest LiDAR return are sky / far structure
    rows_with = np.nonzero(valid.any(1))[0]
    if len(rows_with):
        grid[: rows_with.min()] = far
    grid = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 1.5)
    dm = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(cv2.GaussianBlur(dm, (0, 0), 4.0), 0.5, far)


def _edt_indices(mask):
    from scipy import ndimage
    return ndimage.distance_transform_edt(mask, return_indices=True)


def _koschmieder(img, depth, visibility, airlight):
    beta = 3.912 / visibility
    t = np.exp(-beta * depth)[..., None]
    return img * t + airlight * (1 - t)


def _snowflakes(h, w, count, rng):
    layer = np.zeros((h, w), np.float32)
    xs, ys = rng.integers(0, w, count), rng.integers(0, h, count)
    radii = np.clip(rng.gamma(1.5, 1.0, count), 0.6, 5.0)
    for x, y, rad in zip(xs, ys, radii):
        cv2.ellipse(layer, (int(x), int(y)), (max(int(rad), 1), max(int(rad * 1.8), 1)), 0, 0, 360,
                    float(rng.uniform(0.6, 1.0)), -1)
    k = max(int(3 + 2 * rng.random()), 3)
    kernel = np.zeros((k, k), np.float32)
    kernel[:, k // 2] = 1.0 / k
    return cv2.GaussianBlur(cv2.filter2D(layer, -1, kernel), (0, 0), 0.8)


def _rain_streaks(h, w, count, rng):
    layer = np.zeros((h, w), np.float32)
    angle = rng.normal(0, 0.15)
    for _ in range(count):
        x, y = rng.integers(0, w), rng.integers(0, h)
        L = rng.uniform(8, 28)
        x2, y2 = int(x + L * np.sin(angle)), int(y + L * np.cos(angle))
        cv2.line(layer, (int(x), int(y)), (x2, y2), float(rng.uniform(0.3, 0.7)), 1)
    return cv2.GaussianBlur(layer, (0, 0), 0.7)


def _lens_blobs(img, count, rng, kind):
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(count):
        cx, cy = rng.integers(0, w), rng.integers(int(0.1 * h), h)
        ax = int(rng.uniform(0.04, 0.14) * w)
        ay = int(ax * rng.uniform(0.6, 1.2))
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), ax / 3)[..., None]
        if kind == "snow":        # opaque, bright, diffuse
            blob = np.full_like(img, 0.92)
            alpha = mask * rng.uniform(0.7, 0.95)
        else:                     # water droplet: strong local defocus
            blob = cv2.GaussianBlur(img, (0, 0), 9)
            alpha = mask * 0.9
        out = out * (1 - alpha) + blob * alpha
    return out


def degrade_camera(img_rgb, condition, severity, wcfg, rng, points_clean=None, T_cam_lidar=None, P=None):
    """img_rgb uint8 (H,W,3) -> uint8. Depth-dependent effects need the clean sweep and calibration."""
    if condition == "clear" or severity == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    img = img_rgb.astype(np.float32) / 255.0
    if condition == "night":
        # ~2.5 stops under-exposed scene, auto-gain brings it back up, amplifying shot and read noise
        exposure, photons, digital_gain = 0.06, 1500.0, 1.5
        lin = np.power(img, 2.2) * exposure
        lin = rng.poisson(np.clip(lin, 0, None) * photons) / photons + rng.normal(0, 0.003, lin.shape)
        out = np.power(np.clip(lin * digital_gain, 0, 1), 1 / 2.2)
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.8)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
    depth = depth_map(points_clean, T_cam_lidar, P, w, h) if points_clean is not None else np.full((h, w), 50.0)
    if condition == "fog":
        vis = wcfg["fog_visibility_m"][severity]
        img = _koschmieder(img, depth, vis, airlight=0.82)
        img = cv2.GaussianBlur(img, (0, 0), 0.6 * severity)
    elif condition == "rain":
        R = wcfg["rain_rate_mmh"][severity]
        img = _koschmieder(img, depth, 2500.0 / R ** 0.6, airlight=0.65)
        streaks = _rain_streaks(h, w, int(60 * R ** 0.7), rng)[..., None]
        img = np.clip(img + 0.35 * streaks, 0, 1)
        img = _lens_blobs(img, int(rng.poisson(3 * severity)), rng, "rain")
        img = cv2.GaussianBlur(img, (0, 0), 0.5 * severity)
    elif condition == "snow":
        S = wcfg["snow_rate_mmh"][severity]
        img = _koschmieder(img, depth, 1300.0 / S ** 0.7, airlight=0.88)
        flakes = _snowflakes(h, w, int(900 * S), rng)[..., None]
        img = img * (1 - 0.8 * flakes) + 0.95 * 0.8 * flakes
        n_blobs = {1: 0, 2: int(rng.integers(0, 2)), 3: int(rng.integers(1, 4))}[severity]
        img = _lens_blobs(img, n_blobs, rng, "snow")
    else:
        raise ValueError(condition)
    noise = rng.normal(0, 0.006 * severity, img.shape).astype(np.float32)
    return (np.clip(img + noise, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# Radar

def degrade_radar(radar, condition, severity, wcfg, rng):
    if condition in ("clear", "night") or severity == 0:
        return radar
    polar = radar["polar"].copy()
    A, R_bins = polar.shape
    rng_m = (np.arange(R_bins) + 0.5) * radar["resolution"]
    if condition == "rain":
        rate = wcfg["rain_rate_mmh"][severity]
        gamma_db_km = 1.0 * rate ** 0.7
        clutter = 0.012 * severity
    elif condition == "snow":
        rate = wcfg["snow_rate_mmh"][severity]
        gamma_db_km = 0.6 * rate ** 0.8
        clutter = 0.010 * severity
    elif condition == "fog":
        gamma_db_km = 0.4 * severity / 3
        clutter = 0.0
    else:
        raise ValueError(condition)
    loss = (2 * gamma_db_km * rng_m / 1000.0) / RADAR_DB_PER_UNIT
    polar = polar - loss[None, :].astype(np.float32)
    if clutter > 0:
        envelope = np.exp(-rng_m / 25.0)[None, :]
        polar = polar + (clutter * envelope * np.abs(rng.normal(0, 1, polar.shape))).astype(np.float32)
    polar[:, : int(round(2.5 / radar["resolution"]))] = 0
    out = dict(radar)
    out["polar"] = np.clip(polar, 0, 1)
    return out
