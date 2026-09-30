# CADC — access and loader verification (2026-09-21)

**Access.** Public HTTP server, no registration: `http://wiselab.uwaterloo.ca/cadcd_data/` (3 dates, 75 drives).
`src/fetch_cadc.py` range-reads only `labeled/image_00` and `labeled/lidar_points` out of each remote `labeled.zip`
(~120 MB kept per drive instead of ~0.7–1.4 GB), plus `3d_ann.json` and per-date `calib.zip`.

The three conventions `src/cadc_cache.py` listed as UNVERIFIED, checked on real data:

| assumption | check | result |
|---|---|---|
| cuboid `dimensions.y` = length, `.x` = width | LiDAR points inside the box vs annotated `points_count`, 178 cars | l = d.y: median ratio **1.00**; swapped: 0.54 → **confirmed** |
| extrinsic key `T_LIDAR_CAM00` | real file | key is **`T_LIDAR_00CAMERA`** → loader would have raised KeyError. **Fixed.** Direction: its optical-axis column is LiDAR +x, so it maps camera → LiDAR; `T_cam_lidar = inv(...)` is right |
| `labeled/image_00` already undistorted | project GT car boxes with pinhole K; best IoU with Faster R-CNN boxes, raw vs `cv2.undistort`, 188 cars in 9 drives | raw median IoU **0.728** (0.835 > 0.5) vs undistorted 0.689 (0.713); at the image edges (|u − c| > 350 px) raw 0.749 vs undistorted **0.463** → images are **already rectified**; no undistortion applied |

**Also fixed (not in the original list):**
* LiDAR intensity is in [0, 1] on CADC (max 0.92 = 235/255) but the frozen PointPillars input convention and every
  intensity probe were built on Boreas' 0–255. The loader now scales ×255 when max ≤ 1.5. Without this the detector
  would have seen intensities ~255× too small.
* `Pedestrian_With_Object` → Pedestrian. `Truck`, `Bus`, `Traffic_Guidance_Objects`, etc. are don't-care.

**Role.** External cohort, touched once: Boreas-trained heads, Boreas τ, no recalibration; radar treated as absent
(`NoRadar` wrapper in Exp 5). Exp 5 also measures AP-level and decision-flip harm of the camera on real snow.
