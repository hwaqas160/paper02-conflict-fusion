# Exp 5 — Boreas-trained heads on CADC real snow (radar removed)

7000 frames, 23145 GT cars (≥ 25 pts, ROI). No retraining, no recalibration: models and τ come from Boreas.

| arm | Car mAP (95 % CI) | 2018_03_06 | 2018_03_07 | 2019_02_27 |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.740 (0.733–0.746) | 0.828 | 0.586 | 0.728 |
| Fixed weights | 0.742 (0.735–0.748) | 0.831 | 0.588 | 0.730 |
| Confidence-only | 0.743 (0.737–0.750) | 0.832 | 0.601 | 0.731 |
| Per-frame weather classifier | 0.739 (0.733–0.746) | 0.829 | 0.581 | 0.728 |
| Per-frame oracle weather | 0.736 (0.729–0.742) | 0.823 | 0.576 | 0.724 |
| Per-frame features | 0.739 (0.733–0.746) | 0.832 | 0.576 | 0.727 |
| Per-object (ours) | 0.734 (0.728–0.741) | 0.820 | 0.592 | 0.723 |

Per-object − per-frame classifier: Δ mAP = -0.0050, one-sided p = 1.000.

## Status-quo fusion with a sensor removed (AP-level harm on real snow)

| all (camera + LiDAR) | − camera (LiDAR only) | − LiDAR (camera only) |
|---|---|---|
| 0.740 | 0.705 | 0.078 |

## Decision-flip harm on real snow (status quo, Car)

|  | pooled | camera | lidar |
|---|---|---|---|
| harm % | 3.78 (3.68–3.88) | 5.03 (4.86–5.22) | 3.08 (2.98–3.19) |
| help % | 8.36 | 1.93 | 11.91 |
