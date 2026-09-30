# Exp 1 — Harm rate under status-quo fusion (Car, test scenes)

Status quo = fixed-weight late fusion fitted on clear; operating threshold τ = 0.405 (F1 0.777 on clear train).

Decision-flip harm rate, % of candidates containing the modality (95 % frame-bootstrap CI)

| condition | pooled | camera | lidar | radar | help % (pooled) |
|---|---|---|---|---|---|
| clear0 | 1.9 (1.8–2.0) | 3.8 (3.5–4.1) | 2.4 (2.2–2.5) | 0.7 (0.7–0.8) | 5.3 |
| fog1 | 2.1 (2.0–2.2) | 3.4 (3.1–3.6) | 2.9 (2.7–3.0) | 0.8 (0.7–0.9) | 4.8 |
| fog2 | 1.5 (1.4–1.6) | 1.9 (1.7–2.1) | 2.1 (2.0–2.3) | 0.9 (0.8–1.0) | 3.9 |
| fog3 | 0.8 (0.8–0.9) | 1.4 (1.1–1.7) | 1.2 (1.1–1.4) | 0.6 (0.5–0.7) | 2.8 |
| rain1 | 2.1 (2.0–2.3) | 3.8 (3.5–4.0) | 2.9 (2.7–3.1) | 0.8 (0.7–0.9) | 5.3 |
| rain2 | 1.8 (1.7–1.9) | 2.5 (2.3–2.7) | 2.7 (2.5–2.9) | 0.8 (0.7–0.9) | 5.2 |
| rain3 | 1.6 (1.5–1.7) | 1.8 (1.6–2.0) | 2.5 (2.3–2.6) | 0.9 (0.8–0.9) | 4.9 |
| snow1 | 2.3 (2.2–2.4) | 5.1 (4.8–5.4) | 2.5 (2.4–2.7) | 0.7 (0.6–0.8) | 4.8 |
| snow2 | 2.3 (2.2–2.4) | 4.3 (4.1–4.6) | 2.7 (2.5–2.9) | 0.8 (0.7–0.9) | 4.6 |
| snow3 | 2.2 (2.1–2.3) | 3.2 (3.0–3.4) | 2.9 (2.8–3.1) | 0.8 (0.7–0.9) | 4.6 |
| night2 | 1.5 (1.4–1.6) | 1.5 (1.4–1.7) | 2.3 (2.1–2.5) | 0.9 (0.8–1.0) | 5.9 |
| ADVERSE_POOLED | 1.9 (1.9–1.9) | 3.1 (3.1–3.2) | 2.5 (2.5–2.6) | 0.8 (0.8–0.8) | 4.7 |

## AP-level harm (Car mAP, status quo): removing a sensor *improves* mAP when the cell exceeds `all`

| condition | all | − camera | − lidar | − radar |
|---|---|---|---|---|
| clear0 | 0.800 | 0.780 | 0.186 | 0.795 |
| fog1 | 0.718 | 0.683 | 0.166 | 0.712 |
| fog2 | 0.618 | 0.603 | 0.123 | 0.610 |
| fog3 | 0.303 | 0.292 | 0.029 | 0.292 |
| rain1 | 0.782 | 0.765 | 0.180 | 0.778 |
| rain2 | 0.774 | 0.754 | 0.171 | 0.770 |
| rain3 | 0.750 | 0.737 | 0.148 | 0.747 |
| snow1 | 0.760 | 0.730 | 0.177 | 0.754 |
| snow2 | 0.729 | 0.696 | 0.169 | 0.721 |
| snow3 | 0.698 | 0.669 | 0.154 | 0.690 |
| night2 | 0.797 | 0.780 | 0.121 | 0.792 |

## Operating-threshold sweep (adverse pooled; τ chosen on clear = 0.405)

| τ | pooled harm % | help % | camera % | lidar % | radar % |
|---|---|---|---|---|---|
| 0.200 | 3.52 | 5.97 | 7.22 | 5.05 | 0.60 |
| 0.300 | 2.58 | 5.10 | 4.80 | 3.62 | 0.72 |
| 0.405 | 1.89 | 4.74 | 3.13 | 2.54 | 0.79 |
| 0.500 | 1.53 | 4.64 | 2.18 | 1.96 | 0.87 |
| 0.700 | 0.90 | 5.03 | 0.91 | 0.86 | 0.93 |
| 0.900 | 0.49 | 4.75 | 0.32 | 0.22 | 0.80 |

## Secondary slices (reported, not part of the pre-registered rule)

| slice | condition | pooled harm % | camera % | lidar % | radar % | help % | n pairs |
|---|---|---|---|---|---|---|---|
| multimodal_ge2 | clear0 | 3.46 | 4.37 | 3.91 | 1.97 | 9.45 | 52897 |
| multimodal_ge2 | ADVERSE_POOLED | 3.52 | 3.66 | 4.49 | 2.16 | 8.77 | 533358 |
| gt_backed | clear0 | 2.01 | 0.07 | 1.19 | 5.61 | 20.08 | 21343 |
| gt_backed | ADVERSE_POOLED | 2.26 | 0.19 | 0.93 | 6.43 | 21.61 | 192549 |
| multimodal_ge2_gt_backed | clear0 | 2.07 | 0.07 | 1.28 | 5.65 | 19.75 | 20701 |
| multimodal_ge2_gt_backed | ADVERSE_POOLED | 2.40 | 0.19 | 1.03 | 6.72 | 21.88 | 181610 |

**H1 verdict:** pooled adverse harm = 1.89 % (CI 1.86–1.92); refuted (<2 %): **True**; clear-weather harm = 1.93 %, weather-induced excess = -0.05 pp; AP-level harm cells: 0.
