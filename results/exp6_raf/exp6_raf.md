# Exp 6 — Positioning against RAF (cited numbers; different dataset, not a head-to-head)

> Numbers transcribed from Park, Jeong & Yoon, *RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR*, arXiv 2607.04587 (Tables 1–2), K-Radar, Sedan class, IoU 0.5. **Verify every digit against the PDF before citing** — they were read from the rendered page. They are on K-Radar with 4D radar and are *not* comparable to our Boreas numbers; the table documents what RAF reports, not a ranking.

## RAF Table 1 — AP_BEV

| method | Normal | Overcast | Fog | Rain | Sleet | Lightsnow | Heavysnow | Total |
|---|---|---|---|---|---|---|---|---|
| L4DR (Baseline, L+R) | 76.8 | 88.6 | 89.7 | 78.2 | 59.3 | 80.9 | 53.8 | 77.5 |
| L4DR (Joint, C+L+R) | 77.0 | 80.8 | 89.4 | 78.4 | 59.9 | 81.1 | 53.2 | 77.8 |
| L4DR (Frozen, C+L+R) | 79.8 | 83.5 | 92.2 | 79.1 | 59.2 | 85.7 | 58.4 | 78.6 |
| L4DR (RAF, C+L+R) | 82.0 | 88.2 | 93.9 | 82.6 | 63.0 | 88.5 | 60.4 | 82.0 |
| 3D-LRF (Baseline, L+R) | 62.1 | 88.5 | 84.8 | 69.9 | 37.7 | 84.6 | 36.0 | 64.5 |
| 3D-LRF (Joint, C+L+R) | 70.2 | 81.9 | 75.8 | 74.5 | 18.4 | 74.2 | 40.9 | 65.7 |
| 3D-LRF (Frozen, C+L+R) | 67.2 | 87.8 | 81.6 | 78.3 | 42.4 | 80.1 | 48.6 | 68.3 |
| 3D-LRF (RAF, C+L+R) | 70.8 | 88.0 | 82.3 | 74.3 | 43.3 | 84.3 | 49.6 | 71.0 |

## RAF Table 2 — AP_BEV by camera visibility (Clean / Mixed / Noisy)

| method | Clean | Mixed | Noisy |
|---|---|---|---|
| L4DR Baseline (L+R) | 77.0 | 78.4 | 59.0 |
| L4DR Joint | 81.5 | 77.5 | 55.6 |
| L4DR Frozen | 80.2 | 78.1 | 55.6 |
| L4DR RAF | 80.4 | 82.1 | 59.0 |
| 3D-LRF Baseline (L+R) | 68.8 | 73.0 | 38.8 |
| 3D-LRF Joint | 69.8 | 68.8 | 25.4 |
| 3D-LRF Frozen | 70.4 | 65.5 | 30.8 |
| 3D-LRF RAF | 72.2 | 74.5 | 38.4 |

**Harm evidence inside RAF's own numbers.** 3D-LRF on *Noisy*: L+R baseline 38.8 → C+L+R Joint 25.4 (−13.4 AP_BEV). Sleet: 37.7 → 18.4. Adding the camera without reliability handling hurts — the aggregate form of our H1. RAF recovers to 38.4 (Noisy), i.e. it *neutralises* camera harm; it does not report per-object harm rates or predict harm.

## Property comparison

| property | RAF (2026) | this work |
|---|---|---|
| fusion level | BEV feature fusion (camera branch + BEV fusion encoder + head trained) | output-level late fusion over frozen detectors |
| what is trained | camera backbone, projectors, reliability CNN, fusion encoder, detection head (Swin-T; RTX 3090) | ~10⁴-parameter router on cached detections (CPU, minutes) |
| retrofit to an existing stack | requires re-training the fusion network per backbone | any detectors that emit boxes + scores |
| which modalities are arbitrated | camera only (LiDAR–RADAR assumed reliable reference) | camera, LiDAR and radar symmetrically |
| granularity | per image pixel | per object (plus frame context) |
| supervision of reliability | image-level Clean/Mixed/Noisy labels → cross-modal similarity pseudo-labels | counterfactual harm labels from GT (no weather labels needed) |
| negative evidence ('should have seen it') | no | yes — absence × field-of-view term |
| formal guarantee on output | none | split-conformal localisation discs + conformal risk control of FNR |
| datasets | K-Radar, VoD | Boreas (+ graded simulation), CADC |