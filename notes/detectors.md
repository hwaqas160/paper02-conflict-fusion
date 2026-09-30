# Frozen detectors — exact checkpoints and validation (recorded 2026-09-15)

None of these is trained or fine-tuned in this project. Only *input conventions* are adapted.

## LiDAR — PointPillars (mmdetection3d v1.0.0, KITTI 3-class, SECFPN)

| item | value |
|---|---|
| checkpoint | `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth` |
| URL | https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/ |
| local | `code/checkpoints/pointpillars_kitti_3class_20220301.pth` (19,419,845 bytes) |
| implementation | pure-PyTorch port, `src/detectors/pointpillars.py` (mmdet3d CUDA ops do not build on Windows without nvcc) |
| range | x ∈ [0, 69.12], y ∈ [−39.68, 39.68], z ∈ [−3, 1] (KITTI frame) |
| input adaptation | z += 0.58 m (Boreas ground −2.31 m vs KITTI −1.73 m; ground = median GT box bottom, train split); intensity / 255 |
| NMS | per-class rotated BEV NMS (shapely), IoU 0.01, score_thr 0.05 (lower than mmdet3d's 0.1: late fusion wants low-confidence boxes) |

**Port validation** (`python -m src.detectors.pointpillars --selfcheck`, 40 clear Boreas frames, Car, ROI):

| variant | AP@0.5 m | AP@1 m | AP@2 m | AP@4 m |
|---|---|---|---|---|
| legacy voxel-centre (mmdet3d ≤ 1.0 in-place quirk), z_shift +0.58 | 0.660 | 0.793 | 0.814 | 0.823 |
| non-legacy, z_shift +0.58 | 0.000 | 0.000 | 0.000 | 0.000 |
| legacy, z_shift +0.28 | 0.674 | 0.818 | 0.848 | 0.854 |
| legacy, z_shift +0.88 | 0.610 | 0.733 | 0.756 | 0.770 |

The 0.00 for non-legacy confirms the checkpoint was trained with mmdet3d's `legacy=True` PillarFeatureNet,
where the voxel-centre offset overwrites the xyz channels in place. The principled z shift (+0.58) is kept;
±0.3 m moves AP@2m by ≤ 0.06, so the choice is not tuned on test data.

## Camera — Faster R-CNN ResNet-50-FPN v2 (torchvision, COCO)

| item | value |
|---|---|
| weights | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` → `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` |
| local | `code/checkpoints/torch/hub/checkpoints/` (175,221,657 bytes); set `TORCH_HOME=code/checkpoints/torch` |
| class map | COCO car, truck → Car; person → Pedestrian; bicycle, motorcycle → Cyclist; bus → ignored (Boreas 'Misc') |
| input | half resolution (1224 × 1024) for every condition incl. clear |
| 3-D lift | camera-only: ground-contact range (flat road at the fixed ground height) fused with size-prior range; centre pushed back by (L+W)/4; bearing from the box centre ray |

**Validation** (25 clear frames, Car, ROI): AP@1/2/4 m = 0.109 / 0.233 / 0.480. Range error of confident
matches grows with range — MAE 0.71 m (< 15 m), 1.41 m (15–30 m), 4.70 m (30–70 m). This range-dependent
reliability is exactly what per-object routing is meant to exploit.

## Radar — CA-CFAR + BEV clustering (training-free)

No pretrained detector exists for Navtech CIR304-H polar scans (PLAN §13 risk). Classical pipeline in
`src/detectors/radar_cfar.py`: 1-D CA-CFAR per azimuth (guard 4, train 24, offset 0.09, min power 0.2, ≤ 80 m) →
LiDAR frame via Applanix poses → 0.5 m BEV grid connected components → PCA box; score = monotone in hit count and
mean power. No Doppler in Boreas' Navtech; temporal persistence against the previous scan is the substitute cue.

**Validation** (25 clear frames): azimuth convention confirmed — 70 % of CFAR hits lie within 1 m of LiDAR
structure (31 % with the sign flipped). As a stand-alone Car detector it is useless (AP ≈ 0, 44 clusters/frame,
mostly static clutter), but 58 % of GT cars have a radar cluster within 2.5 m: radar is corroborating evidence,
not a detector. This is reported as is.

## Throughput (CPU, shared i9-10980XE under load)

~6 s per (frame, condition) with 3 workers × 5 threads: LiDAR 2.4 s, camera 8 s (per worker), radar 0.3 s,
probes 0.6 s. On a free P2000: LiDAR 1.0 s, camera 0.4 s. Set `detectors.device: auto` when the GPU is yours.
