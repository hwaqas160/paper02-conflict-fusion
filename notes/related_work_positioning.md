# Related Work positioning against RAF — settled BEFORE finalising claims (PLAN §2, §9 wk 11)

**RAF**: H. Park, J. Jeong, K.-J. Yoon, *RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR for Robust
3D Object Detection in Adverse Weather*, arXiv 2607.04587 (6 Jul 2026). Read in full from `papers/`.

## Correction to PLAN.md

PLAN §2 says RAF "trains end-to-end". It does not, exactly: RAF **freezes a pretrained LiDAR–RADAR backbone**
and trains the camera backbone (Swin-T), MLP projectors, a reliability CNN, the BEV fusion encoder and the
detection head (their "Frozen" variant is the non-RAF control). Claiming "RAF is end-to-end" would be caught by
a reviewer. The accurate distinction is *feature-level fusion that retrains the fusion network and head* versus
*output-level arbitration over frozen detectors*.

## What RAF does

* Per-**pixel** camera reliability map R ∈ [0,1], gating camera features: F_gated = F_cam ⊙ (α + (1−α)R), α = 0.2.
* Weak supervision: image-level Clean / Mixed / Noisy labels → cross-modal (voxel ↔ pixel) cosine similarity →
  sparse pseudo-labels; CALM searches a w×w window for calibration tolerance.
* K-Radar (Sedan, IoU 0.5) and VoD. Best: L4DR+RAF 82.0 AP_BEV / 57.4 AP_3D total (Table 1).
* Their Table 2 shows camera harm in aggregate: 3D-LRF on *Noisy* 38.8 (L+R) → 25.4 (C+L+R Joint).
* Future work (their §5): "extending reliability estimation to LiDAR and RADAR modalities".

## Our differences — state all, precisely

1. **Level and retrofit.** Output-level late fusion over *frozen, off-the-shelf* detectors that only emit boxes and
   scores; the trained part is a ~10⁴-parameter router on cached detections (CPU, minutes). RAF requires feature
   access and retraining fusion encoder + head per backbone.
2. **Symmetric arbitration, including LiDAR.** RAF treats LiDAR–RADAR as the reliable reference and gates only the
   camera. We arbitrate camera, LiDAR and radar symmetrically — directly the extension RAF lists as future work.
   Snow clutter degrades LiDAR (Boreas Fig. 3); a camera-only gate cannot express "trust radar over LiDAR here".
3. **Per-object, with negative evidence.** Routing is conditioned on each object's own degradation probes (range,
   LiDAR density, crop contrast / blur / haze, radar SNR) and on cross-sensor conflict, and includes an explicit
   "should-have-seen-it" absence term. RAF is per-pixel on the image only; it has no per-object notion and no
   absence evidence.
4. **Harm as the target.** We define, measure (H1) and predict (H2) the counterfactual harm of adding a modality,
   per object. RAF shows aggregate harm but neither measures its rate nor predicts it.
5. **Guarantees.** Split-conformal localisation discs with conflict-inflated nonconformity and conformal risk
   control of the false-negative rate, with an explicit clear → adverse transfer test (H3). RAF has none.

## What we must NOT claim

* Not "better than RAF": different dataset, 4D radar vs scanning radar, feature vs late fusion. Exp 6 is a cited
  table, labelled as not comparable.
* Not "per-object beats per-pixel". Our per-frame baselines are the fair comparison for H2; RAF's granularity is
  orthogonal (it could in principle feed a probe).
* Not "real snow" for Boreas results — those are simulated; real snow is CADC (camera + LiDAR) and the unlabelled
  sim-to-real check.

## Other adjacent work to cite (from RAF's related work + our papers/ folder)

3D-LRF (LiDAR–4D-radar, weather-modulated flow), L4DR, RobuRCDet (camera confidence map), SAMFusion, MoE for
sensor failures; LRC-WeatherNet (`papers/LRC_WeatherNet.pdf`, our per-frame classifier baseline); FogDrive graded
fog (`papers/FogDrive_graded_fog.pdf`); 4D radar cooperative adverse weather; Safety-monitoring survey for the
introspection framing; Hahner et al. fog (ICCV 21) and snowfall (CVPR 22) LiDAR simulation; Tibshirani et al. 2019
weighted conformal; Angelopoulos et al. conformal risk control.
