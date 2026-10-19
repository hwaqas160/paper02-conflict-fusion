# E8-C: does the per-object advantage survive an in-domain LiDAR detector?

Written 2026-10-12, BEFORE any router was trained or scored on the new detector's output. Exploratory (not one of the
registered hypotheses), but with the decision rule fixed here so the outcome cannot be chosen after the fact.

## Why
The paper's LiDAR detector is a KITTI PointPillars checkpoint applied to a truck-mounted LiDAR it was never trained on.
A reviewer can say the router merely compensates for a weak, mismatched detector. This tests that directly.

## Detector (src/ts_detector.py)
- Start from the same KITTI checkpoint. Fine-tune the Car anchors of the head and the whole network on TruckScenes Car labels
  (>= 5 LiDAR points, the dataset's own visibility rule; other classes are don't-care regions). Assignment by centre distance
  (no rotated-IoU matching); focal + smooth-L1 + sin-difference + direction losses; AdamW, one-cycle LR, rotation +-0.2 rad and
  y-flip augmentation.
- Leakage control by cross-fitting: scenes are split into two detector folds, stratified by weather (seed 0). Detector k is
  trained on fold k only and applied ONLY to the other fold. Every frame's detections therefore come from a detector that
  never saw its scene, so the router's scene cross-validation stays clean. Trained on all weather conditions of its fold.
- Detector quality is reported on the held-out fold: LiDAR-only Car AP at 0.5/1/2/4 m, by condition, KITTI vs in-domain.

## Router test (same protocol as R4-4)
- Replace only the LiDAR Car detections (boxes, scores) and recompute the LiDAR probes at the new boxes. LiDAR Pedestrian and
  Cyclist detections, camera, radar, frame features and ground truth are unchanged. Candidates are rebuilt, then the
  registered five-fold scene cross-validation retrains the per-object router and the per-frame weather classifier from scratch.
- Sanity check before use: recomputing probes for the ORIGINAL LiDAR boxes must reproduce the cached probes.

## Decision rule (same as R4-2 / R4-4)
Primary comparison: per-object minus per-frame classifier, adverse-pooled out-of-fold Car mAP, scene-cluster bootstrap
(10,000 resamples; 1,000 if the run is too slow, stated), one-sided.
- NOT SUPPORTED if the pooled difference is <= 0, or p >= 0.05, or the per-object arm is ahead in fewer than 4 of 5 folds.
- Also reported, no gate: the difference at the original detector (+0.064) for comparison; the set-transformer, CLOCs and the
  fixed-weight status quo on the new candidates; the minimal per-object arms of E10 on the new candidates; the share of the
  gap to the oracle selector that is closed.
- If the advantage shrinks a lot but stays positive, we report the shrinkage as it is. If it vanishes, the claim is stated as
  holding only for a weak or mismatched LiDAR detector.

## Known limits (stated up front)
- Car anchors only; assignment is centre-distance based, so this is a competent in-domain detector, not a tuned state of the
  art one. Training uses a random subset of frames per epoch (consecutive keyframes are highly correlated).
- Two detectors each see half of the scenes (so each is weaker than one trained on everything). That is the price of keeping
  the router evaluation leak-free.
