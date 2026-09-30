# Pre-registration 2 — written 2026-09-23, BEFORE any of the experiments below were run

Registration 1 (`notes/falsification.md`, 2026-09-15) is unchanged and still scored exactly as written. This
registration adds new, separately scored hypotheses motivated by the senior-review critique
(`notes/senior_review.md`). Nothing here re-scores or replaces an R1 outcome.

**State of knowledge when written:** R1 outcomes known (H1 refuted on simulated weather; H2 supported; H3 holds
on snow, fails on fog3; CADC zero-shot Δ ≈ 0). No TruckScenes data had been downloaded; no external baseline,
normalised-probe, few-shot, cross-validation or conformal-repair result had been computed.

## Common definitions (inherited from R1 unless stated)
Car, ROI as R1 (x ∈ [0,69] m, |y| ≤ 39.5 m, |az| ≤ 40°), GT ≥ 25 LiDAR points (TruckScenes: ≥ 5 LiDAR points,
the dataset's own visibility convention, fixed now). Car mAP = mean centre-distance AP at {0.5,1,2,4} m.
Primary test statistic for every comparison: one-sided paired **scene-cluster** bootstrap, 1,000 resamples,
α = 0.05, Holm-corrected within each hypothesis family.

## R2-A — Real adverse weather with 4D radar (MAN TruckScenes v1.2, public, CC BY-NC-SA 4.0)
Frozen detectors only (same PointPillars + Faster R-CNN checkpoints; radar = training-free clustering of 4D radar
points incl. Doppler). Split by scene: train on the dataset's train scenes, test on val scenes. Adverse = scenes
tagged rain / snow / fog / hail / night (tags read from scene metadata, fixed before looking at results).

*Amendment A (same day, 2026-09-23, before any TruckScenes sensor data was downloaded):* the official val split
contains only 1 fog and 1 snow scene (counted from metadata tags only), too few for per-condition claims.
Therefore: (i) the official train_detect → val split is reported as the standard-protocol number; (ii) the
**primary** test of R2-A1..A3 is a **5-fold scene-level cross-validation over all 598 trainval scenes,
stratified by weather tag** (seed 0), pooling out-of-fold predictions so every scene is tested exactly once;
scene-cluster bootstrap over the pooled out-of-fold scenes. "Adverse" = weather ∈ {rain, snow, fog} or lighting
∈ {dark, twilight}. Sensors fixed now: LIDAR_TOP_FRONT + LIDAR_LEFT + LIDAR_RIGHT (merged), CAMERA_LEFT_FRONT,
RADAR_LEFT_FRONT + RADAR_RIGHT_FRONT (merged), in the LIDAR_TOP_FRONT frame.
- **R2-A1 (H2 on real weather):** per-object − per-frame-classifier Car mAP > 0 on adverse test scenes.
  *Refuted if* Δ ≤ 0 or p ≥ 0.05.
- **R2-A2 (H1 on real weather):** pooled decision-flip harm on adverse test scenes ≥ 2.0 %. *Refuted if* < 2.0 %.
- **R2-A3 (radar value):** per-object with 4D radar − per-object without radar > 0 on adverse test scenes.
  *Refuted if* Δ ≤ 0 or p ≥ 0.05.

## R2-B — External state-of-the-art late-fusion baselines (Boreas-sim adverse, TruckScenes adverse, CADC)
CLOCs-style learned candidate fusion; ProbEn-style probabilistic ensembling; Dempster–Shafer evidential fusion;
transformer set-fusion over each frame's candidates; per-frame mixture-of-experts gating. Each trained on the same
data as ours with the same budget. Oracle per-object sensor selector reported as an upper bound (descriptive).
- **R2-B1:** per-object ≥ every external baseline on adverse pooled Car mAP. *Refuted for a baseline* if that
  baseline beats ours with p < 0.05. All results reported regardless.

## R2-C — Ablation completeness (Boreas-sim, TruckScenes)
Components: absence (negative-evidence) term, learned position precisions, frame context, degradation probes,
conflict features, harm auxiliary loss, sensor dropout. Remove one at a time (retrained).
- **R2-C1:** each component's removal lowers adverse Car mAP. *Reported per component*; a component whose removal
  does not lower mAP (p ≥ 0.05) is reported as non-contributing and dropped from the claims.

## R2-D — Real-data transfer (CADC)
- **R2-D1 (resolution confound):** LiDAR probes normalised to each sensor's own expected returns (beam count,
  range) make zero-shot Boreas→CADC per-object − per-frame Δ > 0. *Refuted if* Δ ≤ 0 or p ≥ 0.05 → the gap is not
  explained by LiDAR resolution.
  *Clarification B (2026-09-24, written before any R2-D number was computed):* "normalised" is fixed as a
  spec-only map into Boreas (reference-sensor) units, no CADC labels and no fitting: point-count probes
  (l_log_pts, l_density, f_l_log_pts, f_l_log_far_pts) n → n/ρ with ρ = 32/128 (VLP-32C vs Alpha Prime beams,
  both 40° vertical FOV, ~0.2° horizontal step); l_ring_ratio recomputed from the true beam index, recovered as
  the nearest of the VLP-32C's 32 published beam elevations (median residual 0.001° on real sweeps). Boreas heads
  and τ applied zero-shot. D2 uses the same normalised probes, 3 random drive draws per k (seeds 1000+10k+r),
  CADC drives always in the fit set; out-of-draw units merged per test drive, paired drive-cluster bootstrap.
- **R2-D2 (few-shot):** with k ∈ {1, 5, 10} labelled CADC training drives (drive-level CV, remaining drives test),
  per-object − per-frame Δ increases with k and is > 0 at k = 10. *Refuted if* Δ(k=10) ≤ 0 or p ≥ 0.05.

## R2-E — Repairing risk control under severe shift (Boreas fog3, TruckScenes fog/snow)
Label-free at deployment: degradation-Mondrian conformal (groups = router-predicted degradation bins, calibrated
on clear + mild conditions) and adaptive conformal inference over the frame stream.
- **R2-E1:** repaired localisation coverage on fog3 ≥ 0.85 and CRC FNR ≤ 0.15 at α = 0.10.
  *Refuted if* either fails. Clear-weather radius inflation reported.

## R2-F — Statistical robustness
- **R2-F1:** 5-fold scene cross-validation over all 37 Boreas scenes: per-object > per-frame classifier in ≥ 4/5
  folds on adverse pooled mAP. *Refuted if* fewer.
- Full-data 5-seed run (already running when this was written): reported as mean ± sd.

## R2-G — Metrics breadth (descriptive; no refutation rule)
BEV-IoU AP@0.5 (fused centre + dimensions/yaw of the best-localised member), Pedestrian mAP, AP by range band
(0–20 / 20–40 / 40–70 m), near-range (< 30 m) miss rate, router latency (CPU, batch = 1).

**Rule:** every R2 outcome is reported in the paper, including refutations, marked as Registration 2.
