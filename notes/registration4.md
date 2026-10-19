# Registration 4 (approved 2026-10-05)

Status: APPROVED for R4-1, R4-2, R4-3, R4-4 and the exploratory items listed above (approval given 2026-10-05). Hypotheses and thresholds are frozen at approval. Once
frozen, the date and do not edit the thresholds afterwards. Any change after the first run requires a
new registration.

Trigger for this registration (stated before any run): the internal peer review (score 5/10) found that the
per-object arm receives a superset of the per-frame arms' input (probes, conflict and detector features, plus
frame-level features), so "per-object beats per-frame" is not yet isolated from "more input beats less input";
and that several ablations contradict the method's stated selling points.

## Common definitions
- Adverse pooling, metrics, and bootstrap as in Appendix A (Car mAP at {0.5, 1, 2, 4} m; scene-cluster bootstrap,
  10,000 resamples for this registration; one-sided tests with Holm correction within this family).
- Information-equal arms: every arm receives the same underlying input pool (see E1 below). Only the decision unit differs.

## Hypotheses

**R4-1 (decision unit at equal information, Boreas).** The per-object router beats the frame-pooled router
(information-equal, per frame) on adverse-pooled Car mAP.
- Refuted if: the mean difference is at most 0, or the one-sided scene-cluster bootstrap p is at least 0.05.

**R4-2 (decision unit at equal information, TruckScenes).** Same comparison under the five-fold scene-level
cross-validation used in R2-A.
- Refuted if: the per-object arm is ahead in fewer than 4 of 5 folds, or the pooled out-of-fold difference is at most 0.

**R4-3 (support check versus standard out-of-distribution scores).** The label-free support check flags
severe-condition frames at least as well as the best of Mahalanobis distance, k-nearest-neighbour distance (k = 10)
and a logistic density-ratio score.
- Metric: AUROC for flagging severe-condition frames (Boreas severe fog and snow; TruckScenes snow).
- Refuted if: the support check's AUROC is more than 0.02 below the best of the three baselines.

**R4-4 (robustness to a stronger camera).** The per-object advantage over the per-frame classifier persists when
the camera's 3D position is replaced by LiDAR-projected depth inside the camera box (sensitivity test, same frozen
2D detections).
- Refuted if: the pooled out-of-fold difference is at most 0, under the same fold rule as R4-2.
- Caveat stated in advance: this couples camera and LiDAR, so it tests sensitivity to camera quality, not a new detector.

## Exploratory (no refutation rule; reported as exploratory in the paper)
- E4: why the harm-auxiliary loss helps (calibration, where the gain sits, learned gains and penalties).
- E6: interpretability measurements, including counterfactual agreement of the largest-gain sensor with the sensor
  whose removal changes the decision most.
- E7: CADC error decomposition and per-probe distribution shift.
- E8-C: controlled detector-quality sweep (score and position noise around ground truth).
- E9: width-40 and no-precision routers reported on TruckScenes CV as disclosure, not adopted.

## Runs that need separate approval
The compute for R4-1 to R4-4 and the exploratory analyses is CPU-only and runs on cached candidates where possible.
Expected time: R4-1 under 30 minutes on Boreas, about 1.5 hours on TruckScenes CV; R4-3 under 2 hours; R4-4 about
4 to 6 hours (candidate rebuild). Authors to confirm before any run starts.

## Approval
- Hypotheses and thresholds approved by: the authors, in chat, 2026-10-05 (R4-1, R4-2, R4-3, R4-4, exploratory items).
- Compute approved: yes, for R4-1 to R4-4 and the exploratory analyses, including the candidate rebuild for R4-4.
