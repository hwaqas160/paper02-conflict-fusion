# Falsification Note

> Filled in BEFORE running any experiment. Do not edit the hypothesis blocks after results arrive.
> Hypotheses from PLAN.md §3, made numeric.

**Date written:** 2026-09-15 13:31
**Git SHA at time of writing:** workspace is not a git repository. Stand-in: `source_tree_hash()` over
`src/` + `configs/` = `c741eba1e70e5118` (at this point only the data, detector and caching code existed;
no fusion, harm or conformal code had been run). Run `git init` and commit this file if you want a real SHA.

**Data state when written:** the frozen-detector cache had produced 3 smoke-test files (deleted). No fused
output, harm statistic or coverage number had been computed on any split.

## Common definitions (fixed now)

- **Frames / splits.** Boreas-objects-v1 labelled frames, split by *scene* (45 / 20 / 35 % train / cal / test,
  seed 0). All numbers below are on **test scenes** only.
- **Conditions.** clear0; fog1–3, rain1–3, snow1–3 (graded simulation, `src/weather.py`); night2.
  "Adverse" = every condition except clear0.
- **Class / region.** Car (primary), in the ROI where all three sensors see (x ∈ [0, 69] m, |y| ≤ 39.5 m,
  |azimuth| ≤ 40°), GT with ≥ 25 LiDAR points.
- **Status-quo fusion** = fixed-weight late fusion (`FixedWeights` in `src/arbitrate.py`): one global weight
  vector, fitted on **clear0 train** frames only.
- **Operating threshold τ** = fused-score threshold maximising F1 on clear0 train frames under the status-quo fusion.
- **Correct outcome** for a fused candidate: (score ≥ τ and matched to a GT car within 2 m) or
  (score < τ and not matched). Matching: greedy by best member centre distance, one GT per candidate.
- **Main detection metric.** Car mAP = mean of centre-distance AP at {0.5, 1, 2, 4} m (nuScenes protocol).

## H1 — harm exists
- **Claim:** under status-quo fusion, adding a degraded modality makes the fused decision worse for a
  measurable fraction of objects.
- **Metric (primary):** object-level *decision-flip harm rate* =
  #{candidates where modality m is present, decision with all modalities is incorrect, decision with m
  removed is correct} / #{candidates where m is present}, pooled over m ∈ {camera, lidar, radar} and over all
  adverse test conditions. 95 % CI by frame-cluster bootstrap (1000 resamples).
- **Metric (secondary):** AP-level harm: count of (condition, modality) cells where Car mAP with all three
  sensors is lower than with that modality removed.
- **Datasets / arms:** Boreas test scenes × 10 adverse conditions; clear0 reported for reference.
- **Refuted if:** pooled harm rate **< 2.0 %** (point estimate). Weak-premise flag (PLAN gate) if the *upper*
  CI bound is < 2.0 %.

## H2 — harm is predictable, per object better than per frame
- **Claim:** a degradation-conditioned per-object arbitration head predicts harm better than chance and better
  than a per-frame weather classifier, and routes trust better than per-frame routing.
- **Arms:** (a) fixed weights, (b) per-frame routing from a *predicted* weather class (frame-level features
  only), (b′) per-frame routing from the *oracle* weather label (reported, not used for refutation),
  (c) confidence-only routing, (d) per-object head (ours). Identical fusion rule, identical training data
  (train scenes, conditions except held-out snow), identical early stopping.
- **Metric 1:** pooled adverse-test Car mAP; paired frame-cluster bootstrap of (d) − (b), one-sided.
- **Metric 2:** AUROC of predicted per-modality harm probability vs realised decision-flip harm, (d) vs (b).
- **Refuted if:** mAP(d) − mAP(b) ≤ 0 **or** one-sided bootstrap p ≥ 0.05, **or** harm-AUROC(d) ≤ harm-AUROC(b).
- **Also reported, not part of refutation:** same comparison restricted to held-out snow (unseen by all heads).

## H3 — risk control calibrated on clear transfers to snow
- **Claim:** split-conformal guarantees calibrated on clear0 cal-scene frames hold on snow test frames.
- **Metric A (localisation):** empirical coverage of the fused-centre prediction disc
  ‖p̂ − p‖ ≤ q̂·σ̂ over matched test cars, nominal 1 − α = 0.90, *vanilla* score (σ̂ ≡ range-only scale).
- **Metric B (detection):** conformal-risk-control false-negative rate at α = 0.10 (frame-level FNR loss).
- **Refuted if:** vanilla coverage on pooled snow (snow1–3) test frames **< 0.85** (more than 5 points below
  nominal), or empirical FNR on pooled snow **> 0.15**. A refutation is the reported Paper 01 link.
- **Contribution-2 check (not a hypothesis gate):** conflict-inflated score closes ≥ 50 % of the snow
  coverage gap relative to vanilla, at no more than 25 % larger mean disc radius on clear0.

---

## Outcome log (fill in AFTER experiments)

| Hypothesis | Result | Supported? | Notes |
|---|---|---|---|
| H1 | Pooled adverse decision-flip harm **1.70 %** (CI 1.65–1.75); clear 1.79 % → weather excess −0.09 pp; AP-level harm cells **0 / 30**; harm *falls* with severity (fog3 0.8 %) | **No — refuted**, and weak-premise flag set (upper CI < 2 %) | Frozen 1,654-frame × 11-condition run, 2026-09-21. Multimodal slice 3.2 %; GT-backed radar harm 5.4 % (localisation pull). τ sweep: 3.5 % at τ = 0.2 → 0.5 % at τ = 0.9 |
| H2 | Δ mAP per-object − per-frame classifier **+0.075** (p = 0.001) adverse pooled, **+0.085** held-out snow; harm AUROC 0.943 vs 0.636 | **Yes** | Per-frame routing ≈ fixed weights even with oracle weather (+0.005). Confidence-only has higher harm AUROC (0.985) but routes worse (−0.074 mAP). Exp 7: gain vanishes when radar fails |
| H3 | Snow pooled vanilla coverage **0.977** (nominal 0.90), CRC FNR **0.073** (≤ 0.15) | **Yes — not refuted** | Contribution-2 check: conflict score does not "close a gap" (there is none on snow) but shrinks clear radius by 29 % at 0.96 coverage. **fog3 breaks the detection guarantee: FNR 0.33 vs α 0.10**; coverage 0.86–0.89. Weighted conformal degenerates (∞ radii, ESS collapse) |

---

**Rule:** if a hypothesis is refuted, report it. A clearly reported negative result is
publishable and is worth more than a positive result nobody can reproduce.

### Exp 5 addendum (CADC real snow, 7,000 frames, radar absent) — secondary, not part of the pre-registered rules
* Decision-flip harm **3.42 %** (CI 3.31–3.52) — above 2 % on *real* snow, where simulated snow gave 1.9–2.1 %.
  Camera harm 3.94 % exceeds camera help 2.01 % (the camera flips more CADC objects wrong than right); AP-level the
  camera still helps (0.739 all vs 0.705 LiDAR-only).
* Per-object − per-frame: **−0.001 mAP** (p = 1.0). With no radar the per-object gain disappears, consistent with Exp 7.
