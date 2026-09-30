# Senior-reviewer critique of Results + Limitations (2026-09-23)

Target venue standard: Information Fusion / IEEE T-IV (top-tier). Written as the reviewer who wants to reject.

## A. What a hostile reviewer attacks today

| # | Weakness | Why it is fatal-ish | Where it shows |
|---|---|---|---|
| W1 | Main results are on **simulated** weather only | "Your headline Δ0.077 is on your own simulator; on real snow (CADC) the gain is ~0" | 06 §6.2 vs §6.7 |
| W2 | The method **does not win on the only real-weather dataset** | Reads as: works on synthetic, fails on real | CADC Δ −0.005, p=1.0 |
| W3 | "Radar" is a training-free CFAR with AP≈0 | Three-sensor arbitration claim rests on a toy radar; RAF uses 4D radar | 05 §5.2 |
| W4 | "Any frozen detectors" claim tested with **one detector per modality** | Title-level claim unsupported | 01 contrib. 1 |
| W5 | Baselines are all our own arms | No external late-fusion SOTA (CLOCs, ProbEn, evidential/DS fusion, MoE gating) | 06 Table 2 |
| W6 | Only centre-distance mAP, Car only | No IoU-based AP, no NDS, no pedestrians, no downstream (tracking) metric | 05 §5.3 |
| W7 | Conformal guarantee fails under fog3; weighted CP degenerates; no fix offered | "Contribution 2 is a negative result without a remedy" | 06 §6.4 |
| W8 | One scene split, 13 test scenes, multi-seed only on dev subset | Split-luck objection | 06 §6.2 |
| W9 | H1 refuted on sim → weak motivation | "Why build a harm router if harm is rare?" | 06 §6.1 |
| W10 | Sim-to-real is a nearest-neighbour feature comparison | Not a validation, just a sanity check | 05 §5.1 |
| W11 | Radar-vs-LiDAR-resolution confound on CADC left unresolved | Limitations admits it; reviewer asks "why didn't you test it?" | 07 |
| W12 | No latency / deployability number despite "retrofit, real-time" framing | Unsubstantiated deployability | 01, A2 |

## B. Concrete upgrades (ranked by acceptance impact per unit effort)

### U1. Real adverse-weather, labelled, 3-sensor data (kills W1, W2, W3, W9) — HIGHEST VALUE
- **K-Radar** (Paek et al., NeurIPS 2022 D&B): LiDAR + camera + **4D radar**, real fog / rain / sleet / light & heavy
  snow, 3D boxes. The dataset RAF reports on → gives a **same-benchmark** comparison instead of "not comparable".
- **SeeingThroughFog / DENSE** (Bijelic et al., CVPR 2020): LiDAR + camera + radar, real fog/snow/rain, 3D labels.
- **nuScenes rain + night splits**: camera + LiDAR + 5 radars, real labels; already downloaded for Paper 01 (zero
  acquisition cost).
- Protocol: train router on each dataset's clear/train split, test on its adverse split; report per-condition.
  Keep Boreas-sim as the *controlled graded-severity* study (its real strength), not the headline.
- Compute: one frozen-detector caching pass per dataset (days, once) + minutes per router. Justified: converts
  the paper from "works on our simulator" to "works on three real adverse-weather benchmarks" — the single
  largest acceptance lever.

### U2. Retrofit on RAF's own frozen backbones (kills W3, W5; turns positioning into a head-to-head)
- Use the released L4DR / 3D-LRF checkpoints on K-Radar **as frozen detectors** and put the router on top.
  Report ours-on-top-of-L4DR vs RAF-on-L4DR on identical splits. This is the reviewer's first question.

### U3. Detector-agnosticism sweep (kills W4)
- LiDAR: PointPillars, SECOND, CenterPoint (+ one strong one, e.g. PV-RCNN). Camera: Faster R-CNN, a DETR/YOLO
  family 2D detector, and a monocular 3D detector (e.g. FCOS3D). Radar: CFAR, a learned radar detector
  (e.g. RadarPillars on 4D radar).
- Report a detector-combination grid: router gain is positive in every cell (or say where it is not).
- Zero-shot transfer: router trained on detector set A, applied to set B (tests "any detectors" literally).

### U4. External SOTA late-fusion baselines (kills W5)
- **CLOCs** (Pang et al., IROS 2020) re-trained on our candidates; **ProbEn** probabilistic ensembling
  (Chen et al., ECCV 2022); **Dempster–Shafer / evidential** fusion; a **transformer set-fusion** head over
  candidates; **MoE gating** per frame; detector-confidence **temperature-scaling** + max.
- **Oracle per-object selector** (GT picks the best sensor per object): an upper bound → report "% of oracle
  headroom captured". Reviewers love headroom numbers.

### U5. Fix the real-data gap you already diagnosed (kills W2, W11)
- **Sensor-resolution-normalised probes**: point counts / density / ring coverage divided by the LiDAR's
  expected returns at that range for its own beam pattern. Re-run Boreas→CADC. This directly tests the
  confound the Limitations section currently only names.
- **Few-shot in-domain adaptation**: fine-tune the router on k ∈ {0, 1, 5, 10} CADC drives (drive-level CV).
  A curve showing the gain appears with a handful of labelled real drives is a strong, honest result.
- Pre-register these as **Registration 2** (dated, before running) so they are not post-hoc rescue.

### U6. Repair the conformal contribution instead of only reporting its failure (kills W7)
- **Adaptive conformal inference** (Gibbs & Candès, NeurIPS 2021) / online CP over the frame stream — needs no
  target labels at deploy time *if* driven by delayed feedback; report long-run coverage under fog3.
- **Degradation-Mondrian** CP: group-conditional quantiles on label-free predicted-degradation bins (the router's
  own frame-context / weather-classifier output), calibrated on a *mix* of clear + mild conditions.
- **Conformal risk control with covariate-shift weights using the degradation features** (not raw frame
  features) and ESS-aware clipping; report where it restores FNR ≤ α.
- Report conditional coverage by range band and by degradation decile (not just marginal).

### U7. Metrics + classes (kills W6)
- Add BEV-IoU and 3D-IoU AP (KITTI-style, 0.5/0.7), an NDS-style composite, Pedestrian (and Cyclist where
  labelled), per-range-band AP (0–20/20–40/40–70 m), and a safety metric: **near-range (< 30 m) miss rate**.
- Downstream: plug fused detections into a simple tracker (AB3DMOT) and report AMOTA — shows the gain survives
  into the stack.

### U8. Statistical robustness (kills W8)
- **5-fold scene-level cross-validation** over the 37 Boreas scenes (each scene tested once), plus the full-data
  5-seed run (running now). Report mean ± sd across folds and a meta-analytic CI.
- Holm–Bonferroni across the H2 comparison family.

### U9. Deployability evidence (kills W12)
- Router latency per frame (CPU, batch=1) and memory; end-to-end added latency over the detectors. Trivially
  cheap and removes a free objection.

### U10. Stronger sim-to-real (kills W10)
- Per-feature two-sample tests (KS / MMD) between real Boreas winter frames and each simulated severity;
  **detector-level** check: detector AP/recall degradation curves on real CADC snow vs simulated snow on Boreas.

## C. Revised contribution statement (use after U1–U5 land)

1. **A detector-agnostic, per-object arbitration layer** that retrofits any frozen camera / LiDAR / radar
   detectors with an explicit negative-evidence term; validated across ≥3 detector combinations and three real
   adverse-weather benchmarks, and on top of RAF's own frozen backbones on K-Radar.
2. **The first counterfactual measurement of sensor harm on real adverse weather**, showing it is rare on
   controlled simulation but material on real snow/fog, and predictable per object.
3. **Degradation-aware risk control**: conformal localisation/FNR guarantees whose failure under severe shift we
   characterise *and* repair with label-free degradation-conditioned calibration.
4. **A graded, GT-preserving adverse-weather extension of Boreas** as a controlled severity benchmark,
   with a quantitative sim-to-real audit.

## D. Revised Limitations (preemptive, after upgrades)

- *Simulation*: used only for the controlled severity study; every headline claim is replicated on real data.
- *Radar*: results reported both for a weak scanning radar and a 4D radar; conclusions stated per radar type.
- *Detectors*: agnosticism shown over the tested grid; untested families named.
- *Real-weather transfer*: zero-shot cross-dataset gain is small; few-shot curve reported; the resolution confound
  resolved (or bounded) by normalised probes.
- *Conformal*: marginal guarantees hold under exchangeability; under severe shift we report the repaired variant's
  empirical coverage and state it is empirical, not a finite-sample guarantee.
- *Scope*: Car + Pedestrian; no night-time thermal; no multi-agent.
- *Pre-registration*: Registration 1 (original H1–H3) and Registration 2 (upgrades), both dated; all outcomes
  reported.

## E. Acceptance-probability checklist

- [ ] Headline result on ≥1 real labelled adverse-weather benchmark (K-Radar or DENSE) — non-negotiable
- [ ] Same-benchmark comparison with RAF (router on RAF's frozen backbones)
- [ ] ≥2 external late-fusion baselines (CLOCs, ProbEn)
- [ ] Detector-combination grid (≥3 combos)
- [ ] Oracle per-object upper bound + % headroom captured
- [ ] IoU-based AP + Pedestrian + range bands + near-range miss rate
- [ ] Conformal repair that restores coverage under fog3 (or honest bound)
- [ ] 5-fold scene CV + full-data 5 seeds
- [ ] Resolution-normalised probes → CADC; few-shot curve
- [ ] Latency table
- [ ] Registration 2 dated before running new experiments
- [ ] Code + cached detections + configs released (Zenodo DOI)
