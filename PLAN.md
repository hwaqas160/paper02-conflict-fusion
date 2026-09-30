# PAPER 02 — Conflict-Aware Late Fusion
### Which Sensor to Believe When Weather Degrades Them Differently

**Months 4–9 · the core method paper · highest ceiling · your Canada application**

---

## 1. PROBLEM STATEMENT

Camera degrades under poor visibility. LiDAR suffers scattering and attenuation in rain and
snow. Radar is largely immune to both and supplies Doppler velocity directly.

So when the three disagree in a snowstorm, the fusion layer must arbitrate — and current
systems arbitrate using fixed learned weights fitted in good weather. The literature already
reports that **adding a camera under adverse weather sometimes *degrades* performance**. That
is the arbitration failure showing through: the fusion layer cannot tell that one of its
inputs has become noise.

A deployed vehicle cannot retrain its detectors when it starts snowing. It needs a fusion
layer that measures degradation and routes trust accordingly, at runtime.

---

## 2. THE GAP

Reliability-aware fusion exists — **RAF** (arXiv 2607.04587) is the closest 2026 work. It
trains end-to-end, which (a) requires hardware you do not have, and (b) produces a system
nobody can retrofit onto an existing stack.

**Your three differentiators — state all of them explicitly in Related Work:**

1. **Late fusion over frozen, off-the-shelf detectors.** A deployable retrofit, not a new
   stack. This is also what makes it runnable on a P2000.
2. **Per-object degradation-conditioned routing**, not per-frame. A camera can be reliable for
   a near object and useless for a distant one in the same frame.
3. **Formal risk guarantees** on the fused output, carried over from Paper 01's machinery.

Cite RAF first and state your difference precisely. Reviewers will find it regardless; better
that you framed the comparison than that they did.

---

## 3. HYPOTHESES

Write to `notes/falsification.md` before experimenting.

- **H1 (harm exists).** There is a measurable set of cases where adding a degraded modality
  reduces detection quality versus omitting it. *Refuted if* the harm rate is under 2%.
- **H2 (predictable).** A degradation-conditioned head predicts those harm cases better than
  chance and better than a per-frame weather classifier. *Refuted if* per-object routing does
  not beat per-frame routing.
- **H3 (guarantee transfers).** Risk control calibrated on clear weather holds under snow.
  *Refuted if* empirical coverage collapses as in Paper 01 — **which is itself a publishable
  link between the two papers.**

---

## 4. CONTRIBUTIONS TO CLAIM

1. A per-object, degradation-conditioned arbitration head that sits on top of **any** frozen
   detectors.
2. Risk-controlled fusion: the fused output carries a coverage guarantee, and residual
   cross-sensor conflict inflates the nonconformity score.
3. Cross-condition external validation — calibrate in clear weather, deploy in snow.
4. Quantification of the **harm case**: how often does adding a modality make things worse,
   and can it be predicted in advance.

---

## 5. DATASETS

| Dataset | Role | Detail | Access |
|---|---|---|---|
| **Boreas** | Primary | 350+ km, repeated route over a full year. 128-ch Velodyne Alpha-Prime LiDAR, 360° Navtech CIR304-H scanning radar, 5 MP FLIR Blackfly S camera, cm-accurate ground truth. IJRR published. | Open AWS S3, **no account needed** |
| **CADC** | Snow validation | Canadian Adverse Driving Conditions, ~7k annotated frames of snowy driving | Waterloo, registration |
| **nuScenes** | Clear-weather calibration | Reuse from Paper 01 — no extra download | Already have it |
| SeeingThroughFog / DENSE | Optional fog | Adds a third degradation type | Registration |

See `data/DOWNLOAD.md`. Boreas commands are verified from the pyboreas README.

---

## 6. STRATEGIC NOTE — this is your Canada application, embedded in a paper

**Boreas was built by Barfoot's ASRL at UTIAS. CADC came out of Waterloo.**

If you publish a strong result on both, your application letter says *"I extended your
benchmark"* rather than *"I admire your work."* That is the entire reason these datasets were
chosen over nuScenes-only. Do not substitute them for convenience.

---

## 7. CODE

- **`code/pyboreas/`** — official UTIAS ASRL devkit. Cloned.
  - `BoreasDataset` loader, calibration handling, projection utilities
  - Download instructions verified in its README
- **Detectors:** do **not** train any. Use published pretrained checkpoints, run them in
  inference mode once, and cache every detection and feature to disk.
- **Your code:** `src/arbitrate.py` — a small GNN or MLP over cached detections. A few million
  parameters at most.

---

## 8. THE CRITICAL COMPUTE PATTERN

> **Run frozen detectors ONCE, offline. Cache everything. Never run them again.**

This is what makes the paper possible on a 5 GB Pascal card.

```
Week 1 of P02:  cache_detections.py  ->  runs for ~1-2 weeks in background
                                          outputs: per-frame detections, features,
                                          confidence, per-object range/visibility
Weeks after:    train arbitration head on cached tensors  ->  minutes per epoch
```

**Budget two weeks for the inference pass, not two days.** Start it in month 4 while Paper 01
is under review — this is the single most important scheduling decision in the programme.

**If you find yourself wanting to train a detector, the paper has drifted out of scope.**

---

## 9. WEEK-BY-WEEK PLAN

### Weeks 1–2 (programme months 4–5) · Data + offline inference
- [ ] Start Boreas download (see `data/DOWNLOAD.md`). It is large — start early.
- [ ] Install pyboreas, verify `BoreasDataset` loads a sequence.
- [ ] Select pretrained detectors for camera / LiDAR / radar. Record exact checkpoints.
- [ ] Write and launch `src/cache_detections.py`. **Let it run in background for 2 weeks.**

### Weeks 3–4 · Degradation characterisation
- [ ] Build `src/degradation.py`: per-object features — range, point density on object,
      image contrast/blur in the object crop, radar RCS and Doppler consistency.
- [ ] Quantify **H1**: how often does 3-sensor fusion underperform a 2-sensor subset?
- **GATE:** harm rate measured. If under 2%, the paper's premise is weak — tell me and pivot.

### Weeks 5–7 · The arbitration head
- [ ] `src/arbitrate.py`: per-object trust weights conditioned on degradation features.
- [ ] Baselines: (a) fixed-weight late fusion, (b) per-frame weather classifier routing,
      (c) confidence-only routing.
- **GATE:** H2 answered — per-object beats per-frame.

### Weeks 8–10 · Risk control + cross-condition validation
- [ ] Port `conformal.py` from Paper 01. Add cross-sensor conflict to the nonconformity score.
- [ ] Calibrate on clear-weather Boreas sequences, deploy on snow sequences and CADC.
- [ ] Report empirical vs nominal coverage — the direct link to Paper 01.
- **GATE:** H3 answered.

### Weeks 11–14 · Write and submit
- [ ] Related Work: position against RAF precisely, in writing, before you finalise claims.
- [ ] arXiv on submission day.

---

## 10. EXPERIMENTS

| # | Experiment | Proves |
|---|---|---|
| 1 | Harm-rate quantification across weather conditions | H1 — the motivating measurement |
| 2 | Per-object vs per-frame vs fixed-weight routing | H2 — the core claim |
| 3 | Ablation over degradation features | Which signals actually carry the routing decision |
| 4 | Clear → snow risk-control transfer | H3, and the link to Paper 01 |
| 5 | Boreas → CADC cross-dataset transfer | Generalises beyond one collection |
| 6 | Comparison against RAF-style end-to-end (cited numbers) | Honest positioning |
| 7 | Per-modality dropout stress test | Graceful degradation when a sensor fails entirely |
| 8 | Qualitative cases incl. one failure | Reviewers at these venues distrust papers without a failure case |

---

## 11. TARGET JOURNALS

| Priority | Venue | IF | Quartile | Cost |
|---|---|---|---|---|
| 1 | **Information Fusion** (Elsevier) | 15.5 | Q1 | Free on subscription route; APC $4,360 only if you choose OA |
| 2 | **IEEE T-IV** | 14.3 | Q1 | Free traditional route |
| 3 | IEEE T-ITS | 9.1 | Q1 | Free traditional route |
| 4 | IEEE Sensors Journal | — | Q1/Q2 | Free traditional route |

Information Fusion is an exact scope match and the highest impact available to you. It is also
slow and demanding — budget for a long review.

---

## 12. TARGET LABS

| Lab | Institution | Why |
|---|---|---|
| [ASRL](https://asrl.utias.utoronto.ca/~tdb/) | UToronto UTIAS 🇨🇦 | Barfoot's group **built Boreas**. Publishing on their benchmark is the strongest possible introduction. |
| [WAVELab](http://wavelab.uwaterloo.ca/) | Waterloo 🇨🇦 | AV state estimation and perception; Waterloo produced CADC |
| [TRAILab](https://www.trailab.utias.utoronto.ca/) | UToronto UTIAS 🇨🇦 | Waslander: 3D detection and tracking |
| [AMRL](https://amrl.cs.utexas.edu/) | UT Austin 🇺🇸 | Biswas: introspective perception — robots that know when they are failing |

---

## 13. RISKS

| Risk | Severity | Mitigation |
|---|---|---|
| Offline inference pass is the one place the P2000 genuinely hurts | **High** | Start it month 4, run in background, budget 2 weeks |
| RAF and successors occupy adjacent ground | **High** | Settle positioning in writing *before* building |
| Boreas download size / time | Medium | Start early; use `--include` filters to pull only needed sensors |
| Pretrained detectors may not exist for Boreas' exact radar format | Medium | Navtech scanning radar is unusual — check pyboreas examples first; fall back to LiDAR+camera and treat radar as the robustness probe |
| Harm rate turns out negligible | Medium | Gate at week 4 catches this before you build the method |

---

## 14. REFERENCE PAPERS (downloaded in `papers/`)

| File | Why it matters |
|---|---|
| `Boreas_dataset_IJRR.pdf` | Your primary dataset, by your target lab. **Read first.** |
| `RAF_reliability_aware_fusion.pdf` | **Nearest competitor.** Position against it precisely. |
| `4DRadar_cooperative_adverse_weather.pdf` | 2026 radar-under-weather work |
| `LRC_WeatherNet.pdf` | LiDAR/radar/camera weather classification — your per-frame baseline |
| `FogDrive_graded_fog.pdf` | Graded fog degradation, synthetic |
| `SafetyMonitoring_ML_perception_survey.pdf` | Survey — Related Work scaffolding |

---
---

# PART II — FULL IMPLEMENTATION PLAN (as built, 2026-09-15)

Sections 1–14 above are the original plan and are left untouched. This part turns them into an
executable plan, records where reality forced a change, and maps every item to code.

## 15. AMENDMENTS FORCED BY THE DATA AND THE LITERATURE

| # | Original assumption | Reality | Amendment |
|---|---|---|---|
| A1 | Boreas snow sequences can be scored against GT (§9 wk 8–10) | All public 3D labels are `boreas-objects-v1`, one **sunny** day (2020-09-01), 4,951 frames / 37 scenes | GT-preserving **graded weather simulation** on labelled frames (fog / rain / snow × 3 severities + night); real snow = **CADC** (labelled) + **unlabelled Boreas snow** sim-to-real check. `notes/data_reality.md` |
| A2 | RAF "trains end-to-end" (§2) | RAF freezes a LiDAR–RADAR backbone; trains camera branch, fusion encoder, head; gates **camera only**, per pixel | Differentiators re-stated precisely: output-level over frozen detectors; **symmetric** 3-sensor arbitration (RAF's own future work); per-object with negative evidence; harm as target; guarantees. `notes/related_work_positioning.md` |
| A3 | "Doppler consistency" radar feature (§9 wk 3–4) | Boreas' Navtech CIR304-H has no Doppler | Temporal persistence vs. the previous scan (ego-compensated) |
| A4 | Pretrained detectors for all three sensors (§7) | None exists for Navtech polar scans; mmdet3d ops don't build on Windows | LiDAR: pure-PyTorch port of the published mmdet3d PointPillars checkpoint (validated, Car AP@2m 0.81 on Boreas). Camera: torchvision COCO Faster R-CNN v2 + camera-only 3-D lift. Radar: training-free CA-CFAR (PLAN §13 fallback). `notes/detectors.md` |
| A5 | Port `conformal.py` from Paper 01 (§9 wk 8) | Paper 01 code is not in this workspace (`F:\CLAUDE\AI2`) | Implemented from primary sources in `src/conformal.py`; swap in Paper 01's version later if the two papers must share one implementation |
| A6 | P2000 available for the inference pass (§8) | GPU is shared and at 100 % with another job | CPU caching by default (`detectors.device: cpu`), dev subset first; flip to `auto` when the GPU is free |
| A7 | Hand-set sensor position priors | Measured on clear train positives: camera bearing error 0.8–2.5° (assumed 0.34°), camera self-reported range σ 3–7× too small, radar −1 m near-face range bias, LiDAR lateral 0.07–0.12 m. With guessed priors (and too few optimiser steps) the status-quo fusion scored 0.44 mAP vs LiDAR-only 0.78 — a straw-man baseline that would have inflated H1/H2 | `PositionPrior` fitted on training positives only (range bias, range σ, lateral σ, each linear in range); `min_steps` and a larger lr for constant routers. Status quo now 0.807 vs LiDAR-only 0.786 on clear test scenes |

| A8 | CADC needs registration (§5) | Public HTTP server, no registration (verified 2026-09-21) | `src/fetch_cadc.py` range-reads only front camera + LiDAR from each remote zip (~4× less traffic). Loader bugs found on real data and fixed: extrinsic key `T_LIDAR_00CAMERA`, intensity [0,1]→0–255; dims order and rectified images confirmed. `notes/cadc.md` |
| A5′ | (A5) Paper 01 code absent | It is at `F:\CLAUDE\AI1\paper01-coverage-transfer` | Cross-check test added. **Paper 01's quantile is one order statistic too high** (conservative; clamps to max when k > n). `notes/paper01_conformal_crosscheck.md` |
| A9 | Two snowy Boreas sequences for sim-to-real | `2021-01-26-10-59` has no post-processed poses; Boreas does not publish which sequences are "snowing" | 4 winter sequences, weather labelled by eye from a contact sheet. Real heavy snowfall matches simulated **fog2**, not snow1–3. `notes/sim2real_findings.md` |
| A10 | Full cache on CPU (~3.8 days) | GPU freed | Full stride-1 cache (4,951 frames) running on the P2000, ~1.9 s/item, reuses the 17,380 frozen-run items; frozen list snapshot `manifest_frozen1654.json`, `candidates --manifest` reproduces it |

## 16. METHOD SPECIFICATION

**Frames.** LiDAR-anchored; camera and radar are the nearest in time, all transforms via per-sensor Applanix
poses (`T_a_b = T_enu_a⁻¹ T_enu_b`). ROI = x ∈ [0, 69] m, |y| ≤ 39.5 m, |azimuth| ≤ 40° (overlap of the KITTI
PointPillars range and the 81° camera). GT: `labels_detection` (≥ 25 points); `labels_tracking` extras and Misc are don't-care.

**Candidates.** Greedy association — LiDAR↔radar by BEV distance (2.5 m); camera↔group by bearing and range
normalised by the camera's own monocular range σ. At most one detection per sensor per candidate.

**Per-object inputs.** Probes read at the best-localised member on the degraded data: geometry (range, azimuth);
LiDAR (points in box, range-normalised density, intensity, low-intensity fraction, z spread, ring coverage);
camera (in-image, visible fraction, crop luminance, RMS contrast, Laplacian variance, dark-channel haze, edges,
saturation); radar (peak / mean power, background, SNR, CFAR hits). Detector features (camera σ, truncation,
ground-vs-size range disagreement; radar cluster stats; LiDAR box volume). Conflict features (pairwise bearing /
range / BEV disagreements normalised by expected σ, score spread, class disagreement). Frame features (12).

**Fusion rule (shared by every arm).**
`z = b + Σ p_m(a_m ℓ_m + c_m) + Σ (1−p_m) f_m u_m`; fused range and bearing are precision-weighted with
`π_m = exp(λ_m^prior + Δ_m)`, where member ranges are bias-corrected and λ^prior comes from a `PositionPrior`
fitted on training positives (A7). Router outputs (a ≥ 0, c, u ≤ 0, Δ^r, Δ^θ, b) plus per-sensor harm logits.
Loss = BCE(existence) + 0.5·Huber(centre error / match radius) + 0.2·BCE(harm). Whole-sensor dropout p = 0.1.

**Arms (H2).** fixed_clear (status quo) · fixed · confidence-only · per-frame weather classifier · per-frame
oracle weather · per-frame features · **per-object (ours, 18k params)**. Snow is held out of head training.

**Harm (H1).** Counterfactual per candidate: decision with all sensors vs. with sensor m removed entirely, at the
clear-F1 threshold of the status-quo fusion.

**Guarantees (H3).** Localisation disc `‖p − p̂‖ ≤ q̂ σ̂(x)`; vanilla σ̂ = a + b·range; conflict σ̂ additionally
× exp(κᵀφ), φ = residual member dispersion around the fused centre + score spread. Calibrated on clear frames of
calibration scenes only. Also: range-Mondrian, likelihood-ratio weighted conformal (shift-aware, uses unlabelled
target frames), and conformal risk control of frame FNR.

**Splits.** By scene: train 45 % (of which 20 % of scenes = internal validation) / cal 20 % / test 35 %, seed 0.

## 17. EXPERIMENT → CODE MAP

| # | Experiment | Code | Output |
|---|---|---|---|
| 0 | Detector port validation, simulator panel | `detectors/pointpillars.py --selfcheck`, `viz_weather.py` | `notes/detectors.md`, `results/figures/weather_panel.png` |
| 1 | Harm rate by condition / sensor / range + AP-level harm | `run_experiments.exp1`, `harm.py` | `results/exp1_harm/` |
| 2 | Routing arms, paired bootstrap, harm AUROC, held-out snow | `exp2`, `arbitrate.py` | `results/exp2_routing/` |
| 3 | Feature-group ablation (+ no frame context) | `exp3` | `results/exp3_ablation/` |
| 4 | Clear → weather conformal transfer (vanilla / conflict / Mondrian / weighted; CRC) | `exp4`, `conformal.py` | `results/exp4_conformal/` |
| 5 | Boreas → CADC (radar removed; real-snow harm) | `fetch_cadc.py`, `cadc_cache.py`, `exp5` | `results/exp5_cadc/` |
| 6 | RAF cited numbers + property table | `exp6` | `results/exp6_raf/` |
| 7 | Whole-sensor failure at test time | `exp7` | `results/exp7_dropout/` |
| 8 | Qualitative win + failure case | `exp8` | `results/exp8_qualitative/` |
| S | Sim-to-real on unlabelled Boreas snow | `sim2real.py` | `results/sim2real/` |

## 18. COMPUTE BUDGET (measured)

| stage | per unit | dev run (347 frames × 11 conditions) | full (4,951 × 11) |
|---|---|---|---|
| detector cache, CPU (3 workers) | ~6 s / item | ~6.3 h | ~3.8 days |
| detector cache, free P2000 | ~2 s / item | ~2 h | ~1.3 days |
| candidates | ~0.05 s / item | 3 min | 45 min |
| all heads + experiments, CPU | — | ~20–40 min | a few hours |
| storage | ~25 KB / item cache | 95 MB | 1.4 GB |

## 19. REVISED WEEK-BY-WEEK PLAN (status as of 2026-09-15)

- [x] Selective download (phase A done; phase B running) · pyboreas working · checkpoints recorded
- [x] Detectors implemented and validated on Boreas GT (LiDAR port, camera lift, radar convention)
- [x] Weather simulator implemented, visually checked, unit-tested (monotone in severity, deterministic)
- [x] Degradation probes, caching pass (resumable, multi-process) — **dev run in progress**
- [x] Pre-registration written with numeric thresholds **before** any experiment (`notes/falsification.md`)
- [x] Association, fusion rule, all routing arms, harm, conformal, Exp 1–8 runner, synthetic end-to-end test
- [x] RAF positioning settled in writing
- [x] **Week-4 GATE triggered (preliminary).** On 4 fully-cached conditions the pooled adverse harm rate is
      1.995 % (CI 1.78–2.21) — just under the pre-registered 2 % line — with ~0.1 pp weather-induced excess and
      **0** AP-level harm cells. H2 nevertheless holds strongly (+0.056 mAP per-object vs per-frame, p ≈ 0.001).
      See `notes/prelim_findings.md`. Severe conditions (fog3, snow3, rain3, night2) were not in that run;
      re-decide the framing after the frozen 1,654-frame × 11-condition run.
- [x] Frozen-list run (1,654 × 11) → candidates → Exp 1–8 (2026-09-21). **Final week-4 GATE: H1 refuted** (1.70 %,
      CI 1.65–1.75, no weather excess, 0 AP-level harm cells). H2 supported (+0.075 mAP, p = 0.001; +0.085 held-out snow).
      H3 not refuted on snow; **fog3 breaks CRC (FNR 0.33 vs 0.10)**. Exp 7: per-object gain depends on radar.
      Outcome log filled in `notes/falsification.md`. Exp 5 (CADC, real snow): harm 3.4 % (camera harm > help); per-object gain 0 without radar. **Framing decision pending (user).**
- [x] `src.sim2real` on 4 winter sequences (A9)
- [x] Phase B download complete (4,951 frames) → full stride-1 cache running on GPU (A10) — then `candidates` → `--exp all --retrain`
- [x] CADC access (A8): download + cache + candidates chained in `logs/cadc_chain.ps1` → then `--exp 5`
- [~] Multi-seed (0–4) Exp 2 on the frozen run: running → `results/multiseed_exp2.md`; repeat on the full run
- [x] Outcome log in `notes/falsification.md`; manuscript skeleton `paper/manuscript.md` (method/setup written, results pending)
- [ ] Results + abstract + framing in the manuscript after the full run; arXiv on submission day

## 19b. SESSION 2026-09-22 — full run, framing decision, and a real bug caught before publication

**Framing decided (user + assistant, 2026-09-22):** lead with "per-frame weather routing doesn't work;
per-object arbitration does" (PLAN §19 option 1). H1 reported as a genuine pre-registered negative result on
simulated weather, with CADC's real-snow harm (3.4 %) as the counterpoint. Full LaTeX manuscript
(`paper/latex/`, Elsevier `elsarticle`) written: intro, related work (RAF numbers verified digit-for-digit
against the PDF), method, setup, limitations, conclusion, pre-registration and reproducibility appendices.
Every number in the manuscript is pulled from `results/*.json` by `paper/latex/make_paper_data.py` into
`autogen_numbers.tex` — never typed by hand — specifically so a reviewer (or a rerun) cannot silently
diverge from what the code actually produced.

**New analyses added for rigor** (`src/supplementary.py`, S1–S5): scene-cluster (not just frame-cluster)
bootstrap of H2; a score/position decomposition of the per-object gain; CLOCs-style and capacity-matched
controls (rules out "it's just more parameters" and "it's just conflict features, not probes"); every arm
retrained with radar removed entirely, evaluated on Boreas and CADC; textbook untrained fusion rules and
single-sensor baselines.

**A real bug found and fixed before it reached the paper:** the radar-free-retrained per-object arm scored
0.258 mAP on CADC (vs 0.740 for the simple baseline) — traced to `s4()` not zeroing CADC's radar
field-of-view flag to match the "structurally absent" convention `remove_modality()` used for training;
CADC's own candidate builder has no notion a whole sensor type is missing, so it left `fov_rad=1`, which
exploded through the radar-free arm's near-zero-variance standardizer column. Causally confirmed (0.258 →
0.728 once fixed) and fixed at the root (`notes/s4_fov_bug.md`). The abstract/results/limitations were
drafted to state the *corrected, weaker* finding (advantage survives radar removal on Boreas, not
established on CADC) rather than the dramatic but wrong "catastrophic collapse" the bug would have implied —
written before the corrected numbers landed, using hedged language that is accurate either way.

**CADC access corrected:** the server is public, no registration (`src/fetch_cadc.py`); PLAN §5's
"registration required" was never true. Loader bugs found and fixed against real data: wrong extrinsic key,
intensity scale (CADC 0–1 vs Boreas 0–255), image already rectified (do not undistort).

**Paper 01 cross-check:** its split-conformal quantile is one order statistic too conservative (documented,
not fixed — separate project). `notes/paper01_conformal_crosscheck.md`.

**Full run launched** (`logs/full_run_chain.ps1`, unattended): candidates (Boreas full 4,951 × 11 + CADC) →
`run_experiments --exp all --retrain` → multiseed (0–4) → `supplementary` (S1–S5, with the fov fix) →
`paper/latex/make_paper_data.py` → recompile. Status when this was written: candidates done, exp1–8 running.

## 20. OPEN RISKS AFTER IMPLEMENTATION

| Risk | Mitigation |
|---|---|
| Simulation realism is the main reviewer attack on A1 | sim-to-real feature check; CADC real snow; report severities as "simulated", never "real" |
| Radar is a weak detector here (AP ≈ 0 alone) → radar-specific conclusions are limited | say so; radar's role is corroboration; 4D-radar datasets (K-Radar) as future work |
| Harm on clear is non-zero (camera-only lifted boxes are FPs) | report clear0 harm as the baseline; the weather effect is adverse − clear |
| Small dev run → wide CIs | full stride-1 cache before any claim; CIs are frame-cluster bootstrap |
