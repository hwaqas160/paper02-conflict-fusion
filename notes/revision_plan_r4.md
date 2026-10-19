# Revision plan R4: raising "Which Sensor to Believe?" to a competitive Information Fusion submission

Written 2026-10-05, after the strict internal review (score 5/10, expected outcome: major revision with real
rejection risk). Target after this plan: about 7 to 7.5/10, meaning a paper whose central claim survives the
obvious reviewer objection, whose tables agree with its text, and whose limits are stated by measurement rather
than by assertion.

Nothing in this plan has been run yet. Each experiment below needs your go-ahead, and the experiments in
Phase 2 must not start until Registration 4 (Phase 0) is written and dated.

---

## 0. Ground rules (apply to every phase)

1. **Register before running.** All new experiments go into `notes/registration4.md`, written and dated before
   the first run, with a refutation rule per hypothesis. This keeps the paper's main strength, sequential
   pre-registration, intact. Anything decided after seeing a number is labelled *exploratory* in the paper.
2. **No model swaps after the fact.** The reported router stays the reported router. Better-scoring variants
   we already know about (Section 2, item T3) are disclosed, not adopted.
3. **Every number comes from a JSON file** through `paper/latex/make_paper_data.py`. No hand-typed numbers.
4. **Do not edit `src/arbitrate.py` or `src/baselines_ext.py`.** Their source hashes are part of the model-cache
   fingerprints; editing them forces refits of every cached model. New arms and analyses go in a new module,
   `src/r4_experiments.py`, which imports from them.
5. **Compute budget is CPU only.** The GPU on this machine is a Quadro P2000 (5 GB) that other users keep at
   about 98 % memory and 82 % utilisation, so it is not usable, and 5 GB would not train a feature-level fusion
   network anyway. The machine has 36 CPU cores, shared. All estimates below assume CPU.
6. **The Zenodo record is frozen once published.** If the DOI is published, revised code and results go up as a
   *new version* of the same record (Zenodo keeps one concept DOI for all versions). The paper should then cite
   the new version's DOI.

---

## 1. Phase 0: Registration 4 (write first, about 2 hours, no compute)

File: `notes/registration4.md`. Trigger, stated honestly: the internal review found that the per-object arm
receives more input than the per-frame arms, and that several ablations contradict the method's stated selling
points. Registration 4 tests these before any new run.

| ID | Hypothesis | Refuted if | Data |
|---|---|---|---|
| R4-1 | With **equal information**, deciding per object still beats deciding per frame (Boreas). | `object − frame_pooled` adverse-pooled Car mAP difference ≤ 0, or one-sided scene-cluster bootstrap p ≥ 0.05 | Boreas test scenes |
| R4-2 | Same, on real weather. | Same rule, out-of-fold, ahead in fewer than 4 of 5 folds also refutes | TruckScenes 5-fold CV |
| R4-3 | The label-free support check is at least as good as simple out-of-distribution scores. | Its AUROC for flagging severe-condition frames is more than 0.02 below the best of Mahalanobis, k-nearest-neighbour and a logistic density-ratio score | Boreas severe conditions, TruckScenes snow |
| R4-4 | The per-object advantage survives a much stronger camera. | `object − frame_cls` on TruckScenes CV ≤ 0 with the stronger-camera candidates (option E8-B) | TruckScenes 5-fold CV |
| R4-E | Exploratory, no refutation rule: harm-auxiliary-loss mechanism analysis, interpretability measurements, CADC error decomposition, per-scene tables, 10,000-resample statistics. | n/a | existing models |

The selection and reporting rules: every R4 outcome is reported as scored, in a new subsection and in the
scorecard (Table 20). If R4-1 or R4-2 is refuted, the paper's framing changes (see Section 6, risk 1).

---

## 2. Phase 1: corrections that need no new experiments (about 1 day)

These are ranked by how badly a reviewer would react if they found them.

| # | Fix | File(s) | Exact change | Done when |
|---|---|---|---|---|
| **T1** | **Table 4 sign error** | `sections/06_results.tex` (table `tab:ablation`) and the paragraph above it | Remove the hard-coded `-` prefix so every row prints one convention, *variant minus full*. Correct values: frame context +0.0001, geometry −0.0119, LiDAR probes −0.0142, camera probes −0.0036, radar probes +0.0005, detector features −0.0035, conflict features +0.0031, detector confidence +0.0000, frame-level features +0.0071. Rewrite the paragraph: LiDAR probes and geometry carry the decision; removing conflict features or frame-level features slightly *improves* adverse mAP on Boreas; frame context has no measurable effect. Drop "consistent margin largest on held-out snow" unless the held-out-snow number supports it (check `results/exp3_ablation/exp3_ablation.json`). | Rendered Table 4 has no double minus; text and table agree in sign for every row |
| T2 | Uncited "companion paper" | `sections/06_results.tex` line ~197 | Either cite the coverage-transfer paper properly (it then also appears on the title page side, not in `main2.tex` if blinded) or replace with a neutral sentence citing published conformal-shift work [26, 27]. Recommended: neutral sentence, since `main2.tex` is meant to be free of author identity. | No self-reference in `main2.tex` |
| T3 | Best model is not the reported model | `sections/06_results.tex` (capacity paragraph), `06b_results_r2.tex` (§7.3), `07_limitations.tex` | State plainly: a narrower router (width 40) scores 0.783 and removing the learned position precisions scores 0.782, both above the reported 0.777; we keep the registered model rather than choosing after the fact. Soften every sentence that presents the learned position precisions as a working part of the interpretable rule (the gain is 97 % existence scoring). | Limitation paragraph present; no sentence claims precisions help |
| T4 | Overclaimed differentiators | `01_introduction.tex`, `02_related.tex`, `04_method.tex` §4.2 | The absence ("should-have-seen-it") term and the conflict features are presented as what sets the method apart from RAF. The retrained ablation shows neither contributes (absence +0.0021, p = 0.400 Boreas; +0.0010 TruckScenes). Keep them as *expressible* features of the rule, not as sources of the gain. The differences from RAF that do hold: output-level retrofit, symmetric arbitration, per-object decision unit. | No sentence attributes accuracy to the absence term or conflict features |
| T5 | Support check presented as a result | Abstract, `01_introduction.tex` contribution 5, `06b_results_r2.tex` §7.6, `08_conclusion.tex` | Label it exploratory (it was added after the scored result) everywhere, including the abstract. Reconcile "either holds or announces that it does not" with TruckScenes snow, where the miss rate on *unflagged* frames is 0.204, worse than the overall 0.167. New wording: on simulated severe conditions the flag isolates the failures; on real snow it does not. | Abstract and conclusion consistent with Table 10 |
| T6 | Formal vs observed guarantees | `04_method.tex` §4.5, `06_results.tex` §6.4, `06b_results_r2.tex` §7.6, new small table | Add one table: which property is a finite-sample guarantee under exchangeability (clear-weather coverage, clear-weather CRC), which is only *observed* under shift (degradation-normalised coverage in snow/fog), which fails (miss rate in severe fog and real snow). Replace "guarantee" with "observed coverage" wherever exchangeability does not hold. | `grep -n guarantee` shows each use qualified |
| T7 | CLOCs beats the router on CADC | Abstract scope sentence, `07_limitations.tex` | Say it: on CADC (no radar), CLOCs-style fusion scores 0.763 against the router's 0.734 (Table 11). | Stated in abstract-level scope and limitations |
| T8 | Abstract wording "bounded correction of 18k parameters" | `main.tex`, `main2.tex` abstracts | "a bounded correction of about 200 parameters on top of the 18k-parameter rule" | Abstract matches Table 17 |
| T9 | Repository path in body | `06_results.tex` §6.4 | Replace `results/exp4_conformal/` with "Appendix B archive" | No file paths in body text |
| T10 | R2-G missing from scorecard | `06d_scorecard.tex`, `fig_timeline.tex` | Add an R2-G row: "descriptive, no refutation rule" (range bands, near-range miss, latency; §7.5). | Figure A.1 and Table 20 list the same items |
| T11 | False precision | All `\Num*` mAP differences | Round mAP and differences to three decimals in prose and tables (keep four in the JSON). One change in `make_paper_data.py` formatting. | No four-decimal mAP values in the PDF |
| T12 | Abstract density | `main.tex`, `main2.tex` | Cut to about six numbers; move the rest to the introduction. Keep under 250 words. | Word count checked by script |

---

## 3. Phase 2: new experiments (CPU, about 2 to 3 days wall-clock)

Order is by impact on acceptability. E1 is the one that decides how the paper is framed, so it runs first.

### E1. Decision-unit control at equal information (R4-1, R4-2): highest impact

**Why.** The per-object arm sees probes, detector features, conflict features *and* frame-level features; the
per-frame arms see only frame-level features. "Per-object beats per-frame" is therefore partly "more input
beats less input." This is the objection most likely to sink the paper.

**What.** A new arm, `frame_pooled`: its router receives, per frame, the pooled statistics of *every* per-object
input the object arm sees (mean, standard deviation, 10th and 90th percentile over the frame's candidates,
plus the frame-level features). Its output is one parameter set per frame, applied to every candidate in that
frame through the same fusion rule. Same information pool, only the decision unit differs.

Secondary arm, `frame_pooled_wide`: same, with hidden width matched to the object arm's parameter count, so the
comparison is not about capacity either.

**How.**
- `src/r4_experiments.py`: build pooled inputs with a group-by on `cand["frame"]` over
  `arbitrate.object_inputs(c, ALL_BLOCKS)`, broadcast back to candidates.
- Register the arm at run time (`ARM_BLOCKS["frame_pooled"] = ...` plus a custom input hook) rather than editing
  `arbitrate.py`. Check first that cache fingerprints hash source files, not runtime state.
- Boreas: train on the registered split, score on the 13 test scenes, scene-cluster bootstrap, 10,000 resamples.
- TruckScenes: same 5-fold CV as R2-A, reuse cached candidates.

**Cost.** Boreas: under 30 minutes. TruckScenes: about 15 minutes per fold, about 1.5 hours (based on the R2-A
rerun timing).

**Outputs.** `results/r4/r4_decision_unit.json`, macros `RfourOne*`, `RfourTwo*`, new table, new forest-plot rows.

### E2. Statistics at proper resolution (R4-E): high impact, low cost

- Rerun the headline paired tests (H2, R2-A1, R2-A3, R2-F1, R3-1, R3-2, and E1) with 10,000 resamples, so
  p-values below 0.001 can be reported instead of sitting at the floor.
- Add effect sizes with intervals to every primary comparison, rounded to three decimals.
- Report seed variance on TruckScenes for the original router and V2 (3 seeds; about 45 minutes per seed with
  cached candidates). This answers "seed variance was only measured on Boreas."

**How.** `src/r4_experiments.py stats`, loading cached models and predictions. Bootstrap count via
`configs/default.yaml` override, not by editing the registered default.

**Cost.** About 2 to 3 hours CPU in total.

### E3. Per-scene appendix (R4-E): medium impact, low cost

- Table and dot plot of per-scene paired differences (object minus per-frame classifier): 13 Boreas test scenes,
  230 adverse TruckScenes scenes (as a histogram), 75 CADC drives.
- Shows whether the gain is broad or carried by a few scenes. Data source:
  `results/supplementary/s1_scene_bootstrap.json` (Boreas) and recomputation from cached TruckScenes predictions.

**Cost.** Under 1 hour.

### E4. Why does the harm-auxiliary loss help? (R4-E): medium impact

It is the only mechanism that helps, yet harm is rare and harm AUROC does not track routing quality. Compare the
full router with `results/models_r2/r2c_no_harm_aux_loss.pkl` on:
- calibration of the fused existence score (expected calibration error, reliability curve);
- where the gain sits (condition, range band, number of sensors present, single-sensor vs multi-sensor
  candidates);
- the learned gains and absence penalties with and without the loss.

Hypothesis to check (exploratory): the auxiliary loss acts as a regulariser on the existence head for
multi-sensor candidates, not as a harm predictor. Report whatever the numbers show.

**Cost.** Analysis only, under 1 hour.

### E5. Conformal and support-check comparison (R4-3): medium impact

- **OOD baselines.** On the same frame-level features: Mahalanobis distance to the clear-weather calibration
  set, k-nearest-neighbour distance (k = 10), and a logistic density-ratio score. Metrics: AUROC for flagging
  severe-condition frames, false-alarm rate on clear frames, miss rate on unflagged frames.
- **Coverage with size.** Report mean radius next to every coverage number, so improved coverage cannot come
  from inflating the region.
- **Calibration-set size.** Coverage vs number of calibration frames (100, 250, 500, 1,000, all), 20 random
  draws each.

**Cost.** Under 2 hours CPU, cached features only.

### E6. Interpretability, measured (R4-E): medium impact

For the original router, on Boreas and TruckScenes:
- distribution of learned evidence gains, absence penalties and position precisions by condition and range;
- how often each sensor carries the largest gain, per condition;
- **counterfactual agreement:** for each multi-sensor candidate, does the sensor with the largest gain match the
  sensor whose removal changes the fused decision most? Report agreement rate;
- for V2: size of the correction by condition, and the fraction of decisions the correction flips.

One figure: original-rule outputs and V2's correction on the same objects, a win and a failure.

**Cost.** Analysis only, about 1 to 2 hours.

### E7. CADC failure diagnosis (R4-E): medium impact

- **Error decomposition** for the object arm vs the per-frame arm on CADC: association failures, missed cars,
  false positives, localisation error. Shows whether the gap starts before or after the fusion rule.
- **Probe shift:** Kolmogorov-Smirnov distance per probe between Boreas and CADC, and whether each probe's
  correlation with detection error keeps its sign.
- **Sensor availability on Boreas:** retrain camera+LiDAR-only and LiDAR+radar-only routers (camera+LiDAR
  already exists as the radar-free retrain), so the radar-dependence statement rests on a sweep rather than one
  arm.

**Cost.** About 1 hour analysis plus about 30 minutes of retraining.

### E8. Stronger-detector sensitivity (R4-4): high impact, needs your decision

Reviewers will ask whether the gain survives modern detectors. Options, in order of strength:

| Option | What | Feasible here? | Strength of evidence |
|---|---|---|---|
| E8-A | Train a real feature-level baseline (L4DR or a BEVFusion-style model) on TruckScenes | **No.** Needs a ≥16 GB GPU for days; the local GPU is unavailable and too small | Strongest |
| **E8-B (recommended)** | **Stronger camera on TruckScenes:** replace the monocular range estimate with LiDAR-projected depth inside the camera box, giving a far more accurate camera 3D position. Rebuild candidates only (the frozen 2D detections are reused), rerun the 5-fold CV for object vs per-frame | Yes, CPU, about 4 to 6 hours (candidate rebuild plus CV) | Medium. Honest caveat: it couples camera and LiDAR, so it is a sensitivity test, not a new detector |
| E8-C | Controlled detector-quality sweep: degrade or sharpen each sensor's detections around ground truth (score noise, position noise) and measure how the per-object gain changes | Yes, about 2 hours | Weaker, synthetic, but shows the trend |

Recommendation: run E8-B as registered R4-4, and E8-C as exploratory. State E8-A as future work with the
reason (compute). If you can get access to an external GPU (university cluster or cloud, about 24 GB), E8-A
becomes possible and would lift the paper the most.

### E9. Simplified-router disclosure (exploratory)

Report, without adopting, the width-40 router and the no-precision router on TruckScenes CV too, so the
"best model is not the reported model" disclosure (T3) covers both datasets. About 1.5 hours.

---

## 4. Phase 3: manuscript rewrite (about 1 day, after Phase 2 results exist)

1. **Contributions, cut to four** (from the external review's suggestion, adjusted to our results):
   1. a controlled, pre-registered test of the fusion decision unit, *at equal information* (E1), including an
      oracle-weather per-frame router;
   2. cross-condition evidence on simulated and real weather, with a measured operating boundary (radar
      dependence, CADC);
   3. a transparent fusion rule plus a bounded learned extension, with the interpretability cost measured (E6);
   4. an empirical audit of uncertainty under weather shift: what is guaranteed, what is only observed, what
      fails (T6, E5).
2. **Related work comparison matrix** (new table): rows are input level, decision unit, detector retraining,
   sensor symmetry, explicit absence evidence, oracle-weather control, number of datasets, conformal evaluation;
   columns are this paper, feature-level weather-robust fusion (RAF, L4DR, 3D-LRF, SAMFusion, RobuRCDet),
   per-frame weather routing (LRC-WeatherNet, weather-conditioned branch routing), candidate-level late fusion
   (CLOCs, ProbEn).
3. **Gap statement**, one paragraph, no "first to" claims: instance-level fusion, reliability weighting and
   weather routing all exist; the effect of the decision unit itself, at equal information, over frozen
   detectors, has not been isolated.
4. **New subsection 7.10, Registration 4**, with E1 to E8 results as scored, and new scorecard rows.
5. **Abstract and highlights** rewritten from the E1 outcome (see risk 1 below), under 250 words, about six
   numbers.
6. **Limitations** updated: detector strength (E8), best-model disclosure (T3), radar sweep (E7), compute
   reason for no feature-level baseline.
7. **Figures:** forest plot gains the R4 rows; new per-scene figure (E3); new interpretability figure (E6).
   All through `src/plots.py`'s current style.

---

## 5. Phase 4: QA, release, submission package (about half a day)

1. `python paper/latex/make_paper_data.py`, compile `main.tex` and `main2.tex` (bibtex cycle), 0 undefined
   references, 0 errors.
2. Render and read every changed page (tables, figures, the Appendix B/References transition), not only the log.
3. `pytest tests` (currently 43 tests; add tests for the pooled-input builder and the new OOD scores).
4. Page count still within 10 to 35, references still at most 50.
5. Rebuild `release/` with `tools/make_release.py`; confirm no manuscript files inside.
6. Zenodo: upload as a **new version** of record 23032756 (you do this; see ground rule 6), then update the DOI
   in the Data availability section and the `[dataset]` reference.
7. GitHub: push the new code and results (only with your go-ahead).
8. Rebuild `submission/InformationFusion/` (main2, title page, highlights, graphical abstract, cover letter
   updated for the R4 results), update the README.

---

## 6. Risks and what we do about them

| # | Risk | Likelihood | Response |
|---|---|---|---|
| 1 | **E1 refutes the decision-unit claim** (the frame-pooled arm matches the object arm) | Real. The ablation already shows frame-level features add nothing on Boreas, which hints the per-object signal matters, but it is not proof | Report it as refuted. Reframe the paper honestly: the gain comes from *object-level degradation information*, and the decision unit itself adds little once that information is pooled. That is still a publishable, cleaner finding, and pre-registering it is what makes it credible. Title and abstract change accordingly |
| 2 | E8-B shrinks the gain | Medium | Report it as the operating boundary: per-object routing helps most when the camera's position is unreliable, which is exactly the motivating case. State it in scope |
| 3 | Compute contention on the shared machine | Medium | All runs are resumable and cached; run overnight; E1 and E2 first since they matter most |
| 4 | Plan grows the paper past 35 pages | Low (currently 23) | Move E3, E6 and E7 detail to a supplementary file; keep summaries in the paper |
| 5 | Reviewers still want a real feature-level baseline | Medium | Matrix + positioning table + compute statement; offer it in the response letter if a GPU becomes available |
| 6 | Zenodo version mismatch with the paper | Low | Cite the new version's DOI; the concept DOI resolves to the latest |

---

## 7. Expected effect on acceptability

| Work | Score effect (out of 10) |
|---|---|
| Phase 1 only (T1 to T12) | 5 to about 6: internally consistent, no obvious errors |
| + E1 (whichever way it falls) and E2 | about 6.5: the main objection answered |
| + E8-B, E5, E6 | about 7: robustness and interpretability measured |
| + E3, E4, E7, E9 and the Phase 3 rewrite | about 7 to 7.5 |
| + E8-A (only with external GPU) | about 7.5 to 8 |

No amount of work guarantees acceptance; this plan removes the reasons a careful reviewer would reject it.

---

## 8. Decisions needed from you before starting

1. Approve Phase 1 (text and table fixes). These change no results; I can start immediately.
2. Approve writing Registration 4 as above (or edit the hypotheses and thresholds first).
3. Approve the Phase 2 runs, and choose for E8: B (recommended), C, or both. Say whether external GPU access for
   E8-A is possible.
4. Decide T2: cite the companion paper, or replace with a neutral sentence (recommended for the blinded file).
5. Confirm that Zenodo is published, so the release goes up as a new version afterwards.

## 9. Order of work and time estimate

| Step | Wall-clock |
|---|---|
| Phase 0: Registration 4 | 2 hours |
| Phase 1: T1 to T12 | 1 day |
| Phase 2: E1, E2 (first), then E8-B, E5, E6, E3, E4, E7, E9 | 2 to 3 days (mostly unattended CPU) |
| Phase 3: rewrite | 1 day |
| Phase 4: QA, release, package | half a day |
| **Total** | **about 5 to 6 working days** |

---

## Status log

**2026-10-05: Phase 1 applied (text and table fixes, no new results).**
- T1 done: Table (feature-group ablation) now prints one convention, full minus ablated; prose rewritten to match.
- T2 done: companion-paper sentence replaced with a neutral citation (zhang2024conformalshift).
- T3 done: narrower router and no-precision variant disclosed as findings, registered model kept.
- T4 done: absence term and conflict features no longer credited as sources of gain (intro, related work, method).
- T5 done: support check labelled exploratory in abstract, introduction, and limitations; "holds or announces" removed.
- T6 done: status table added (guarantee vs observed vs failing), all generated from macros; "guarantee" reworded where exchangeability does not hold.
- T7 done: CLOCs-beats-router-on-CADC stated in limitations (no new numbers typed).
- T8 done: abstract states the correction as about 200 parameters.
- T9 done: repository path removed from body text.
- T10 done: R2-G row added to scorecard.
- T12 done: abstract is 248 words, measured on the compiled blinded PDF.
- T11 done: mAP differences and intervals now three decimals (make_paper_data.py); p-values and rates unchanged.

Compiled: paper/latex/main.pdf 26 pages; submission main.pdf 26 pages; submission main2.pdf 24 pages; 0 undefined references, 0 errors.

Still pending, needs your go-ahead: Phase 2 experiments (E1 to E9), Registration 4 (Phase 0), GitHub push, Zenodo new version.

**Phase 0 drafted:** notes/registration4.md exists as a DRAFT with R4-1 to R4-4 and exploratory items. It is not approved and nothing in it has been run.

**Implementation (no results generated):** src/r4_experiments.py adds the information-equal arm frame_pooled (per-frame pooled statistics of the full per-object input pool plus frame features, identical for every candidate in a frame). It is registered at run time and does not edit arbitrate.py. tests/test_r4_pooled.py passes (3/3) on the synthetic cache. Registration 4 remains a draft; R4-1 has not been run and must not be run until the hypotheses are approved and dated.


**2026-10-06: R4-1 and R4-2 run. Both NOT REFUTED.**
- R4-1 (Boreas): per-object 0.777 vs frame_pooled (equal information, 27,539 params) 0.688. Diff +0.089 mAP,
  scene-cluster p ~ 0.0001 (10,000 resamples, 13 test scenes). The pooled arm, despite having the full per-object
  input pool and MORE parameters than the router, scores about the same as the simple frame_feats arm (0.690,
  Table 2) -- the extra information does not help a per-frame decision.
- R4-2 (TruckScenes, 5-fold CV): per-object 0.822 vs frame_pooled 0.772, out-of-fold. Diff +0.050 mAP, scene-cluster
  p ~ 0.0001 (230 scenes), ahead in 5/5 folds.
- Conclusion: the decision-unit claim is NOT an artefact of unequal information. This is the single most
  important result for the paper's acceptability, since it was the main objection raised in review.
- Deviation logged: the frame-cluster secondary test for R4-1 was planned at 10,000 resamples but that is
  infeasible over ~50k frames in reasonable time; reduced to 1,000 and documented in the JSON. The registered
  primary test (scene-cluster, 10,000 resamples) ran at full resamples as registered.
- Files: results/r4/r4_1_decision_unit.json, results/r4/r4_2_truckscenes_decision_unit.json.

**2026-10-06 continued: R4-1/R4-2/R4-3 fully integrated into the manuscript; R4-4 running.**
- New subsection 7.9 "Registration 4" added to both main.tex and main2.tex, with R4-1, R4-2, R4-3 reported
  as scored. Scorecard now 18 hypotheses (recount caught a stale "14" that was never updated when R2-G was
  added earlier). Appendix A and the pre-registration timeline figure both updated to include Registration 4.
  Forest plot (Figure 8) has two new points under a "R4 -- equal-information per-frame control" group.
- Found and fixed a real corruption bug: an earlier script wrote a literal carriage-return byte into
  `\ref{tab:r2b}`, silently breaking that cross-reference (rendered as "Table eftab:r2b"). Scanned the whole
  source tree for the same pattern; nothing else affected. Fixed and reverified.
- E3 (exploratory, Boreas per-scene): per-object arm ahead in all 13/13 test scenes individually, range
  +0.013 to +0.193 mAP. Added as one sentence in the R4 section. File: results/r4/e3_per_scene_boreas.json.
- main.pdf 27 pages, main2.pdf 25 pages, 0 undefined references, 0 errors (both copies, submission folder in
  sync). Full test suite 46/46 passing as of the last run.
- R4-4 in progress: candidate rebuild with LiDAR-projected camera range (reuses cached 2D boxes and detections,
  no re-detection). Caught and fixed a real bug during smoke-testing: camera_geometry() returns full-resolution
  intrinsics, but boxes2d were detected on a 0.5x-scaled image (ts_cache.py's CAM_SCALE) -- missing that scaling
  silently put ~85% of projected LiDAR points outside their boxes (8-40% override rate instead of the ~52-98%
  the fix produced). Verified on a 10-frame smoke test before committing to the full build.
- Full build started 16:07, running with 8 workers; not yet complete as of this entry. R4-4's 5-fold CV
  (object vs frame_cls, both retrained on the new candidates) is queued to start automatically when the build
  finishes.

**2026-10-07: R4-4 complete. All of Registration 4 (R4-1 through R4-4) is now run, scored, and in the paper.**
- R4-4 (TruckScenes, stronger LiDAR-projected camera): NOT REFUTED. Object 0.779 vs frame_cls 0.714, diff +0.066
  mAP (95% CI [0.053, 0.078], p<0.001, 230 scenes, 5/5 folds ahead) -- essentially unchanged from the original
  camera's +0.064. The decision-unit advantage is not an artefact of a weak monocular camera estimate.
- Session survival problems during this run: the sandboxed environment was torn down twice mid-run (once
  overnight before any candidate file was written, once mid-training after fold 0 only got its harm-baseline
  model cached). `disown` did not help either time, which suggests the whole environment is recycled, not just
  the shell. Recovery worked cleanly both times because of two independent safety nets: (1) the candidate
  rebuild writes one tag file at a time, so nothing but wall-clock time was lost; (2) Ctx.arm()'s existing
  per-fold model caching (keyed by fold tag, already in the codebase for other reasons) meant every restart of
  the training step reloaded already-fitted folds instead of refitting them, so each restart got cheaper. No
  custom checkpointing had to be written to get this resilience.
- A third, separate problem during the run: the final pooled scene-cluster bootstrap (10,000 resamples x 230
  TruckScenes scenes x 4 thresholds, ~1.27M candidates) looked identical to a hang (process alive per `ps -W`,
  but ~0 reported CPU time and ~3MB reported memory in `tasklist` for over 30 minutes) and was killed once as a
  false positive. Added explicit start/elapsed-time logging around this step; the second attempt confirmed it
  was genuinely just slow: 3634s (~61 minutes), not stuck. R4-2 had run the identical computation at the
  identical scale earlier without this being noticed, so this was a pre-existing cost, not something R4-4
  introduced. Logging this reasoning here in case the same shape of result (alive process, near-zero reported
  resource use) recurs for a future large TruckScenes bootstrap -- check elapsed wall-clock against this
  benchmark before concluding it is stuck.
- Manuscript: new R4-4 paragraph in sections/06e_r4.tex, new scorecard row, hypothesis count corrected 18->19,
  Bonferroni threshold updated to 0.05/19, Appendix A registration text extended, timeline figure extended,
  forest plot gains a thirteenth point (R4-4) and a dedicated group. main.pdf 27 pages, main2.pdf 25 pages, both
  copies (main repo + submission folder) compiling 0 errors / 0 undefined. Caught and fixed one more real bug in
  the new 06e_r4.tex prose before this: an override-rate sentence had accidentally cited R4-3's frame-count
  macros instead of R4-4's own LiDAR-override statistics; replaced with a real pooled-override-rate macro
  (75.18%) computed from the candidate-rebuild summary.
- Remaining from the original Phase 2 list: E4 (why the harm-aux loss helps), E5 (conformal calibration-size
  curves, partially overlaps R4-3), E6 (interpretability measurements), E7 (CADC error decomposition), E8-C
  (controlled detector-quality sweep), E9 (disclose the width-40/no-precision routers on TruckScenes CV too).
  None of these touch raw sensor data, so none should have R4-4's long-running-process risk.

## 2026-10-09: E10 minimal per-object baselines (review item 1; exploratory, NOT registered, not in the scorecard)
Reason: an independent strict review argued the per-object gain might be mostly range-conditioned fusion
(range + detector scores), not degradation-aware arbitration. Code: src/e10_minimal.py (run:
`python -m src.e10_minimal boreas 10000`, `python -m src.e10_minimal truckscenes 1000`). Three per-object routers, same
rule/loss/optimiser/stopping/dropout as every arm, differing only in input blocks: geom (range, azimuth), geom_conf
(+ detector logits/presence/fov/class), geom_clocs (+ detector and conflict features). No degradation probe, no
frame-level statistic. Existing frame_cls/object models load from cache (TruckScenes uses the ts_fold{i} dirs).
Results (adverse-pooled Car mAP, delta vs per-frame classifier, scenes ahead):
- Boreas (13 test scenes): frame_cls 0.700, geom 0.733 (+0.032, 12/13), geom_conf 0.746 (+0.046, 13/13), geom_clocs
  0.769 (+0.069, 13/13), full 0.777 (+0.077, 13/13). Full minus geom_clocs +0.008 [0.001, 0.015], ahead 10/13,
  exact sign test p=0.092 -> on simulated weather ~90% of the gain needs no degradation probe.
- TruckScenes (230 scenes, 5-fold OOF, 1,000 resamples): frame_cls 0.763, geom 0.776 (+0.014), geom_conf 0.779
  (+0.017), geom_clocs 0.796 (+0.033), full 0.827 (+0.064). Full minus geom_clocs +0.031 [0.026, 0.037], ahead
  186/225 scenes, all 5 folds, sign p<1e-23 -> on real weather ~half the gain depends on the probes.
Interpretation written into the paper (sec:e10): the decision-unit claim is unaffected (every per-object arm beats
the per-frame classifier on both datasets); degradation awareness is claimed for real weather only. Untested
hypothesis stated as such: the simulator's attenuation/airlight are functions of range.
Manuscript: new sections/06f_e10.tex + Table e10, macros via make_paper_data.py e10_macros (EtenRows etc.), abstract,
intro (What we find, contribution 1), limitations (What we do not claim), conclusion (four findings), exact sign
test added to the Boreas scene-robustness paragraph, stale "Registration 1, 2 and 3" in the intro fixed to 1 to 4.
Also fixed: corrupted "\ref" (carriage return) in sections/06b_results_r2.tex line 226 (both copies).
main.pdf 29 pages, main2.pdf 26 pages, 0 undefined, 0 errors, both copies in sync.
Timing note: TruckScenes fit ~8 min/fold for 3 arms, 7 bootstraps at n=1000 took ~30 min (n=10000 would be ~5 h).
Remaining review items not yet done: #2 strong in-domain detector on TruckScenes (E8-C), #3 restructure/shorten,
#4 stronger per-frame learner, #7 rescope real fog/snow claims (12 fog/23 snow scenes), #8 conformal at matched radius,
#9 sim-to-real transfer test (train Boreas-sim, test TruckScenes), #10 title/highlights tone.

## 2026-10-09 (later): E11 per-condition TruckScenes, E12 matched-radius conformal (both exploratory, not registered)
- E11 (src/e11_percond.py, Table e11 in 06b): per-object minus per-frame classifier, OOF, per real condition, 1,000
  resamples + exact sign test. fog +0.075 (12/12 scenes), snow +0.053 (23/23), rain +0.072 (98/119), night +0.052
  (68/76), clear +0.066. Answers review item 7: the advantage holds in each condition incl. fog and snow, but fog/snow
  rows rest on 12/23 scenes (stated). Cached fold models only, ~7 min/fold.
- E12 (src/e12_matched_radius.py, paragraph "What the repair is, and is not" in 06b): vanilla radius rescaled to the
  degradation-normalised mean radius reaches the SAME coverage: snow 0.910 (x1.46) vs 0.918, fog 0.910 (x1.19) vs
  0.911, rain 0.910 vs 0.909. So the repair is label-free adaptive inflation, not a tighter radius. Abstract, intro
  contribution 5 and limitations reworded accordingly. Abstract trimmed to 243 words (limit 250): removed the support-flag
  sentence (still in body/limitations/conclusion).
- INCIDENT: an inline python -c edit of 06b_results_r2.tex duplicated everything after the table (file doubled to 500
  lines, 32 pages). Caught because the page count jumped 29->32. Fixed by truncating the duplicate (verified identical
  apart from trailing CRs) and normalising to CRLF. LESSON: do multi-line file surgery with a script file, assert
  invariants, and treat an unexpected page-count change as a bug signal.
- Page counts: main 29, main2 (blinded) 27, 0 undefined, both copies in sync.
Still open from the review: #2 strong in-domain detector (needs detector training, multi-day), #3 restructure/shorten,
#9 Boreas-sim -> TruckScenes transfer, #10 title tone. #4 is covered by R4 (frame_pooled has more capacity, full inputs).

## 2026-10-09 (evening): E13 zero-shot transfer Boreas-sim -> real TruckScenes (exploratory, not registered)
src/e13_sim2real.py. Cached Boreas arms applied unchanged (Boreas standardiser/prior/weather classifier) to all 230 real
adverse TruckScenes scenes. RESULT: FAILS. object 0.567 vs frame_cls 0.750 vs fixed 0.745; object behind frame_cls by
-0.183 [-0.196,-0.170], behind in 212/230 scenes. Minimal arms degrade less (geom -0.009, geom+clocs -0.042).
Diagnostic (pipeline OK: shapes match, geometry z=0.04): radar frame stats f_r_log_hits/f_r_hit_pow/f_r_noise sit ~7 sd
from Boreas scale (sd x2-4), radar probe power ~1.9 sd. Two label-free repairs of those five features made it WORSE
(recentre on target stats 0.475; neutralise 0.455), so the cause is NOT established; stopped there on purpose (more
variants chosen after seeing results = forking paths). Written into sec:e10 (retitled "Exploratory controls: degradation
probes and the simulator") + limitations "What we do not claim". Framing: real-weather evidence never used the simulator
(R2-A trains on real TruckScenes); a router is per-dataset, consistent with CADC zero-shot failure.
Page counts now: main 30, main2 27. Abstract 243 words. Still open: strong in-domain detector, shorten/restructure,
title tone. Gotcha: standardising by a near-constant feature's own sd (r_bg_pow sd x0.0) blows z-scores up.

## 2026-10-12: consistency pass after the exploratory runs
- cover_letter.md rewritten where it had gone stale/overclaiming: now mentions Registration 4 and the equal-information
  control; no longer says the support flag identifies untrustworthy frames (it does not on real snow); states the
  conformal repair is label-free adaptation, not a tighter radius; reports the narrowing controls (probes ~90% of the
  simulated gain needed none, ~half on real weather; zero-shot sim->real transfer fails).
- submission README.md corrected: 17 section files, four registrations, page counts 30 (main) / 27 (main2), author bios
  REMOVED (it said they were included), graphical abstract note (kept old single-row layout on purpose: guide wants a
  wide graphic), differences-from-Zenodo list updated.
- Intro: contribution 6 said "three dated pre-registrations" -> four; new Roadmap paragraph (the paper is in evidence
  order because each registration reacts to an earlier result). Scorecard caption: Registrations 2 to 4.
- A2_repro commands now list r3_experiments, r4_experiments {r4_1,r4_2,r4_3,r4_4,e3}, e10..e13 (added a __main__ to
  src/r4_experiments.py, which had no CLI). tests/test_e10_sign.py added (exact sign test). 46 + 4 tests pass.
- main 30 pp, main2 27 pp, abstract 243 words, 0 undefined, all three copies in sync.
Still open: strong in-domain detector (days of training), real restructure/shortening, title tone.

## 2026-10-13: E8-C in-domain LiDAR detector (review item 2) -- IN PROGRESS
Protocol (fixed before any router result): notes/e8c_indomain_detector.md. Code: src/ts_detector.py (cross-fitted PointPillars
Car fine-tune), src/e8c_rebuild.py (swap LiDAR Car detections into the cache, recompute probes), src/e8c_router.py (5-fold
router CV, same rule as R4-2/R4-4), tools/run_e8c_train.sh (trains fold 0 then fold 1).
Findings so far:
- Probe-equivalence check passed: recomputing LiDAR probes for the ORIGINAL boxes reproduces the cache (12/12 frames, max
  abs diff 7.6e-6).
- Sanity run (200 its, 400 frames): held-out Car AP 0.722 (KITTI) -> 0.807 mean over 0.5/1/2/4 m; 0.5 m: 0.60 -> 0.72.
- PERFORMANCE BUG found and fixed: dense frames (3 merged LiDARs) hit the 40k-pillar cap and the pillar feature net needs
  several GB of activations -> 7.6 s/it thrash on the 5 GB P2000. Fix: gradient-checkpointed pillar encoder in chunks
  (encode_pillars) -> ~1.5 s/it at batch 2, peak ~4 GB. First iteration also spends ~33 s in cudnn autotuning.
Full training launched 2026-10-13 11:16: 4 epochs x 3,000 random frames per fold, batch 2, lr 2e-4 (one-cycle), ~2.8 h per fold.
RESUME if interrupted: `bash tools/run_e8c_train.sh` (skips finished folds, resumes from fold{k}_last.pt each epoch).
THEN: (1) `python -m src.ts_detector eval --fold 0` and `--fold 1` (stride 10) -> eval_fold{k}.json; (2)
`python -m src.e8c_rebuild build --workers 5` (~1-2 h) then `... candidates`; (3) `python -m src.e8c_router` (primary
bootstrap 10,000 ~1 h); (4) make_paper_data (e8c_macros exists), write the paper subsection, sync, recompile.

## 2026-10-14: E8-C detectors trained; detection/rebuild/router pipeline running
- Both detectors finished (fold 0: 4 epochs; fold 1: 4 epochs). The slowdown on 2026-10-13 (epoch 2 at 6.6 s/it, epoch 3 at ~11 s/it,
  fold 0 took 11.2 h instead of 2.5 h) was GPU contention with the user's AI1 job (train_autobot.py --device cuda) on the
  shared 5 GB P2000. Not touched. Fold 1 then ran at ~1.6 h/epoch.
- Inference on the contended GPU: 2.7 s/frame (projected 16-20 h). CPU inference: 0.9 s/frame at 6 threads, so detection now runs as
  6 CPU processes x 4 threads at 0.56 s/frame (ETA ~3.6 h from 06:59). Lesson: on this shared machine, CPU beat the GPU.
- Added: centre-distance pre-filter ahead of the rotated NMS (cars >= 3.9 x 1.6 m always overlap within 1.2 m); voxel-level entry
  point in InDomainCarDetector; e8c_rebuild now has detect (CPU or GPU) / build (CPU) / candidates; tools/run_e8c_pipeline.sh.
- Queued watcher: when 23,902 files exist in results/ts_detector/dets it runs tools/run_e8c_pipeline.sh
  (build -> candidates -> e8c_router 10000 1000). Logs: logs/e8c_detect.log, e8c_build.log, e8c_candidates.log, e8c_router.log
  (ends with PIPELINE_DONE). If the session dies: re-run `bash tools/run_e8c_pipeline.sh` (every phase skips finished work).
- Detector quality eval (held-out folds, stride 20, CPU): `python -m src.ts_detector eval --fold {0,1} --stride 20` ->
  results/ts_detector/eval_fold{k}.json.
- Mistake caught: a queued waiter used pgrep, which does not exist here, so the pipeline started immediately and launched a duplicate
  detection run. Killed the duplicate. Use a file-count condition for waiters.

## 2026-10-14 (afternoon): machine reboot at 12:43
All background jobs were killed. On disk: 23,902/23,902 detections, both detector weights, eval_fold{0,1}.json (detector AP 0.704 KITTI ->
0.868 in-domain on 1,196 held-out frames), 16,257/23,902 rebuilt cache frames (no .tmp leftovers). Resumed with
`bash tools/run_e8c_pipeline.sh` at 12:51 (build had 7,645 frames left, ~0.9 h; then candidates; then src.e8c_router). Re-armed the
PIPELINE_DONE watcher. Nothing was lost beyond wall-clock time: every phase writes one file at a time and skips finished work.

## 2026-10-14 (evening): full-paper review fixes
Verified errors fixed (06_results.tex, main.tex abstract, 08_conclusion.tex, 01_introduction.tex, A2_repro.tex; main2 abstract too):
- "per-frame routing statistically indistinguishable from not routing" contradicted Delta=0.012, p=0.001 -> now "a small but significant gain (about one sixth of the per-object gain)".
- "negative-evidence term at test time costs 0.771 mAP" -> "lowers adverse mAP from 0.777 to 0.771".
- "beats five published baselines" -> "four of five (ties the set-transformer on simulated, trails it on real)"; CLOCs beats it on CADC (already in text).
- Conclusion no longer says the flag is a practical safeguard (it fails on real snow); contribution 5 retitled.
- Repro appendix: four pre-registrations, 10,000 resamples for R4 tests. Abstract 246 words (cap 250).
Open: Zenodo/GitHub publish (needs user token + go-ahead), E8-C router result (running: logs/e8c_router.log, ~25 min/fold + 10,000-resample bootstrap),
then Sec. 3 wording ("nothing retrained anywhere") and Sec. 7.10 text for E8-C.

## 2026-10-14 16:14: E8-C folds done -- PRE-STATED RULE SAYS NOT SUPPORTED
Out-of-fold adverse mAP per fold (object vs frame_cls): f0 0.892 vs 0.894, f1 0.892 vs 0.893, f2 0.854 vs 0.855,
f3 0.856 vs 0.854, f4 0.905 vs 0.901 -> per-object ahead in 2/5 folds (rule needs >= 4/5). Fixed weights, frame-pooled and
no-probe arms all within ~0.003 of each other. With an in-domain LiDAR detector (held-out AP 0.868) every late-fusion arm
converges; the per-object advantage (+0.064 with the KITTI detector) essentially vanishes. Bootstraps still running.
Per notes/e8c_indomain_detector.md: "If it vanishes, the claim is stated as holding only for a weak or mismatched LiDAR detector."
This must be integrated into the abstract, intro, contributions, limitations and conclusion before submission.

## 2026-10-14 (night): E8-C result integrated; paper reframed for Information Fusion
E8-C final (src/e8c_router.py, 10,000 resamples): pooled per-object minus per-frame = -0.0004, 95% CI [-0.003, +0.003], p=0.49, ahead on 120/230 scenes
(behind 104), ahead in 2/5 folds -> NOT SUPPORTED by the rule fixed in notes/e8c_indomain_detector.md. All five arms within 0.003 mAP
(0.876-0.879). Detector held-out Car AP 0.704 (KITTI) -> 0.868 (cross-fitted in-domain).
Paper changes: new title "When Does Per-Object Arbitration Help? Late Fusion over Frozen Camera, LiDAR and Radar Detectors in Simulated and
Real Adverse Weather"; new abstract (238 words), highlights, intro (find/contributions/roadmap), new Sec. 7.11 (sections/06g_e8c.tex, Table e8c),
limitations (frozen, mismatched detectors scope), conclusion (five bounding findings), Sec. 3/5 detector sentences, repro appendix (GPU only for the
E8-C detector), cover letter (conditional main claim), highlights.docx (edited XML), title_page, README, refs.bib/CITATION.cff/zenodo_draft.py titles.
Release archive rebuilt (182 files, 45 MB; includes r4, truckscenes/r4, ts_detector results and fold*_final.pt; no .tex, no manuscript pdf).
NOT DONE (needs the user): publish Zenodo draft 23032756 (new token + explicit go-ahead) and re-push GitHub; submit via Editorial Manager.

## 2026-10-16: IEEE T-IV version
submission/IEEE_TIV/main_tiv.tex: condensed IEEEtran (journal, double column) manuscript, 8 pages (Regular paper suggested 10), abstract 201 words, 4 keywords, names-only
author line with affiliations in the first-page footnote, three biographies restored from the 2026-10-05 session (author-supplied text), all numbers from macros.
New row macros EtenRowsShort / EelevenRowsShort. Correction: the exact in-domain pooled difference is -0.00004 (not -0.0004 as noted above); the macro now prints 5 decimals.
Elsevier version unchanged (31 / 28 pp). Portal items for the user: COI, institutional emails, OA choice; publish Zenodo first. See submission/IEEE_TIV/README_TIV.md.

## 2026-10-16 (later): IEEE T-IV version expanded to 10 pages
Added: 4 generated figures (src/tiv_figures.py: effect-size summary across all settings, accuracy by simulated condition, gain build-up by input group, coverage by
condition; all colour + marker + line style, CVD-safe), the cropped weather-simulation image, and equations (harm definition, decision-unit eqs 4-5, conformal
quantile and coverage eq 6, conformal risk control eq 7, mAP eq 8, bootstrap p-value eq 9), a five-question results guide, a practitioner-implications paragraph.
Fixed: long equations overflowing the column, two overfull tables, biographies split across pages. Final: 10 pages, 0 errors, 0 overfull, 0 unresolved refs.

## 2026-10-16 (night): IEEE version references 37 -> 50
13 references added, each confirmed from its arXiv/proceedings page (authors, venue, year): MOS (ICLR 2025), DPO (ACM MM 2024), Durasov evidential 3D detection (arXiv 2024), WeatherOcc3D
(arXiv 2026), radar-vision fusion survey (arXiv 2024), Kumar dual-threshold conformal (COINS 2025), Timans two-step conformal (ECCV 2024), LiRaFusion (ICRA 2024), TripleMixer (arXiv 2024),
K-Radar (NeurIPS D&B 2022), Robo3D (ICCV 2023), Dreissig LiDAR-adverse-weather survey (IV 2023), Stutts conformal feature fusion (arXiv 2023). Venue names normalised to IEEE abbreviations.
Result: 50 cited, 25 from 2024-2026, 35 from 2021+. Still 10 pages. verify_tiv.py (scratchpad) checks abstract/keywords/ASCII/captions/pages: all PASS.

## 2026-10-19: decision - IEEE T-IV is the only target; Zenodo to be recreated
User: submit to IEEE T-IV only; old Zenodo draft (23032756) will be deleted by the user; a NEW token will be provided.
Procedure once the token arrives (token only via env var ZENODO_TOKEN, never written to disk):
 1. `ZENODO_TOKEN=... python tools/zenodo_draft.py create`            -> new draft id + reserved DOI
 2. `python tools/set_doi.py 10.5281/zenodo.<NEW>`                      -> replaces the DOI in 7 files (CITATION.cff, 3 refs.bib, 2 main.tex, title_page.tex)
 3. `python tools/zenodo_draft.py meta <id>` ; `python tools/make_release.py` ; `python tools/zenodo_draft.py upload <id>`
 4. recompile submission/IEEE_TIV (bibtex cycle) and the Elsevier copies; show the user the file list
 5. publish ONLY on the user's explicit go-ahead: `python tools/zenodo_draft.py publish <id>` (permanent)
 6. GitHub: gh is logged in as hwaqas160; repo hwaqas160/paper02-conflict-fusion; push the curated release set (no .tex, no manuscript pdf).
Dry-run of set_doi.py verified (7 occurrences).
