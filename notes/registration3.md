# Registration 3 -- closing the gap to the set-transformer (written 2026-09-28, BEFORE any of these experiments)

**State of knowledge when written.** TruckScenes 5-fold CV (out-of-fold, adverse pooled Car mAP, means over folds):
per-object router 0.827; black-box set-transformer 0.842 (ahead in 4/5 folds); oracle 0.871; per-frame classifier 0.762.
On Boreas the router and the set-transformer tie (0.777 vs 0.777, p_Holm 0.39). No variant below has been run.
This registration is a *reaction to an unfavourable result*; it is reported as such, next to -- never instead of --
the original router's numbers (R2-A, R2-B). The original router remains "the router" in all earlier sections.

## Why not just tune until it works
Every choice below is fixed now; selection uses only **development** data; the final numbers come from scenes no
development step ever touched.

## Data roles
* **Development (TruckScenes official *train* scenes only, 523):** 20 % of them (stratified by weather tag, seed 0)
  form **dev-test**; the rest are dev-fit (internally split 80/20 into fit/early-stopping-validation exactly as before).
  The 75 official *val* scenes are excluded from development entirely.
* **Final evaluation (once, after the variant is frozen):** (i) the 75 official TruckScenes val scenes;
  (ii) Boreas test scenes (never used in any TruckScenes development step), retrained from Boreas train scenes with the
  frozen variant and unchanged hyperparameters. Also reported, flagged as **not independent** of development because
  the folds contain development scenes: TruckScenes 5-fold CV.

## Diagnosis (fixed before running; descriptive)
Trained on dev-fit, scored on dev-test, all conditions and adverse pooled Car mAP, paired scene bootstrap:
* D0 original router; D1 set-transformer; D2 set-transformer **without** cross-object attention (per-object MLP, same free
  score and position heads). If D2 ~ D1 the edge comes from the unconstrained heads; if D2 ~ D0 it comes from attention.

## Candidate variants (the complete list; no others will be tried)
* **V1 attention context:** frame context by 2-layer, 4-head attention over the frame's candidates instead of mean-pooling;
  the fusion rule is untouched (weights, gains and absence terms still set by the router).
* **V2 bounded correction:** V0 + a bounded residual on the fused logit (2 tanh(r)) and on the fused position
  (2 tanh(o) metres, per axis). The rule stays; the correction is reported and ablated.
* **V3 = V1 + V2.**
Capacity is matched: reported parameter counts; no width/depth search.

## Selection rule
Choose the **simplest** variant (V1 < V2 < V3) whose dev-test all-condition mAP is within 0.005 of the set-transformer's
or, failing that, the variant with the highest dev-test mAP **provided** it improves on V0 with scene-bootstrap p < 0.05.
If none does, we report that the gap could not be closed by these means and keep the original router.

## Hypotheses (scored on the final evaluation)
* **R3-1:** the frozen variant's TruckScenes-val adverse Car mAP is not lower than the set-transformer's by more than
  0.01 (non-inferiority margin, one-sided scene-cluster bootstrap, alpha 0.05). *Refuted if* the lower bound of the
  paired difference (variant - set-transformer) is below -0.01.
* **R3-2:** on Boreas test adverse scenes the frozen variant is not lower than V0 (no regression): difference >= -0.005.
* **R3-3 (cost of the fix):** parameter count and CPU latency reported; interpretability claims restricted to what the
  variant still exposes (gains/weights vs. bounded correction magnitude).

**Rule:** all outcomes go into the paper as Registration 3, including "the gap was not closed".

---
## Development outcome and FROZEN variant (appended 2026-09-28, before any final-evaluation number was computed)
Dev-test = 104 held-out official-train scenes (`results/truckscenes_dev/r3/r3_dev.json`), Car mAP:

| model | params | all | adverse | clear |
|---|---|---|---|---|
| original router | 18,195 | 0.812 | 0.796 | 0.825 |
| V0 (same router, new code path) | 18,195 | 0.811 | 0.799 | 0.822 |
| V1 attention context | 85,139 | 0.811 | 0.799 | 0.822 |
| **V2 bounded correction** | **18,390** | 0.825 | 0.813 | 0.837 |
| V3 = V1 + V2 | 85,334 | 0.832 | 0.820 | 0.843 |
| set-transformer | 72,387 | 0.818 | 0.808 | 0.828 |
| set-transformer, attention removed (D2) | 38,851 | 0.820 | 0.807 | 0.832 |

* **Diagnosis:** D2 ~ D1 (0.820 vs 0.818): cross-object attention is *not* the source of the set-transformer's edge; the
  unconstrained score/position heads are. V1 confirms it (attention context alone: no gain over V0).
* **Selection rule -> V2** (simplest variant within 0.005 of the set-transformer: 0.8254 vs 0.8179). V3 scores higher but
  is a 4.6x larger model and the rule prefers the simplest. **V2 is frozen: no further change of any kind.**
* Note (reported, not a deviation): V1/V3 carry 66k extra attention parameters, so the earlier "quarter the size of the
  set-transformer" statement applies to V0/V2 only.
* Final evaluation now runs exactly as registered above: TruckScenes official val (75 scenes), Boreas test, plus
  descriptive extras (rule-only ablation of the correction, correction magnitudes, CPU latency, 5-fold CV flagged as
  not independent).
