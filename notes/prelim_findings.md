# PRELIMINARY findings — 2026-09-16 (NOT the final result)

Run: `results_prelim/` · 347 frames × 4 fully-cached conditions (clear0, fog1, rain1, snow2) · stride-3 dev subset
of the phase-A download · test = 13 scenes. **Superseded** by the frozen 1,654-frame × 11-condition run in
progress. Recorded because it triggers the PLAN §9 week-4 gate.

## H1 gate: FAILS the pre-registered threshold, on the knife edge

| quantity | value |
|---|---|
| pooled adverse decision-flip harm | **1.995 %** (95 % CI 1.78–2.21) |
| pre-registered refutation rule | harm < 2.0 % ⇒ refuted |
| clear-weather harm | 1.9 % |
| **weather-induced excess** | **≈ 0.1 pp** |
| AP-level harm cells (sensor removal improves mAP) | **0 of 12** |
| help rate (adverse) | 4.8 % (camera 3.1 %, lidar 2.8 %, radar 0.7 % harm vs help ≈ 2.4× higher) |

Interpretation — three things are true at once:

1. **Harm is real but rare and not weather-driven** at these severities. The mild/moderate conditions cached so far
   (fog1 = 150 m visibility, rain1 = 5 mm/h, snow2 = moderate) barely move it. The severe conditions that should
   produce harm (fog3 = 35 m, snow3, rain3, night2) were *not* in this run.
2. **Removing a sensor never helped in aggregate.** Every modality's help rate exceeds its harm rate, so the
   "adding a camera degrades performance" phenomenon RAF reports does not reproduce here at the AP level. A
   *properly fitted* fixed late fusion is already fairly robust: it learns a low camera evidence gain (0.26) and a
   strong absence penalty. Harm was 3× larger before the priors were fitted (amendment A7) — i.e. much of the harm
   reported by a naive fusion is an artefact of a weak baseline, which is itself a finding worth stating.
3. **The method still wins, for a different reason than the plan assumed.** Per-object routing beats the per-frame
   weather classifier by **+0.056 mAP** (0.818 vs 0.763, p ≈ 0.001; +0.063 on held-out snow). The gain comes from
   better per-object weighting and localisation, not from avoiding rare decision flips.

## H2: supported (preliminary)

| arm | clear0 | fog1 | rain1 | snow2 (held out) | adverse pooled |
|---|---|---|---|---|---|
| fixed (clear-fit, status quo) | 0.816 | 0.730 | 0.801 | 0.753 | 0.761 |
| per-frame weather classifier | 0.818 | 0.738 | 0.800 | 0.751 | 0.763 |
| per-frame oracle weather | 0.818 | 0.739 | 0.799 | 0.749 | 0.762 |
| per-frame features | 0.814 | 0.730 | 0.801 | 0.750 | 0.760 |
| confidence-only | 0.824 | 0.751 | 0.808 | 0.765 | 0.774 |
| **per-object (ours)** | **0.857** | **0.790** | **0.850** | **0.815** | **0.818** |

Per-frame routing is worth ~nothing over fixed weights (+0.002); per-object is worth +0.057. That is the paper's
central comparison and it survives on real data. **But** harm-prediction AUROC is higher for confidence-only
(0.968) than for the per-object head (0.839) — scores alone predict *flips* well; they just do not route as well.
Report both.

## H3: not refuted (preliminary, mild conditions only)

Clear-calibrated coverage at nominal 0.90 was 0.973 under snow2 (over-covers, i.e. conservative) and frame FNR
0.087 ≤ α = 0.10. No collapse, so conflict inflation had nothing to repair here (it shrank clear radii by 11 %).
The interesting test is fog3 / snow3 / night2, which are still caching.

## What this changes

* Do **not** write the paper around "adding a modality often hurts". At these severities it does not, once the
  baseline is fitted properly.
* The defensible framing is: *per-object degradation-conditioned arbitration improves fused detection under
  weather, while per-frame weather routing does not* — plus the honest negative result on harm.
* Decide the final framing only after the 1,654-frame × 11-condition run (includes the severe conditions).
