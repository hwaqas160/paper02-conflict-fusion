# S4 CADC evaluation bug — found and fixed (2026-09-22)

`src.supplementary.s4()` retrains every routing arm with radar removed from the Boreas data
(`remove_modality`), then evaluates those radar-free-trained arms on CADC (which has no radar). First run
gave the per-object arm **0.258 mAP** on CADC — a catastrophic collapse next to the fixed baseline's 0.740
(same radar-free training) and next to the *original*, with-radar-trained-then-masked-at-test-time arm's
0.740 (Exp 5).

## Diagnosis

Traced by comparing the standardized input feature vector's mean z-score (relative to the radar-free arm's
own training statistics) across all 66 router inputs on CADC. One feature dominated by three orders of
magnitude: `fov_rad` (radar field-of-view flag), z ≈ **+1000**.

`remove_modality()` zeroes `present` *and* `fov` for radar on every Boreas training/test candidate — the
router is trained believing radar is **structurally absent** (never worth an absence penalty). Its
standardizer therefore fits `fov_rad` as an (almost) zero-variance column (mean 0.000, sd 0.001).

CADC's own candidate builder (`src/candidates.py`, dataset-agnostic) sets `fov` from geometry alone — whether
a position lies within a sensor's field of view — with no notion that a whole sensor type is absent from the
dataset. Every CADC candidate in the ROI gets `fov_rad = 1` (a radar, if this dataset had one, could
plausibly have seen this position). Feeding that un-zeroed `1.0` through a standardizer whose radar-fov
column has sd ≈ 0.001 produces a z-score in the hundreds for every single candidate, saturating the MLP and
making its output close to a constant, near-arbitrary logit — the systematic under-confidence (recall 6,342
vs 15,933 TP-eligible detections at matched τ, only 111 false positives) that produced the 0.258 mAP.

**Causal confirmation:** forcing `T["cand"]["fov"][:, M_RAD] = 0` before prediction (matching the training
convention) recovers **0.728 mAP** — back in the range of every other arm and of Exp 5's original transfer
number (0.740). A partial test (setting `l_ring_ratio`, CADC's other visibly degenerate feature — see below —
to a Boreas-typical value) moved mAP by < 0.02, ruling it out as the primary cause.

## Also noted, not the cause, worth recording

CADC's LiDAR loader (`cadc_cache.py`) appends two zero columns for ring and timestamp
("ring/time unavailable"), so `l_ring_ratio` (fraction of unique LiDAR rings hit) is **pinned at exactly
1/32 for every CADC candidate** (32 = the `n_rings` passed for the VLP-32C) — a real, structural data
limitation, but not the driver of this particular collapse (see causal test above). Any future work that
leans harder on this feature for a CADC comparison should be aware it currently carries no information.

## Fix applied

`s4()` now zeroes `T["cand"]["fov"][:, M_RAD]` on the merged CADC table once, right after loading, before any
radar-free-trained arm predicts on it — matching `remove_modality()`'s Boreas convention. Rerun after the fix:
0.728 mAP for the per-object arm (see `results/supplementary/s4_noradar.json`, `S4.md`).

## Confirmed end-to-end (2026-09-22, frozen 1,654-frame run, real pipeline not just the diagnostic script)

| arm | CADC mAP before fix | CADC mAP after fix |
|---|---|---|
| Confidence-only (also reads raw `fov`) | 0.667 | 0.735 |
| Per-object | 0.258 | 0.728 (95% CI 0.720–0.735) |

Corrected drive-cluster bootstrap (75 drives): per-object − per-frame-classifier Δ = −0.0113, one-sided
p = 0.991 — i.e. **not established** that per-object beats per-frame on CADC, cleanly matching the
manuscript's hedged pre-fix language ("the advantage... does not carry over to a real dataset that never had
radar"). Confidence-only was mildly affected by the same bug (it also reads raw `fov` in its input block),
now corrected too.

## Lesson for the paper

The corrected number changes the finding materially. The earlier draft's framing ("the per-object method
collapses without radar, even on Boreas' own radar-free retraining") is **not supported** once the evaluation
bug is fixed — the radar-free-retrained arm transfers to CADC about as well as every other arm does
(≈ 0.73–0.74), and continues to win comfortably against per-frame routing *on Boreas itself* even with radar
entirely removed from training. The real, robust finding is narrower and still true: the method's *margin*
over per-frame routing shrinks when radar is removed (Exp 7 test-time dropout; S4 Boreas retraining), and
the original with-radar-trained arm's CADC transfer (Exp 5, 0.740, statistically indistinguishable from the
fixed baseline) shows no advantage over per-frame routing on a radar-free real dataset. Both of those
conclusions survive; "catastrophic collapse when retrained without radar" does not, and must not appear in
the manuscript.

## A second staleness bug caught the same day, different mechanism (2026-09-22, later)

`full_run_chain.ps1`'s multiseed step (`run_experiments --seeds 0 1 2 3 4`) writes each seed's models to
`results/seed{s}/models/`, a directory the chain never cleared. Stale `results/seed0..4/models/*.pkl` from
an earlier dev-data (1,654-frame) 5-seed test run (2026-09-21) were still on disk. `Ctx._load_or_fit` only
checks whether a `.pkl` file exists — it has no notion of which candidate data it was fitted on — so seed 0
of the full-data multiseed run silently loaded the year-old dev-trained model and would have evaluated it
against full-data candidates, producing numbers attributed to "full data, seed 0" that were actually stale.

Caught within ~4 minutes (the giveaway: log said `loaded from fixed_clear.pkl` instead of `params=... epochs=...`,
immediately after `--retrain` had *just* finished retraining everything else fresh — an oddly fast "fit" for
a run that should take minutes). Killed the process, deleted `results/seed0..4/`, relaunched. Confirmed fixed:
log now shows real training (`params=19 epochs=15 best_val=0.4079`) instead of a load.

**General lesson, now applied everywhere in this project's chained scripts**: any script that reuses
`Ctx`/`_load_or_fit`'s pickle-cache-by-filename convention across a chain of runs on different data must
explicit delete every model directory it will touch before starting, not just the ones inside `results/models/`
that `--retrain` covers. `results/models_noradar/`, `results/seed*/models/`, and any future `derived_ctx`
tag need the same treatment. Checklist added to `full_run_chain.ps1`'s header comment.
