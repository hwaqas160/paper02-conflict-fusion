# Paper 01 ↔ Paper 02 conformal cross-check (2026-09-21)

PLAN amendment A5 assumed Paper 01's `conformal.py` was unavailable. It is at
`F:\CLAUDE\AI1\paper01-coverage-transfer\src\conformal.py`. `tests/test_core.py::test_quantile_agrees_with_paper01`
now loads it (skips if absent) and pins the relationship between the two quantiles.

## Finding: Paper 01's split-conformal quantile is one rank too high

The split-conformal quantile is the k-th smallest calibration score, k = ⌈(n+1)(1−α)⌉ (Vovk et al. 2005; Lei et al. 2018).

| | implementation | returns |
|---|---|---|
| Paper 02 `conformal_quantile` | `sort(s)[k-1]`, ∞ if k > n | k-th order statistic ✔ |
| Paper 01 `split_conformal_quantile` | `np.quantile(s, k/n, method="higher")` | (k+1)-th order statistic (numpy's "higher" interpolates on position (n−1)·p) |

Example: n = 20, α = 0.1 → k = 19. Paper 01 returns the maximum (20th), not the 19th.

**Effect:** conservative. Coverage guarantee still holds; the interval is one order statistic wider, i.e.
over-coverage of at most ~1/n. Negligible for n in the thousands, visible for small groups (Paper 01's
`GroupConditionalConformal` uses groups down to n = 50 → up to 2 pp over-coverage).

**Fix for Paper 01** (not applied — separate project, its jobs were running): replace the unweighted branch with
`k = ceil((n+1)(1-alpha)); return inf if k > n else sort(s)[k-1]` (or `np.quantile(s, k/n, method="inverted_cdf")`).
Note the k > n case: Paper 01 clamps to the max score, which *breaks* the guarantee for tiny n; the
valid answer is +∞.

## Weighted conformal differs by design

Paper 01 uses the mean calibration weight as a proxy for the test point's weight. Paper 02 uses the test point's
own likelihood ratio, as in Tibshirani et al. (2019, Eq. 5). If the papers are to share one implementation, adopt
Paper 02's.
