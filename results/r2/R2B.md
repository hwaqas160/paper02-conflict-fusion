## R2-B -- External late-fusion baselines (Car mAP, test scenes)

| method | params | clear0 | ADVERSE_POOLED | HELDOUT | fog3 | CADC (zero-shot) | ours − method, adverse (scene bootstrap) |
|---|---|---|---|---|---|---|---|
| Oracle selector (upper bound) | 0 | 0.914 | 0.847 | 0.886 | 0.511 | 0.856 | -- |
| Per-object (ours) | 18195 | 0.865 | 0.777 | 0.816 | 0.404 | 0.734 | -- |
| Set-transformer fusion (black box) | 72387 | 0.854 | 0.777 | 0.811 | 0.416 | 0.681 | 0.0005 (p_Holm 0.392) |
| CLOCs (Pang et al., 2020) | 5761 | 0.832 | 0.732 | 0.777 | 0.329 | 0.763 | 0.0455 (p_Holm 0.007) |
| ProbEn (Chen et al., 2022) | 7 | 0.761 | 0.653 | 0.681 | 0.283 | 0.705 | 0.1237 (p_Holm 0.007) |
| Dempster--Shafer evidential | 12 | 0.743 | 0.633 | 0.666 | 0.270 | 0.685 | 0.1438 (p_Holm 0.007) |
| Per-frame mixture of experts | 128 | 0.778 | 0.680 | 0.705 | 0.303 | 0.734 | 0.0971 (p_Holm 0.007) |
| Per-frame classifier | 10195 | 0.804 | 0.700 | 0.733 | 0.314 | 0.739 | 0.0770 (p_Holm 0.007) |
| Fixed (status quo) | 19 | 0.800 | 0.688 | 0.729 | 0.303 | 0.740 | 0.0892 (p_Holm 0.007) |

Oracle headroom captured by the per-object arm (adverse pooled): 56 %

