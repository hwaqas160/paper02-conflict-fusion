## R2-F -- 5-fold scene cross-validation (Boreas, all 37 scenes, adverse pooled Car mAP)

| fold | status quo | per-frame classifier | per-object (ours) | Δ ours − per-frame |
|---|---|---|---|---|
| 0 | 0.647 | 0.667 | 0.748 | 0.0803 |
| 1 | 0.650 | 0.669 | 0.755 | 0.0861 |
| 2 | 0.596 | 0.611 | 0.716 | 0.1049 |
| 3 | 0.666 | 0.675 | 0.761 | 0.0859 |
| 4 | 0.706 | 0.719 | 0.790 | 0.0716 |

Out-of-fold pooled: ours 0.750 vs per-frame 0.664, Δ 0.0861 (95 % CI [0.0684, 0.1075], scene p 0.001); ours ahead in 5/5 folds. R2-F1 refuted: False

