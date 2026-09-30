## R2-A -- MAN TruckScenes (real weather, 4D radar): 23902 keyframes, 598 scenes; scenes per condition {'clear0': 368, 'fog1': 12, 'rain1': 119, 'snow1': 23, 'night1': 76}

Primary: 5-fold scene CV stratified by weather, out-of-fold pooled Car mAP (every scene tested once).

| method | clear0 | ADVERSE_POOLED | ours − method, adverse (scene bootstrap) |
|---|---|---|---|
| Fixed (status quo) | 0.768 | 0.762 |  |
| Per-frame classifier | 0.767 | 0.763 | 0.0643 (p 0.001) |
| Per-object (ours) | 0.832 | 0.827 | -- |
| Per-object, radar removed | 0.758 | 0.750 | 0.0772 (p 0.001) |
| ProbEn (Chen et al., 2022) | 0.731 | 0.711 | 0.1162 (p_Holm 0.005) |
| Dempster--Shafer evidential | 0.650 | 0.609 | 0.2178 (p_Holm 0.005) |
| CLOCs (Pang et al., 2020) | 0.780 | 0.769 | 0.0582 (p_Holm 0.005) |
| Set-transformer fusion (black box) | 0.849 | 0.842 | -0.0149 (p_Holm 1.000) |
| Per-frame mixture of experts | 0.766 | 0.761 | 0.0658 (p_Holm 0.005) |
| Oracle selector (upper bound) | 0.867 | 0.872 |  |

R2-A1 (ours > per-frame, adverse): Δ 0.0643 [0.0559, 0.0737], p 0.001, 230 adverse scenes; folds ahead 5/5. Refuted: False

R2-A2 (status-quo harm ≥ 2 %): 0.86 % [0.83, 0.89] over 511508 pairs. Refuted: True

R2-A3 (4D radar adds value, adverse): Δ 0.0772 [0.0668, 0.0886], p 0.001. Refuted: False

Oracle headroom captured by ours: 59 %

### R2-C on TruckScenes (out-of-fold, adverse; each mechanism removed and retrained per fold)

| removed | adverse mAP | full − variant (p_Holm) | contributes? |
|---|---|---|---|
| no_absence_term | 0.826 | 0.0010 (0.749) | no |
| no_learned_precision | 0.824 | 0.0033 (0.085) | no |
| no_harm_aux_loss | 0.822 | 0.0051 (0.006) | yes |
| no_localisation_loss | 0.824 | 0.0032 (0.085) | no |
| no_sensor_dropout | 0.828 | -0.0015 (0.817) | no |
| no_frame_context | 0.826 | 0.0007 (0.749) | no |

### R2-E on TruckScenes (alpha = 0.1; calibrated on held-out CLEAR scenes only; out-of-fold)

| tag | cov vanilla | cov conflict | cov degr.-norm. / radius | FNR | FNR floor | flagged | FNR unflagged |
|---|---|---|---|---|---|---|---|
| clear0 | 0.900 | 0.901 | 0.901 / 0.59 m | 0.107 | 0.047 | 0.002 | 0.107 |
| fog1 | 0.879 | 0.900 | 0.911 / 0.70 m | 0.044 | 0.012 | 0.116 | 0.045 |
| night1 | 0.920 | 0.922 | 0.910 / 0.58 m | 0.074 | 0.026 | 0.000 | 0.074 |
| rain1 | 0.893 | 0.908 | 0.909 / 0.66 m | 0.107 | 0.042 | 0.012 | 0.107 |
| snow1 | 0.813 | 0.840 | 0.918 / 0.85 m | 0.167 | 0.079 | 0.414 | 0.204 |
### Standard protocol (official train_detect → val)

| method | clear0 | ADVERSE_POOLED |
|---|---|---|
| Fixed (status quo) | 0.756 | 0.801 |
| Per-frame classifier | 0.761 | 0.795 |
| Per-object (ours) | 0.811 | 0.849 |
| Per-object, radar removed | 0.751 | 0.783 |
| ProbEn (Chen et al., 2022) | 0.721 | 0.753 |
| Dempster--Shafer evidential | 0.649 | 0.667 |
| CLOCs (Pang et al., 2020) | 0.765 | 0.800 |
| Set-transformer fusion (black box) | 0.825 | 0.869 |
| Per-frame mixture of experts | 0.757 | 0.803 |
| Oracle selector (upper bound) | 0.856 | 0.881 |

Official split, ours − per-frame (adverse): Δ 0.0539, p 0.001 (31 adverse val scenes); status-quo harm 0.89 %
