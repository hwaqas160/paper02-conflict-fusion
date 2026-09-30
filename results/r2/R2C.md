## R2-C -- Mechanism ablations, each retrained (per-object arm, Car mAP, test)

| variant | clear0 | ADVERSE_POOLED | HELDOUT | fog3 | full − variant, adverse (scene bootstrap) | contributes? |
|---|---|---|---|---|---|---|
| full (ours) | 0.865 | 0.777 | 0.816 | 0.404 | -- | -- |
| no_absence_term | 0.865 | 0.775 | 0.815 | 0.397 | 0.0021 (p_Holm 0.400) | no |
| no_learned_precision | 0.872 | 0.782 | 0.815 | 0.406 | -0.0048 (p_Holm 1.000) | no |
| no_harm_aux_loss | 0.864 | 0.771 | 0.812 | 0.392 | 0.0066 (p_Holm 0.030) | yes |
| no_localisation_loss | 0.872 | 0.782 | 0.815 | 0.406 | -0.0048 (p_Holm 1.000) | no |
| no_sensor_dropout | 0.867 | 0.779 | 0.817 | 0.405 | -0.0016 (p_Holm 1.000) | no |
| no_frame_context | 0.864 | 0.777 | 0.817 | 0.403 | -0.0001 (p_Holm 1.000) | no |
