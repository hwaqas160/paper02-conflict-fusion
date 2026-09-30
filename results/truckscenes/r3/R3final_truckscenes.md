## R3 final -- frozen variant V2: TruckScenes official train -> val (75 scenes, 31 adverse)

| model | params | clear | adverse | all |
|---|---|---|---|---|
| original | 18195 | 0.811 | 0.849 | 0.832 |
| set_transformer | 72387 | 0.825 | 0.869 | 0.849 |
| V2 | 18390 | 0.824 | 0.872 | 0.850 |
| variant_rule_only | 18390 | 0.779 | 0.781 | 0.782 |

R3-1 lower bound of (variant - set-transformer), adverse: -0.0039 (margin -0.01); refuted: False
