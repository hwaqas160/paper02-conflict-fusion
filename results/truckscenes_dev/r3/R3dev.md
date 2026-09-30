## R3 development (TruckScenes official-train scenes only; dev-test = 104 held-out scenes)

| model | params | mAP all | mAP adverse | mAP clear | (set-transformer − model), adverse [95 % CI] |
|---|---|---|---|---|---|
| original | 18195 | 0.812 | 0.796 | 0.825 | -0.0123 [-0.0198, -0.0058] |
| V0 | 18195 | 0.811 | 0.799 | 0.822 | -0.0094 [-0.0175, -0.0022] |
| V1 | 85139 | 0.811 | 0.799 | 0.822 | -0.0089 [-0.0174, -0.0003] |
| V2 | 18390 | 0.825 | 0.813 | 0.837 | 0.0046 [-0.0044, 0.0123] |
| V3 | 85334 | 0.832 | 0.820 | 0.843 | 0.0115 [0.0043, 0.0198] |
| set_transformer | 72387 | 0.818 | 0.808 | 0.828 | -- |
| set_noattn | 38851 | 0.820 | 0.807 | 0.832 | -0.0011 [-0.0108, 0.0067] |

Selected: **V2** -- simplest variant within 0.005 of the set-transformer (0.8254 vs 0.8179)
