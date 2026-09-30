## R2-E -- Repairing risk control under shift (alpha = 0.1, nominal coverage 0.90, FNR target 0.1)

Localisation: coverage / mean radius; all scales fitted on training scenes, calibrated on clear cal-scene frames only. Detection: frame FNR with lambda calibrated on clear only vs per-degradation-bin (label-free) on clear + mild cal-scene frames.

| condition | vanilla | conflict | degradation-normalised | FNR clear-calibrated | FNR degradation-Mondrian |
|---|---|---|---|---|---|
| clear0 | 0.958 / 0.75 m | 0.953 / 0.67 m | 0.945 / 0.66 m | 0.060 | 0.050 |
| fog1 | 0.931 / 0.75 m | 0.928 / 0.75 m | 0.926 / 0.72 m | 0.072 | 0.077 |
| fog2 | 0.921 / 0.75 m | 0.924 / 0.81 m | 0.935 / 0.90 m | 0.127 | 0.135 |
| fog3 | 0.839 / 0.75 m | 0.867 / 0.86 m | 0.915 / 1.09 m | 0.364 | 0.379 |
| rain1 | 0.960 / 0.75 m | 0.957 / 0.70 m | 0.945 / 0.66 m | 0.070 | 0.061 |
| rain2 | 0.955 / 0.75 m | 0.954 / 0.72 m | 0.942 / 0.66 m | 0.069 | 0.069 |
| rain3 | 0.950 / 0.75 m | 0.953 / 0.77 m | 0.939 / 0.68 m | 0.071 | 0.075 |
| snow1 | 0.954 / 0.75 m | 0.950 / 0.69 m | 0.941 / 0.66 m | 0.070 | 0.062 |
| snow2 | 0.950 / 0.75 m | 0.951 / 0.70 m | 0.938 / 0.67 m | 0.084 | 0.080 |
| snow3 | 0.950 / 0.75 m | 0.944 / 0.72 m | 0.933 / 0.69 m | 0.105 | 0.108 |
| night2 | 0.954 / 0.75 m | 0.952 / 0.73 m | 0.942 / 0.67 m | 0.069 | 0.055 |
| SNOW_POOLED | 0.951 / 0.75 m | 0.948 / 0.70 m | 0.937 / 0.67 m | 0.086 | 0.083 |
| SEVERE | 0.926 / 0.75 m | 0.931 / 0.77 m | 0.931 / 0.77 m | 0.180 | 0.187 |

R2-E1 refuted (fog3 coverage < 0.85 or FNR > 0.15): True

### Exploratory: detection floor and label-free support check

FNR floor = frame FNR when EVERY candidate is accepted (lambda = 0): no threshold-based risk control can go below it. Flagged = frames whose label-free degradation score exceeds the 99th percentile of the calibration frames.

| condition | FNR floor (accept all) | flagged out of support | FNR on unflagged frames |
|---|---|---|---|
| clear0 | 0.032 | 0.000 | 0.060 |
| fog1 | 0.042 | 0.032 | 0.074 |
| fog2 | 0.077 | 0.914 | 0.169 |
| fog3 | 0.301 | 1.000 | – |
| rain1 | 0.041 | 0.000 | 0.070 |
| rain2 | 0.041 | 0.000 | 0.069 |
| rain3 | 0.038 | 0.005 | 0.071 |
| snow1 | 0.046 | 0.000 | 0.070 |
| snow2 | 0.051 | 0.000 | 0.084 |
| snow3 | 0.057 | 0.000 | 0.105 |
| night2 | 0.036 | 0.000 | 0.069 |
| SNOW_POOLED | 0.051 | 0.000 | 0.086 |
| SEVERE | 0.132 | 0.335 | 0.088 |
