## R2-D -- Transfer to CADC real snow (75 drives, radar absent)

### D1: zero-shot, LiDAR probes in reference-sensor units (spec-only normalisation, no CADC labels)

| arm | raw probes (Exp 5) | resolution-normalised |
|---|---|---|
| Fixed (status quo) | 0.740 | 0.740 |
| Per-frame classifier | 0.739 | 0.739 |
| Per-object (ours) | 0.734 | 0.734 |

D1_raw: ours − per-frame Δ -0.0050 [-0.0094, 0.0001], drive p 0.973

D1_normalised: ours − per-frame Δ -0.0047 [-0.0089, 0.0002], drive p 0.974

R2-D1 refuted: True

### D2: few-shot (k labelled CADC drives, 3 draws each)

| k | Fixed (status quo) | Per-frame classifier | Per-object (ours) | Δ ours − per-frame [95 % CI] | drive p |
|---|---|---|---|---|---|
| 1 | 0.740 | 0.742 | 0.743 | 0.0008 [-0.0009, 0.0025] | 0.195 |
| 5 | 0.743 | 0.741 | 0.759 | 0.0178 [0.0131, 0.0227] | 0.001 |
| 10 | 0.740 | 0.741 | 0.739 | -0.0021 [-0.0043, 0.0004] | 0.955 |

Δ non-decreasing in k: False. R2-D2 refuted: True
