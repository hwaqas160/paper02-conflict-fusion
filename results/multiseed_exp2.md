# Exp 2 over seeds [2, 3, 4] — Car mAP mean ± sd

| arm | clear0 | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.800 ± 0.001 | 0.689 ± 0.001 | 0.731 ± 0.001 | 0.674 ± 0.001 |
| Fixed weights | 0.806 ± 0.001 | 0.700 ± 0.001 | 0.739 ± 0.002 | 0.685 ± 0.001 |
| Confidence-only | 0.815 ± 0.001 | 0.709 ± 0.001 | 0.746 ± 0.001 | 0.694 ± 0.001 |
| Per-frame weather classifier | 0.801 ± 0.001 | 0.699 ± 0.002 | 0.731 ± 0.003 | 0.686 ± 0.002 |
| Per-frame oracle weather | 0.800 ± 0.002 | 0.699 ± 0.004 | 0.732 ± 0.008 | 0.688 ± 0.002 |
| Per-frame features | 0.785 ± 0.005 | 0.689 ± 0.006 | 0.726 ± 0.007 | 0.674 ± 0.005 |
| Per-object (ours) | 0.866 ± 0.005 | 0.779 ± 0.003 | 0.816 ± 0.003 | 0.762 ± 0.004 |