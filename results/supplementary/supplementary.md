# Supplementary analyses

## S1 — Scene-cluster paired bootstrap (1,000 resamples of the test scenes)

Frames inside a scene are temporally correlated, so the frame bootstrap of Exp 2 is anti-conservative; this resamples whole scenes.

| pool | comparison (Car mAP) | Δ | 95 % CI | one-sided p | scenes won |
|---|---|---|---|---|---|
| clear0 | object − frame_cls | 0.0613 | [0.0405, 0.0910] | 0.0010 | 13/13 |
| clear0 | object − frame_oracle | 0.0618 | [0.0411, 0.0918] | 0.0010 | 13/13 |
| clear0 | object − frame_feats | 0.0729 | [0.0511, 0.1030] | 0.0010 | 13/13 |
| clear0 | object − confidence | 0.0504 | [0.0345, 0.0700] | 0.0010 | 13/13 |
| clear0 | object − fixed_clear | 0.0654 | [0.0442, 0.0965] | 0.0010 | 13/13 |
| clear0 | frame_cls − fixed_clear | 0.0042 | [0.0029, 0.0061] | 0.0010 | 13/13 |
| ADVERSE_POOLED | object − frame_cls | 0.0770 | [0.0543, 0.1058] | 0.0010 | 13/13 |
| ADVERSE_POOLED | object − frame_oracle | 0.0756 | [0.0534, 0.1033] | 0.0010 | 13/13 |
| ADVERSE_POOLED | object − frame_feats | 0.0867 | [0.0611, 0.1153] | 0.0010 | 13/13 |
| ADVERSE_POOLED | object − confidence | 0.0696 | [0.0476, 0.0948] | 0.0010 | 13/13 |
| ADVERSE_POOLED | object − fixed_clear | 0.0892 | [0.0654, 0.1194] | 0.0010 | 13/13 |
| ADVERSE_POOLED | frame_cls − fixed_clear | 0.0122 | [0.0086, 0.0165] | 0.0010 | 13/13 |
| HELDOUT | object − frame_cls | 0.0833 | [0.0543, 0.1223] | 0.0010 | 13/13 |
| HELDOUT | object − frame_oracle | 0.0783 | [0.0521, 0.1087] | 0.0010 | 13/13 |
| HELDOUT | object − frame_feats | 0.0936 | [0.0625, 0.1315] | 0.0010 | 13/13 |
| HELDOUT | object − confidence | 0.0721 | [0.0460, 0.1034] | 0.0010 | 13/13 |
| HELDOUT | object − fixed_clear | 0.0871 | [0.0582, 0.1279] | 0.0010 | 13/13 |
| HELDOUT | frame_cls − fixed_clear | 0.0038 | [0.0014, 0.0065] | 0.0010 | 11/13 |
| fog3 | object − frame_cls | 0.0900 | [0.0723, 0.1064] | 0.0010 | 13/13 |
| fog3 | object − frame_oracle | 0.0873 | [0.0679, 0.1069] | 0.0010 | 13/13 |
| fog3 | object − frame_feats | 0.0941 | [0.0767, 0.1117] | 0.0010 | 13/13 |
| fog3 | object − confidence | 0.0785 | [0.0612, 0.0947] | 0.0010 | 13/13 |
| fog3 | object − fixed_clear | 0.1012 | [0.0841, 0.1154] | 0.0010 | 13/13 |
| fog3 | frame_cls − fixed_clear | 0.0112 | [0.0061, 0.0151] | 0.0010 | 10/13 |


## S2 — Where does the per-object gain come from?

Car AP per centre-distance threshold, and swaps of fused scores (existence) and fused positions (localisation) between the status quo and the per-object arm.

| pool | variant | mAP | AP@0.5 m | AP@1.0 m | AP@2.0 m | AP@4.0 m |
|---|---|---|---|---|---|---|
| clear0 | status_quo | 0.800 | 0.717 | 0.795 | 0.831 | 0.856 |
| clear0 | frame_cls | 0.804 | 0.720 | 0.799 | 0.837 | 0.860 |
| clear0 | object | 0.865 | 0.763 | 0.860 | 0.908 | 0.930 |
| clear0 | object_scores+status_quo_positions | 0.862 | 0.762 | 0.858 | 0.902 | 0.927 |
| clear0 | status_quo_scores+object_positions | 0.803 | 0.721 | 0.798 | 0.834 | 0.858 |
| clear0 | → share of gain from scores / positions | 96 % / 5 % |  |  |  |  |
| ADVERSE_POOLED | status_quo | 0.688 | 0.601 | 0.685 | 0.720 | 0.746 |
| ADVERSE_POOLED | frame_cls | 0.700 | 0.611 | 0.697 | 0.733 | 0.759 |
| ADVERSE_POOLED | object | 0.777 | 0.648 | 0.770 | 0.830 | 0.861 |
| ADVERSE_POOLED | object_scores+status_quo_positions | 0.774 | 0.649 | 0.766 | 0.826 | 0.857 |
| ADVERSE_POOLED | status_quo_scores+object_positions | 0.690 | 0.603 | 0.687 | 0.723 | 0.749 |
| ADVERSE_POOLED | → share of gain from scores / positions | 97 % / 3 % |  |  |  |  |
| HELDOUT | status_quo | 0.729 | 0.644 | 0.727 | 0.758 | 0.786 |
| HELDOUT | frame_cls | 0.733 | 0.647 | 0.732 | 0.763 | 0.789 |
| HELDOUT | object | 0.816 | 0.697 | 0.815 | 0.863 | 0.888 |
| HELDOUT | object_scores+status_quo_positions | 0.813 | 0.702 | 0.810 | 0.857 | 0.883 |
| HELDOUT | status_quo_scores+object_positions | 0.731 | 0.645 | 0.730 | 0.761 | 0.787 |
| HELDOUT | → share of gain from scores / positions | 97 % / 2 % |  |  |  |  |
| fog3 | status_quo | 0.303 | 0.211 | 0.288 | 0.339 | 0.373 |
| fog3 | frame_cls | 0.314 | 0.219 | 0.298 | 0.353 | 0.387 |
| fog3 | object | 0.404 | 0.235 | 0.362 | 0.479 | 0.541 |
| fog3 | object_scores+status_quo_positions | 0.398 | 0.231 | 0.349 | 0.472 | 0.539 |
| fog3 | status_quo_scores+object_positions | 0.307 | 0.215 | 0.293 | 0.344 | 0.376 |
| fog3 | → share of gain from scores / positions | 94 % / 4 % |  |  |  |  |


## S3 — Probes vs geometry vs capacity (Car mAP, test)

| arm | params | clear0 | ADVERSE_POOLED | HELDOUT | fog3 |
|---|---|---|---|---|---|
| per-object (ours) | 18195 | 0.865 | 0.777 | 0.816 | 0.404 |
| CLOCs-style (conf + conflict + det feats, no probes) | 11795 | 0.846 | 0.745 | 0.793 | 0.345 |
| CLOCs-style + frame context | 15891 | 0.843 | 0.746 | 0.788 | 0.351 |
| probes only (no conf / conflict / det feats) | 16275 | 0.853 | 0.764 | 0.804 | 0.387 |
| per-frame features, hidden 88 (≈ ours' params) | 18851 | 0.790 | 0.688 | 0.723 | 0.309 |
| per-frame classifier, hidden 88 | 18235 | 0.802 | 0.699 | 0.729 | 0.316 |
| per-object, hidden 40 (≈ per-frame params) | 8499 | 0.873 | 0.783 | 0.827 | 0.407 |
| per-frame classifier (Exp 2) | 10195 | 0.804 | 0.700 | 0.733 | 0.314 |
| per-object, negative-evidence term removed (test time) | 18195 | 0.860 | 0.771 | 0.808 | 0.401 |


## S4 — Radar removed from the data; every arm retrained on camera + LiDAR only

| arm | Boreas clear0 | Boreas ADVERSE_POOLED | Boreas HELDOUT | Boreas fog3 | CADC real snow (95 % CI) |
|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.800 | 0.690 | 0.728 | 0.297 | 0.742 (0.735–0.748) |
| Fixed weights | 0.801 | 0.694 | 0.731 | 0.299 | 0.738 (0.732–0.745) |
| Confidence-only | 0.801 | 0.692 | 0.729 | 0.300 | 0.738 (0.731–0.744) |
| Per-frame weather classifier | 0.801 | 0.696 | 0.729 | 0.300 | 0.742 (0.736–0.749) |
| Per-frame oracle weather | 0.801 | 0.696 | 0.732 | 0.299 | 0.740 (0.734–0.747) |
| Per-frame features | 0.780 | 0.683 | 0.714 | 0.296 | 0.741 (0.735–0.748) |
| Per-object (ours) | 0.866 | 0.771 | 0.813 | 0.351 | 0.726 (0.718–0.732) |

Per-object − per-frame classifier on radar-free Boreas: ADVERSE_POOLED: Δ 0.0754 (frame p 0.001, scene p 0.001); HELDOUT: Δ 0.0845 (frame p 0.001, scene p 0.001)

CADC object − frame_cls: Δ -0.0162 (frame p 1.000; drive-cluster p 1.000, CI [-0.0259, -0.0090], 75 drives)

CADC object − fixed_clear: Δ -0.0156 (frame p 1.000; drive-cluster p 1.000, CI [-0.0252, -0.0080], 75 drives)

CADC object − confidence: Δ -0.0115 (frame p 1.000; drive-cluster p 0.996, CI [-0.0221, -0.0026], 75 drives)


## S5 — Textbook late-fusion rules and single sensors (Car mAP, test)

| method | clear0 | ADVERSE_POOLED | HELDOUT | fog3 |
|---|---|---|---|---|
| untrained max | 0.394 | 0.334 | 0.352 | 0.142 |
| untrained noisy_or | 0.440 | 0.368 | 0.398 | 0.158 |
| untrained sum_logits | 0.363 | 0.270 | 0.312 | 0.054 |
| camera only (raw detector) | 0.169 | 0.128 | 0.151 | 0.022 |
| lidar only (raw detector) | 0.766 | 0.654 | 0.679 | 0.278 |
| radar only (raw detector) | 0.012 | 0.009 | 0.011 | 0.005 |
| fitted fixed (status quo) | 0.800 | 0.688 | 0.729 | 0.303 |
| per-object (ours) | 0.865 | 0.777 | 0.816 | 0.404 |
