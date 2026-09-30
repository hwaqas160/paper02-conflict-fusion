# Paper 02 — Results report

Source hash `c3a6052e08425c5e`; generated 2026-09-22 22:16.

Frames: 54461 (all conditions); tags: clear0, fog1, fog2, fog3, rain1, rain2, rain3, snow1, snow2, snow3, night2.


---
<!-- exp1_harm/exp1_harm.md -->
# Exp 1 — Harm rate under status-quo fusion (Car, test scenes)

Status quo = fixed-weight late fusion fitted on clear; operating threshold τ = 0.405 (F1 0.777 on clear train).

Decision-flip harm rate, % of candidates containing the modality (95 % frame-bootstrap CI)

| condition | pooled | camera | lidar | radar | help % (pooled) |
|---|---|---|---|---|---|
| clear0 | 1.9 (1.8–2.0) | 3.8 (3.5–4.1) | 2.4 (2.2–2.5) | 0.7 (0.7–0.8) | 5.3 |
| fog1 | 2.1 (2.0–2.2) | 3.4 (3.1–3.6) | 2.9 (2.7–3.0) | 0.8 (0.7–0.9) | 4.8 |
| fog2 | 1.5 (1.4–1.6) | 1.9 (1.7–2.1) | 2.1 (2.0–2.3) | 0.9 (0.8–1.0) | 3.9 |
| fog3 | 0.8 (0.8–0.9) | 1.4 (1.1–1.7) | 1.2 (1.1–1.4) | 0.6 (0.5–0.7) | 2.8 |
| rain1 | 2.1 (2.0–2.3) | 3.8 (3.5–4.0) | 2.9 (2.7–3.1) | 0.8 (0.7–0.9) | 5.3 |
| rain2 | 1.8 (1.7–1.9) | 2.5 (2.3–2.7) | 2.7 (2.5–2.9) | 0.8 (0.7–0.9) | 5.2 |
| rain3 | 1.6 (1.5–1.7) | 1.8 (1.6–2.0) | 2.5 (2.3–2.6) | 0.9 (0.8–0.9) | 4.9 |
| snow1 | 2.3 (2.2–2.4) | 5.1 (4.8–5.4) | 2.5 (2.4–2.7) | 0.7 (0.6–0.8) | 4.8 |
| snow2 | 2.3 (2.2–2.4) | 4.3 (4.1–4.6) | 2.7 (2.5–2.9) | 0.8 (0.7–0.9) | 4.6 |
| snow3 | 2.2 (2.1–2.3) | 3.2 (3.0–3.4) | 2.9 (2.8–3.1) | 0.8 (0.7–0.9) | 4.6 |
| night2 | 1.5 (1.4–1.6) | 1.5 (1.4–1.7) | 2.3 (2.1–2.5) | 0.9 (0.8–1.0) | 5.9 |
| ADVERSE_POOLED | 1.9 (1.9–1.9) | 3.1 (3.1–3.2) | 2.5 (2.5–2.6) | 0.8 (0.8–0.8) | 4.7 |

## AP-level harm (Car mAP, status quo): removing a sensor *improves* mAP when the cell exceeds `all`

| condition | all | − camera | − lidar | − radar |
|---|---|---|---|---|
| clear0 | 0.800 | 0.780 | 0.186 | 0.795 |
| fog1 | 0.718 | 0.683 | 0.166 | 0.712 |
| fog2 | 0.618 | 0.603 | 0.123 | 0.610 |
| fog3 | 0.303 | 0.292 | 0.029 | 0.292 |
| rain1 | 0.782 | 0.765 | 0.180 | 0.778 |
| rain2 | 0.774 | 0.754 | 0.171 | 0.770 |
| rain3 | 0.750 | 0.737 | 0.148 | 0.747 |
| snow1 | 0.760 | 0.730 | 0.177 | 0.754 |
| snow2 | 0.729 | 0.696 | 0.169 | 0.721 |
| snow3 | 0.698 | 0.669 | 0.154 | 0.690 |
| night2 | 0.797 | 0.780 | 0.121 | 0.792 |

## Operating-threshold sweep (adverse pooled; τ chosen on clear = 0.405)

| τ | pooled harm % | help % | camera % | lidar % | radar % |
|---|---|---|---|---|---|
| 0.200 | 3.52 | 5.97 | 7.22 | 5.05 | 0.60 |
| 0.300 | 2.58 | 5.10 | 4.80 | 3.62 | 0.72 |
| 0.405 | 1.89 | 4.74 | 3.13 | 2.54 | 0.79 |
| 0.500 | 1.53 | 4.64 | 2.18 | 1.96 | 0.87 |
| 0.700 | 0.90 | 5.03 | 0.91 | 0.86 | 0.93 |
| 0.900 | 0.49 | 4.75 | 0.32 | 0.22 | 0.80 |

## Secondary slices (reported, not part of the pre-registered rule)

| slice | condition | pooled harm % | camera % | lidar % | radar % | help % | n pairs |
|---|---|---|---|---|---|---|---|
| multimodal_ge2 | clear0 | 3.46 | 4.37 | 3.91 | 1.97 | 9.45 | 52897 |
| multimodal_ge2 | ADVERSE_POOLED | 3.52 | 3.66 | 4.49 | 2.16 | 8.77 | 533358 |
| gt_backed | clear0 | 2.01 | 0.07 | 1.19 | 5.61 | 20.08 | 21343 |
| gt_backed | ADVERSE_POOLED | 2.26 | 0.19 | 0.93 | 6.43 | 21.61 | 192549 |
| multimodal_ge2_gt_backed | clear0 | 2.07 | 0.07 | 1.28 | 5.65 | 19.75 | 20701 |
| multimodal_ge2_gt_backed | ADVERSE_POOLED | 2.40 | 0.19 | 1.03 | 6.72 | 21.88 | 181610 |

**H1 verdict:** pooled adverse harm = 1.89 % (CI 1.86–1.92); refuted (<2 %): **True**; clear-weather harm = 1.93 %, weather-induced excess = -0.05 pp; AP-level harm cells: 0.


---
<!-- exp2_routing/exp2_routing.md -->
# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.800 | 0.718 | 0.618 | 0.303 | 0.782 | 0.774 | 0.750 | 0.760 | 0.729 | 0.698 | 0.797 | 0.674 | 0.729 | 0.688 |
| Fixed weights | 19 | 0.807 | 0.730 | 0.635 | 0.310 | 0.791 | 0.784 | 0.764 | 0.768 | 0.739 | 0.709 | 0.805 | 0.686 | 0.738 | 0.700 |
| Confidence-only | 10579 | 0.815 | 0.736 | 0.640 | 0.326 | 0.798 | 0.789 | 0.766 | 0.775 | 0.743 | 0.714 | 0.807 | 0.693 | 0.744 | 0.707 |
| Per-frame weather classifier | 10195 | 0.804 | 0.735 | 0.641 | 0.314 | 0.787 | 0.780 | 0.759 | 0.763 | 0.732 | 0.703 | 0.802 | 0.687 | 0.733 | 0.700 |
| Per-frame oracle weather | 10259 | 0.803 | 0.725 | 0.642 | 0.317 | 0.785 | 0.778 | 0.761 | 0.767 | 0.739 | 0.709 | 0.802 | 0.690 | 0.738 | 0.702 |
| Per-frame features | 10643 | 0.792 | 0.712 | 0.621 | 0.310 | 0.774 | 0.768 | 0.756 | 0.751 | 0.723 | 0.695 | 0.788 | 0.677 | 0.722 | 0.690 |
| Per-object (ours) | 18195 | 0.865 | 0.806 | 0.717 | 0.404 | 0.855 | 0.846 | 0.831 | 0.844 | 0.818 | 0.785 | 0.855 | 0.760 | 0.816 | 0.777 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.618 | 0.605 | 0.623 |
| Confidence-only | 0.988 | 0.988 | 0.988 |
| Per-frame weather classifier | 0.623 | 0.609 | 0.630 |
| Per-frame oracle weather | 0.657 | 0.662 | 0.644 |
| Per-frame features | 0.660 | 0.661 | 0.651 |
| Per-object (ours) | 0.980 | 0.975 | 0.983 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0770 | 0.001 | 0.0867 | 0.001 | 0.0696 | 0.001 | 0.980 | 0.623 | 0.988 |
| HELDOUT | 0.0833 | 0.001 | 0.0936 | 0.001 | 0.0721 | 0.001 | 0.975 | 0.609 | 0.988 |
| SEEN_ADVERSE | 0.0731 | 0.001 | 0.0834 | 0.001 | 0.0672 | 0.001 | 0.983 | 0.630 | 0.988 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992


---
<!-- exp3_ablation/exp3_ablation.md -->
# Exp 3 — Ablation over degradation-feature groups (per-object arm, Car mAP, test)

| variant | ADVERSE_POOLED (Δ vs full) | HELDOUT (Δ vs full) | clear0 (Δ vs full) |
|---|---|---|---|
| full | 0.777 (0.000) | 0.816 (0.000) | 0.865 (0.000) |
| no_context | 0.777 (0.000) | 0.817 (0.001) | 0.864 (-0.001) |
| -geometry | 0.765 (-0.012) | 0.796 (-0.020) | 0.863 (-0.002) |
| -lidar_probes | 0.763 (-0.014) | 0.801 (-0.015) | 0.851 (-0.014) |
| -camera_probes | 0.773 (-0.004) | 0.807 (-0.009) | 0.869 (0.004) |
| -radar_probes | 0.778 (0.001) | 0.817 (0.001) | 0.867 (0.002) |
| -det_feats | 0.774 (-0.003) | 0.810 (-0.006) | 0.865 (-0.000) |
| -conflict | 0.780 (0.003) | 0.819 (0.003) | 0.869 (0.004) |
| -confidence | 0.777 (0.000) | 0.816 (0.000) | 0.869 (0.004) |
| -frame_feats | 0.784 (0.007) | 0.818 (0.002) | 0.873 (0.007) |

---
<!-- exp4_conformal/exp4_conformal.md -->
# Exp 4 — Calibrate on clear (cal scenes), deploy under weather (test scenes)


## Fixed (clear-fit, status quo) — α = 0.05 (nominal coverage 0.95), CRC λ̂ = 0.000

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.05) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.987 / 1.17 m | 0.986 / 1.09 m | 0.966 / 0.78 m | – | 0.031 | 37.92 | 6728 |
| fog1 | 0.979 / 1.16 m | 0.977 / 1.09 m | 0.960 / 0.78 m | 1.000 / inf m (ESS 1422) | 0.041 | 43.54 | 6067 |
| fog2 | 0.983 / 1.12 m | 0.983 / 1.06 m | 0.965 / 0.79 m | 1.000 / inf m (ESS 2135) | 0.076 | 41.87 | 4649 |
| fog3 | 0.976 / 1.04 m | 0.975 / 0.99 m | 0.960 / 0.88 m | 1.000 / inf m (ESS 2547) | 0.304 | 35.95 | 2348 |
| rain1 | 0.983 / 1.16 m | 0.983 / 1.09 m | 0.965 / 0.78 m | 1.000 / 3.28 m (ESS 342) | 0.038 | 38.62 | 6622 |
| rain2 | 0.985 / 1.16 m | 0.984 / 1.09 m | 0.967 / 0.78 m | 1.000 / inf m (ESS 734) | 0.037 | 39.37 | 6406 |
| rain3 | 0.988 / 1.15 m | 0.987 / 1.08 m | 0.974 / 0.78 m | 1.000 / inf m (ESS 776) | 0.035 | 39.41 | 5858 |
| snow1 | 0.985 / 1.17 m | 0.985 / 1.09 m | 0.966 / 0.77 m | 0.999 / 1.74 m (ESS 160) | 0.047 | 43.31 | 6536 |
| snow2 | 0.982 / 1.16 m | 0.981 / 1.09 m | 0.965 / 0.77 m | 1.000 / 2.20 m (ESS 297) | 0.050 | 45.06 | 6262 |
| snow3 | 0.985 / 1.16 m | 0.984 / 1.08 m | 0.968 / 0.78 m | 1.000 / inf m (ESS 465) | 0.053 | 44.68 | 5869 |
| night2 | 0.987 / 1.16 m | 0.988 / 1.10 m | 0.972 / 0.79 m | 1.000 / inf m (ESS 1459) | 0.033 | 38.58 | 6273 |
| SNOW_POOLED | 0.984 / 1.16 m | 0.983 / 1.09 m | 0.967 / 0.77 m | – | 0.050 | 44.35 | 18667 |
| ADVERSE_POOLED | 0.984 / 1.15 m | 0.983 / 1.08 m | 0.967 / 0.78 m | – | 0.071 | 41.04 | 56890 |

## Fixed (clear-fit, status quo) — α = 0.1 (nominal coverage 0.90), CRC λ̂ = 0.012

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.1) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.936 / 0.50 m | 0.938 / 0.51 m | 0.931 / 0.62 m | – | 0.038 | 20.97 | 6728 |
| fog1 | 0.900 / 0.50 m | 0.904 / 0.51 m | 0.920 / 0.62 m | 1.000 / inf m (ESS 1422) | 0.048 | 27.44 | 6067 |
| fog2 | 0.908 / 0.48 m | 0.912 / 0.50 m | 0.918 / 0.63 m | 1.000 / inf m (ESS 2135) | 0.100 | 22.70 | 4649 |
| fog3 | 0.859 / 0.45 m | 0.869 / 0.46 m | 0.905 / 0.76 m | 1.000 / inf m (ESS 2547) | 0.460 | 11.80 | 2348 |
| rain1 | 0.934 / 0.50 m | 0.936 / 0.51 m | 0.929 / 0.62 m | 1.000 / 0.71 m (ESS 342) | 0.046 | 22.08 | 6622 |
| rain2 | 0.934 / 0.50 m | 0.936 / 0.51 m | 0.930 / 0.62 m | 1.000 / inf m (ESS 734) | 0.045 | 22.94 | 6406 |
| rain3 | 0.938 / 0.50 m | 0.940 / 0.51 m | 0.936 / 0.63 m | 1.000 / inf m (ESS 776) | 0.045 | 23.14 | 5858 |
| snow1 | 0.931 / 0.50 m | 0.934 / 0.51 m | 0.929 / 0.61 m | 0.998 / 1.36 m (ESS 160) | 0.054 | 27.29 | 6536 |
| snow2 | 0.918 / 0.50 m | 0.918 / 0.51 m | 0.929 / 0.61 m | 1.000 / 1.89 m (ESS 297) | 0.062 | 29.34 | 6262 |
| snow3 | 0.904 / 0.50 m | 0.904 / 0.51 m | 0.930 / 0.62 m | 1.000 / inf m (ESS 465) | 0.066 | 29.19 | 5869 |
| night2 | 0.939 / 0.50 m | 0.942 / 0.52 m | 0.939 / 0.63 m | 1.000 / inf m (ESS 1459) | 0.046 | 20.72 | 6273 |
| SNOW_POOLED | 0.918 / 0.50 m | 0.919 / 0.51 m | 0.929 / 0.61 m | – | 0.061 | 28.61 | 18667 |
| ADVERSE_POOLED | 0.921 / 0.50 m | 0.923 / 0.51 m | 0.928 / 0.63 m | – | 0.097 | 23.66 | 56890 |

## Fixed (clear-fit, status quo) — α = 0.2 (nominal coverage 0.80), CRC λ̂ = 0.166

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.2) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.840 / 0.34 m | 0.842 / 0.34 m | 0.834 / 0.34 m | – | 0.120 | 3.07 | 6728 |
| fog1 | 0.795 / 0.33 m | 0.800 / 0.34 m | 0.783 / 0.34 m | 1.000 / inf m (ESS 1422) | 0.167 | 3.77 | 6067 |
| fog2 | 0.792 / 0.32 m | 0.793 / 0.33 m | 0.780 / 0.33 m | 1.000 / inf m (ESS 2135) | 0.298 | 2.03 | 4649 |
| fog3 | 0.735 / 0.30 m | 0.742 / 0.31 m | 0.744 / 0.34 m | 1.000 / inf m (ESS 2547) | 0.608 | 0.58 | 2348 |
| rain1 | 0.839 / 0.33 m | 0.842 / 0.34 m | 0.828 / 0.34 m | 1.000 / 0.85 m (ESS 342) | 0.126 | 3.40 | 6622 |
| rain2 | 0.837 / 0.33 m | 0.840 / 0.34 m | 0.825 / 0.34 m | 1.000 / inf m (ESS 734) | 0.143 | 3.09 | 6406 |
| rain3 | 0.835 / 0.33 m | 0.837 / 0.33 m | 0.822 / 0.34 m | 1.000 / inf m (ESS 776) | 0.174 | 2.54 | 5858 |
| snow1 | 0.831 / 0.34 m | 0.833 / 0.34 m | 0.826 / 0.34 m | 0.984 / 0.54 m (ESS 160) | 0.136 | 4.08 | 6536 |
| snow2 | 0.813 / 0.34 m | 0.813 / 0.34 m | 0.806 / 0.34 m | 1.000 / 1.24 m (ESS 297) | 0.154 | 4.41 | 6262 |
| snow3 | 0.804 / 0.33 m | 0.803 / 0.33 m | 0.789 / 0.34 m | 1.000 / inf m (ESS 465) | 0.179 | 4.34 | 5869 |
| night2 | 0.846 / 0.33 m | 0.851 / 0.34 m | 0.840 / 0.34 m | 1.000 / inf m (ESS 1459) | 0.147 | 2.08 | 6273 |
| SNOW_POOLED | 0.816 / 0.33 m | 0.817 / 0.34 m | 0.808 / 0.34 m | – | 0.156 | 4.28 | 18667 |
| ADVERSE_POOLED | 0.819 / 0.33 m | 0.821 / 0.33 m | 0.810 / 0.34 m | – | 0.213 | 3.03 | 56890 |

## Per-object (ours) — α = 0.05 (nominal coverage 0.95), CRC λ̂ = 0.000

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.05) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.983 / 1.30 m | 0.980 / 1.10 m | 0.981 / 1.11 m | – | 0.032 | 15.14 | 7579 |
| fog1 | 0.973 / 1.30 m | 0.973 / 1.23 m | 0.973 / 1.25 m | 1.000 / inf m (ESS 1606) | 0.042 | 23.36 | 7327 |
| fog2 | 0.965 / 1.30 m | 0.971 / 1.33 m | 0.972 / 1.35 m | 1.000 / inf m (ESS 2424) | 0.077 | 25.81 | 6666 |
| fog3 | 0.945 / 1.30 m | 0.961 / 1.42 m | 0.965 / 1.43 m | 1.000 / inf m (ESS 2952) | 0.301 | 28.16 | 4002 |
| rain1 | 0.985 / 1.30 m | 0.984 / 1.15 m | 0.984 / 1.17 m | 1.000 / 1.30 m (ESS 396) | 0.041 | 16.21 | 7516 |
| rain2 | 0.981 / 1.30 m | 0.981 / 1.18 m | 0.982 / 1.19 m | 1.000 / inf m (ESS 825) | 0.041 | 19.59 | 7499 |
| rain3 | 0.977 / 1.30 m | 0.980 / 1.26 m | 0.982 / 1.28 m | 1.000 / inf m (ESS 872) | 0.038 | 21.53 | 7371 |
| snow1 | 0.983 / 1.30 m | 0.981 / 1.13 m | 0.981 / 1.14 m | 0.999 / 1.66 m (ESS 197) | 0.046 | 18.45 | 7529 |
| snow2 | 0.978 / 1.30 m | 0.977 / 1.15 m | 0.978 / 1.17 m | 1.000 / 3.28 m (ESS 344) | 0.051 | 19.89 | 7301 |
| snow3 | 0.981 / 1.30 m | 0.980 / 1.18 m | 0.981 / 1.19 m | 1.000 / inf m (ESS 533) | 0.057 | 20.09 | 6911 |
| night2 | 0.985 / 1.30 m | 0.986 / 1.19 m | 0.986 / 1.21 m | 1.000 / inf m (ESS 1662) | 0.036 | 16.16 | 7535 |
| SNOW_POOLED | 0.980 / 1.30 m | 0.979 / 1.15 m | 0.980 / 1.17 m | – | 0.051 | 19.48 | 21741 |
| ADVERSE_POOLED | 0.977 / 1.30 m | 0.978 / 1.21 m | 0.979 / 1.23 m | – | 0.073 | 20.93 | 69657 |

## Per-object (ours) — α = 0.1 (nominal coverage 0.90), CRC λ̂ = 0.039

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.1) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.958 / 0.75 m | 0.953 / 0.67 m | 0.954 / 0.69 m | – | 0.060 | 2.90 | 7579 |
| fog1 | 0.931 / 0.75 m | 0.928 / 0.75 m | 0.932 / 0.77 m | 1.000 / inf m (ESS 1606) | 0.072 | 4.83 | 7327 |
| fog2 | 0.921 / 0.75 m | 0.924 / 0.81 m | 0.926 / 0.84 m | 1.000 / inf m (ESS 2424) | 0.126 | 5.27 | 6666 |
| fog3 | 0.839 / 0.75 m | 0.867 / 0.86 m | 0.876 / 0.90 m | 1.000 / inf m (ESS 2952) | 0.362 | 5.39 | 4002 |
| rain1 | 0.960 / 0.75 m | 0.957 / 0.70 m | 0.958 / 0.72 m | 1.000 / 1.02 m (ESS 396) | 0.069 | 3.07 | 7516 |
| rain2 | 0.955 / 0.75 m | 0.954 / 0.72 m | 0.955 / 0.73 m | 1.000 / inf m (ESS 825) | 0.069 | 3.80 | 7499 |
| rain3 | 0.950 / 0.75 m | 0.953 / 0.77 m | 0.954 / 0.78 m | 1.000 / inf m (ESS 872) | 0.070 | 4.18 | 7371 |
| snow1 | 0.954 / 0.75 m | 0.950 / 0.69 m | 0.952 / 0.71 m | 0.997 / 1.17 m (ESS 197) | 0.070 | 3.54 | 7529 |
| snow2 | 0.950 / 0.75 m | 0.951 / 0.70 m | 0.953 / 0.72 m | 1.000 / 1.26 m (ESS 344) | 0.084 | 3.92 | 7301 |
| snow3 | 0.950 / 0.75 m | 0.944 / 0.72 m | 0.948 / 0.74 m | 1.000 / inf m (ESS 533) | 0.105 | 3.68 | 6911 |
| night2 | 0.954 / 0.75 m | 0.952 / 0.73 m | 0.959 / 0.75 m | 1.000 / inf m (ESS 1662) | 0.068 | 2.59 | 7535 |
| SNOW_POOLED | 0.951 / 0.75 m | 0.948 / 0.70 m | 0.951 / 0.72 m | – | 0.086 | 3.71 | 21741 |
| ADVERSE_POOLED | 0.941 / 0.75 m | 0.942 / 0.74 m | 0.945 / 0.76 m | – | 0.109 | 4.03 | 69657 |

## Per-object (ours) — α = 0.2 (nominal coverage 0.80), CRC λ̂ = 0.279

| condition | vanilla cov / radius | conflict | conflict + Mondrian | weighted conflict | FNR (target ≤ 0.2) | FP/frame | n objects |
|---|---|---|---|---|---|---|---|
| clear0 | 0.875 / 0.40 m | 0.875 / 0.42 m | 0.873 / 0.42 m | – | 0.112 | 0.75 | 7579 |
| fog1 | 0.811 / 0.40 m | 0.828 / 0.47 m | 0.825 / 0.47 m | 1.000 / inf m (ESS 1606) | 0.140 | 1.09 | 7327 |
| fog2 | 0.776 / 0.40 m | 0.821 / 0.51 m | 0.816 / 0.51 m | 1.000 / inf m (ESS 2424) | 0.202 | 1.18 | 6666 |
| fog3 | 0.658 / 0.40 m | 0.738 / 0.54 m | 0.722 / 0.53 m | 1.000 / inf m (ESS 2952) | 0.490 | 0.79 | 4002 |
| rain1 | 0.877 / 0.40 m | 0.880 / 0.44 m | 0.877 / 0.44 m | 1.000 / 0.94 m (ESS 396) | 0.119 | 0.77 | 7516 |
| rain2 | 0.872 / 0.40 m | 0.882 / 0.45 m | 0.877 / 0.45 m | 1.000 / inf m (ESS 825) | 0.122 | 0.92 | 7499 |
| rain3 | 0.859 / 0.40 m | 0.880 / 0.48 m | 0.873 / 0.48 m | 1.000 / inf m (ESS 872) | 0.133 | 0.96 | 7371 |
| snow1 | 0.857 / 0.40 m | 0.867 / 0.43 m | 0.864 / 0.43 m | 0.986 / 0.71 m (ESS 197) | 0.120 | 0.90 | 7529 |
| snow2 | 0.847 / 0.40 m | 0.856 / 0.44 m | 0.854 / 0.44 m | 1.000 / 1.35 m (ESS 344) | 0.139 | 0.97 | 7301 |
| snow3 | 0.832 / 0.40 m | 0.845 / 0.45 m | 0.842 / 0.45 m | 1.000 / inf m (ESS 533) | 0.172 | 0.84 | 6911 |
| night2 | 0.871 / 0.40 m | 0.881 / 0.46 m | 0.877 / 0.45 m | 1.000 / inf m (ESS 1662) | 0.118 | 0.65 | 7535 |
| SNOW_POOLED | 0.846 / 0.40 m | 0.856 / 0.44 m | 0.854 / 0.44 m | – | 0.144 | 0.90 | 21741 |
| ADVERSE_POOLED | 0.835 / 0.40 m | 0.854 / 0.46 m | 0.849 / 0.46 m | – | 0.176 | 0.91 | 69657 |

**H3 (per-object arm, α = 0.1):** {'snow_vanilla_coverage': 0.9512901890437422, 'snow_conflict_coverage': 0.9483004461616301, 'snow_fnr_at_alpha_0.1': 0.08598951464805124, 'refuted': False, 'conflict_closes_half_gap': False, 'clear_radius_ratio_conflict_over_vanilla': 0.8934627301591104}


---
<!-- exp5_cadc/exp5_cadc.md -->
# Exp 5 — Boreas-trained heads on CADC real snow (radar removed)

7000 frames, 23145 GT cars (≥ 25 pts, ROI). No retraining, no recalibration: models and τ come from Boreas.

| arm | Car mAP (95 % CI) | 2018_03_06 | 2018_03_07 | 2019_02_27 |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.740 (0.733–0.746) | 0.828 | 0.586 | 0.728 |
| Fixed weights | 0.742 (0.735–0.748) | 0.831 | 0.588 | 0.730 |
| Confidence-only | 0.743 (0.737–0.750) | 0.832 | 0.601 | 0.731 |
| Per-frame weather classifier | 0.739 (0.733–0.746) | 0.829 | 0.581 | 0.728 |
| Per-frame oracle weather | 0.736 (0.729–0.742) | 0.823 | 0.576 | 0.724 |
| Per-frame features | 0.739 (0.733–0.746) | 0.832 | 0.576 | 0.727 |
| Per-object (ours) | 0.734 (0.728–0.741) | 0.820 | 0.592 | 0.723 |

Per-object − per-frame classifier: Δ mAP = -0.0050, one-sided p = 1.000.

## Status-quo fusion with a sensor removed (AP-level harm on real snow)

| all (camera + LiDAR) | − camera (LiDAR only) | − LiDAR (camera only) |
|---|---|---|
| 0.740 | 0.705 | 0.078 |

## Decision-flip harm on real snow (status quo, Car)

|  | pooled | camera | lidar |
|---|---|---|---|
| harm % | 3.78 (3.68–3.88) | 5.03 (4.86–5.22) | 3.08 (2.98–3.19) |
| help % | 8.36 | 1.93 | 11.91 |


---
<!-- exp7_dropout/exp7_dropout.md -->
# Exp 7 — Whole-sensor failure at test time (Car mAP, test scenes)


## clear0

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.800 | 0.780 | 0.186 | 0.795 |
| Fixed weights | 0.807 | 0.783 | 0.216 | 0.801 |
| Per-frame weather classifier | 0.804 | 0.783 | 0.196 | 0.798 |
| Per-object (ours) | 0.865 | 0.851 | 0.293 | 0.798 |
## fog3

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.303 | 0.292 | 0.029 | 0.292 |
| Fixed weights | 0.310 | 0.295 | 0.031 | 0.297 |
| Per-frame weather classifier | 0.314 | 0.297 | 0.030 | 0.298 |
| Per-object (ours) | 0.404 | 0.387 | 0.089 | 0.301 |
## snow3

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.698 | 0.669 | 0.154 | 0.690 |
| Fixed weights | 0.709 | 0.674 | 0.178 | 0.697 |
| Per-frame weather classifier | 0.703 | 0.673 | 0.168 | 0.692 |
| Per-object (ours) | 0.785 | 0.763 | 0.242 | 0.699 |
## night2

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.797 | 0.780 | 0.121 | 0.792 |
| Fixed weights | 0.805 | 0.783 | 0.145 | 0.799 |
| Per-frame weather classifier | 0.802 | 0.780 | 0.169 | 0.797 |
| Per-object (ours) | 0.855 | 0.858 | 0.215 | 0.797 |

---
<!-- exp8_qualitative/exp8_qualitative.md -->
# Exp 8 — Qualitative cases

Adverse test candidates with ≥ 2 sensors: per-object arm fixes the status quo in 18714 cases and breaks it in 5464 cases.

## win

![win](win_1598988320569023_fog3.png)

```
{'kind': 'win', 'frame': {'frame_id': '1598988320569023', 'scene': 22, 'split': 'test', 'condition': 'fog', 'severity': 3}, 'evidence_gain': {'lidar': 0.31, 'radar': 0.01}, 'range_share': {'lidar': 1.0, 'radar': 0.0}, 'score_status_quo': 0.01703999750316143, 'score_object': 0.9876018166542053, 'gt_backed': True, 'range_m': 11.933473587036133, 'harm_prob': [0.0, 0.0, 0.00800000037997961], 'figure': 'win_1598988320569023_fog3.png'}
```

## failure

![failure](failure_1598990020898242_fog1.png)

```
{'kind': 'failure', 'frame': {'frame_id': '1598990020898242', 'scene': 37, 'split': 'test', 'condition': 'fog', 'severity': 1}, 'evidence_gain': {'lidar': 0.3, 'radar': 0.01}, 'range_share': {'lidar': 1.0, 'radar': 0.0}, 'score_status_quo': 0.02632170543074608, 'score_object': 0.9753339290618896, 'gt_backed': False, 'range_m': 22.29705047607422, 'harm_prob': [0.0, 0.0, 0.0020000000949949026], 'figure': 'failure_1598990020898242_fog1.png'}
```



---
<!-- exp6_raf/exp6_raf.md -->
# Exp 6 — Positioning against RAF (cited numbers; different dataset, not a head-to-head)

> Numbers transcribed from Park, Jeong & Yoon, *RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR*, arXiv 2607.04587 (Tables 1–2), K-Radar, Sedan class, IoU 0.5. **Verify every digit against the PDF before citing** — they were read from the rendered page. They are on K-Radar with 4D radar and are *not* comparable to our Boreas numbers; the table documents what RAF reports, not a ranking.

## RAF Table 1 — AP_BEV

| method | Normal | Overcast | Fog | Rain | Sleet | Lightsnow | Heavysnow | Total |
|---|---|---|---|---|---|---|---|---|
| L4DR (Baseline, L+R) | 76.8 | 88.6 | 89.7 | 78.2 | 59.3 | 80.9 | 53.8 | 77.5 |
| L4DR (Joint, C+L+R) | 77.0 | 80.8 | 89.4 | 78.4 | 59.9 | 81.1 | 53.2 | 77.8 |
| L4DR (Frozen, C+L+R) | 79.8 | 83.5 | 92.2 | 79.1 | 59.2 | 85.7 | 58.4 | 78.6 |
| L4DR (RAF, C+L+R) | 82.0 | 88.2 | 93.9 | 82.6 | 63.0 | 88.5 | 60.4 | 82.0 |
| 3D-LRF (Baseline, L+R) | 62.1 | 88.5 | 84.8 | 69.9 | 37.7 | 84.6 | 36.0 | 64.5 |
| 3D-LRF (Joint, C+L+R) | 70.2 | 81.9 | 75.8 | 74.5 | 18.4 | 74.2 | 40.9 | 65.7 |
| 3D-LRF (Frozen, C+L+R) | 67.2 | 87.8 | 81.6 | 78.3 | 42.4 | 80.1 | 48.6 | 68.3 |
| 3D-LRF (RAF, C+L+R) | 70.8 | 88.0 | 82.3 | 74.3 | 43.3 | 84.3 | 49.6 | 71.0 |

## RAF Table 2 — AP_BEV by camera visibility (Clean / Mixed / Noisy)

| method | Clean | Mixed | Noisy |
|---|---|---|---|
| L4DR Baseline (L+R) | 77.0 | 78.4 | 59.0 |
| L4DR Joint | 81.5 | 77.5 | 55.6 |
| L4DR Frozen | 80.2 | 78.1 | 55.6 |
| L4DR RAF | 80.4 | 82.1 | 59.0 |
| 3D-LRF Baseline (L+R) | 68.8 | 73.0 | 38.8 |
| 3D-LRF Joint | 69.8 | 68.8 | 25.4 |
| 3D-LRF Frozen | 70.4 | 65.5 | 30.8 |
| 3D-LRF RAF | 72.2 | 74.5 | 38.4 |

**Harm evidence inside RAF's own numbers.** 3D-LRF on *Noisy*: L+R baseline 38.8 → C+L+R Joint 25.4 (−13.4 AP_BEV). Sleet: 37.7 → 18.4. Adding the camera without reliability handling hurts — the aggregate form of our H1. RAF recovers to 38.4 (Noisy), i.e. it *neutralises* camera harm; it does not report per-object harm rates or predict harm.

## Property comparison

| property | RAF (2026) | this work |
|---|---|---|
| fusion level | BEV feature fusion (camera branch + BEV fusion encoder + head trained) | output-level late fusion over frozen detectors |
| what is trained | camera backbone, projectors, reliability CNN, fusion encoder, detection head (Swin-T; RTX 3090) | ~10⁴-parameter router on cached detections (CPU, minutes) |
| retrofit to an existing stack | requires re-training the fusion network per backbone | any detectors that emit boxes + scores |
| which modalities are arbitrated | camera only (LiDAR–RADAR assumed reliable reference) | camera, LiDAR and radar symmetrically |
| granularity | per image pixel | per object (plus frame context) |
| supervision of reliability | image-level Clean/Mixed/Noisy labels → cross-modal similarity pseudo-labels | counterfactual harm labels from GT (no weather labels needed) |
| negative evidence ('should have seen it') | no | yes — absence × field-of-view term |
| formal guarantee on output | none | split-conformal localisation discs + conformal risk control of FNR |
| datasets | K-Radar, VoD | Boreas (+ graded simulation), CADC |