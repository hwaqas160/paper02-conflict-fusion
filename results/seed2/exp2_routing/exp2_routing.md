# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.801 | 0.720 | 0.620 | 0.308 | 0.783 | 0.774 | 0.750 | 0.763 | 0.733 | 0.703 | 0.797 | 0.675 | 0.733 | 0.690 |
| Fixed weights | 19 | 0.805 | 0.728 | 0.633 | 0.311 | 0.788 | 0.782 | 0.762 | 0.766 | 0.736 | 0.709 | 0.805 | 0.684 | 0.737 | 0.698 |
| Confidence-only | 10579 | 0.816 | 0.739 | 0.644 | 0.329 | 0.798 | 0.790 | 0.769 | 0.778 | 0.747 | 0.719 | 0.809 | 0.696 | 0.748 | 0.710 |
| Per-frame weather classifier | 10195 | 0.799 | 0.735 | 0.641 | 0.317 | 0.781 | 0.775 | 0.753 | 0.758 | 0.727 | 0.697 | 0.797 | 0.684 | 0.727 | 0.696 |
| Per-frame oracle weather | 10259 | 0.797 | 0.725 | 0.639 | 0.318 | 0.777 | 0.772 | 0.754 | 0.751 | 0.720 | 0.692 | 0.800 | 0.685 | 0.721 | 0.693 |
| Per-frame features | 10643 | 0.793 | 0.721 | 0.625 | 0.317 | 0.777 | 0.772 | 0.754 | 0.759 | 0.735 | 0.711 | 0.793 | 0.681 | 0.735 | 0.697 |
| Per-object (ours) | 18195 | 0.873 | 0.813 | 0.722 | 0.413 | 0.861 | 0.853 | 0.836 | 0.847 | 0.819 | 0.785 | 0.864 | 0.768 | 0.818 | 0.783 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.609 | 0.598 | 0.613 |
| Confidence-only | 0.988 | 0.988 | 0.989 |
| Per-frame weather classifier | 0.615 | 0.594 | 0.619 |
| Per-frame oracle weather | 0.624 | 0.598 | 0.634 |
| Per-frame features | 0.650 | 0.649 | 0.643 |
| Per-object (ours) | 0.982 | 0.980 | 0.982 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0866 | 0.001 | 0.0854 | 0.001 | 0.0723 | 0.001 | 0.982 | 0.615 | 0.988 |
| HELDOUT | 0.0906 | 0.001 | 0.0828 | 0.001 | 0.0698 | 0.001 | 0.980 | 0.594 | 0.988 |
| SEEN_ADVERSE | 0.0845 | 0.001 | 0.0870 | 0.001 | 0.0727 | 0.001 | 0.982 | 0.619 | 0.989 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992
