# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.799 | 0.718 | 0.619 | 0.306 | 0.782 | 0.773 | 0.750 | 0.760 | 0.729 | 0.700 | 0.797 | 0.674 | 0.730 | 0.688 |
| Fixed weights | 19 | 0.807 | 0.731 | 0.635 | 0.315 | 0.790 | 0.783 | 0.761 | 0.770 | 0.740 | 0.712 | 0.806 | 0.686 | 0.741 | 0.700 |
| Confidence-only | 10579 | 0.814 | 0.735 | 0.643 | 0.324 | 0.798 | 0.789 | 0.767 | 0.774 | 0.745 | 0.719 | 0.809 | 0.694 | 0.746 | 0.708 |
| Per-frame weather classifier | 10195 | 0.801 | 0.736 | 0.643 | 0.318 | 0.788 | 0.781 | 0.760 | 0.761 | 0.732 | 0.706 | 0.806 | 0.689 | 0.733 | 0.702 |
| Per-frame oracle weather | 10259 | 0.800 | 0.730 | 0.641 | 0.317 | 0.782 | 0.779 | 0.763 | 0.765 | 0.736 | 0.708 | 0.805 | 0.690 | 0.736 | 0.701 |
| Per-frame features | 10643 | 0.781 | 0.705 | 0.613 | 0.313 | 0.764 | 0.761 | 0.747 | 0.742 | 0.719 | 0.699 | 0.783 | 0.671 | 0.720 | 0.685 |
| Per-object (ours) | 18195 | 0.861 | 0.804 | 0.712 | 0.404 | 0.851 | 0.844 | 0.826 | 0.840 | 0.813 | 0.782 | 0.860 | 0.760 | 0.812 | 0.776 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.621 | 0.614 | 0.621 |
| Confidence-only | 0.990 | 0.990 | 0.990 |
| Per-frame weather classifier | 0.637 | 0.619 | 0.635 |
| Per-frame oracle weather | 0.640 | 0.618 | 0.649 |
| Per-frame features | 0.672 | 0.662 | 0.668 |
| Per-object (ours) | 0.984 | 0.983 | 0.985 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0741 | 0.001 | 0.0908 | 0.001 | 0.0674 | 0.001 | 0.984 | 0.637 | 0.990 |
| HELDOUT | 0.0793 | 0.001 | 0.0924 | 0.001 | 0.0664 | 0.001 | 0.983 | 0.619 | 0.990 |
| SEEN_ADVERSE | 0.0709 | 0.001 | 0.0891 | 0.001 | 0.0661 | 0.001 | 0.985 | 0.635 | 0.990 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992
