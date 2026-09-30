# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.790 | 0.710 | 0.615 | 0.303 | 0.772 | 0.765 | 0.742 | 0.751 | 0.721 | 0.693 | 0.790 | 0.667 | 0.721 | 0.680 |
| Fixed weights | 19 | 0.806 | 0.730 | 0.634 | 0.311 | 0.790 | 0.783 | 0.763 | 0.768 | 0.738 | 0.710 | 0.805 | 0.685 | 0.738 | 0.699 |
| Confidence-only | 10579 | 0.814 | 0.737 | 0.643 | 0.326 | 0.799 | 0.790 | 0.769 | 0.777 | 0.747 | 0.720 | 0.809 | 0.695 | 0.748 | 0.710 |
| Per-frame weather classifier | 10195 | 0.802 | 0.735 | 0.641 | 0.317 | 0.786 | 0.779 | 0.759 | 0.762 | 0.731 | 0.703 | 0.804 | 0.687 | 0.732 | 0.700 |
| Per-frame oracle weather | 10259 | 0.801 | 0.726 | 0.640 | 0.318 | 0.782 | 0.777 | 0.759 | 0.761 | 0.733 | 0.705 | 0.804 | 0.689 | 0.732 | 0.698 |
| Per-frame features | 10643 | 0.790 | 0.718 | 0.624 | 0.317 | 0.773 | 0.767 | 0.753 | 0.753 | 0.727 | 0.704 | 0.791 | 0.679 | 0.728 | 0.693 |
| Per-object (ours) | 18195 | 0.860 | 0.807 | 0.716 | 0.404 | 0.848 | 0.839 | 0.817 | 0.842 | 0.814 | 0.777 | 0.848 | 0.755 | 0.811 | 0.772 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.662 | 0.674 | 0.651 |
| Confidence-only | 0.988 | 0.989 | 0.988 |
| Per-frame weather classifier | 0.650 | 0.639 | 0.645 |
| Per-frame oracle weather | 0.668 | 0.678 | 0.660 |
| Per-frame features | 0.670 | 0.671 | 0.662 |
| Per-object (ours) | 0.987 | 0.987 | 0.988 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0721 | 0.001 | 0.0787 | 0.001 | 0.0623 | 0.001 | 0.987 | 0.650 | 0.988 |
| HELDOUT | 0.0794 | 0.001 | 0.0838 | 0.001 | 0.0633 | 0.001 | 0.987 | 0.639 | 0.989 |
| SEEN_ADVERSE | 0.0674 | 0.001 | 0.0759 | 0.001 | 0.0600 | 0.001 | 0.988 | 0.645 | 0.988 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992
