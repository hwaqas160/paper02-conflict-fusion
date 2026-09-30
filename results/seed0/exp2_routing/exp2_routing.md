# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.800 | 0.718 | 0.618 | 0.303 | 0.782 | 0.774 | 0.750 | 0.760 | 0.729 | 0.698 | 0.797 | 0.674 | 0.729 | 0.688 |
| Fixed weights | 19 | 0.807 | 0.730 | 0.635 | 0.310 | 0.791 | 0.784 | 0.764 | 0.768 | 0.739 | 0.709 | 0.805 | 0.686 | 0.738 | 0.700 |
| Confidence-only | 10579 | 0.815 | 0.736 | 0.640 | 0.326 | 0.798 | 0.789 | 0.766 | 0.775 | 0.743 | 0.714 | 0.807 | 0.693 | 0.744 | 0.707 |
| Per-frame weather classifier | 10195 | 0.804 | 0.735 | 0.641 | 0.314 | 0.787 | 0.780 | 0.759 | 0.763 | 0.732 | 0.703 | 0.802 | 0.687 | 0.733 | 0.700 |
| Per-frame oracle weather | 10259 | 0.803 | 0.725 | 0.642 | 0.317 | 0.785 | 0.778 | 0.761 | 0.767 | 0.739 | 0.709 | 0.802 | 0.690 | 0.738 | 0.702 |
| Per-frame features | 10643 | 0.792 | 0.712 | 0.621 | 0.310 | 0.774 | 0.768 | 0.756 | 0.751 | 0.723 | 0.695 | 0.788 | 0.677 | 0.722 | 0.690 |
| Per-object (ours) | 18195 | 0.865 | 0.804 | 0.709 | 0.397 | 0.855 | 0.845 | 0.828 | 0.846 | 0.817 | 0.783 | 0.853 | 0.759 | 0.815 | 0.776 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.618 | 0.605 | 0.623 |
| Confidence-only | 0.988 | 0.988 | 0.988 |
| Per-frame weather classifier | 0.623 | 0.609 | 0.630 |
| Per-frame oracle weather | 0.657 | 0.662 | 0.644 |
| Per-frame features | 0.660 | 0.661 | 0.651 |
| Per-object (ours) | 0.984 | 0.982 | 0.985 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0761 | 0.001 | 0.0858 | 0.001 | 0.0687 | 0.001 | 0.984 | 0.623 | 0.988 |
| HELDOUT | 0.0823 | 0.001 | 0.0926 | 0.001 | 0.0711 | 0.001 | 0.982 | 0.609 | 0.988 |
| SEEN_ADVERSE | 0.0717 | 0.001 | 0.0819 | 0.001 | 0.0658 | 0.001 | 0.985 | 0.630 | 0.988 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992
