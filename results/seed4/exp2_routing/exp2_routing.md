# Exp 2 — Per-object vs per-frame vs fixed routing (Car mAP, test scenes)

| arm | params | clear0 | fog1 | fog2 | fog3 | rain1 | rain2 | rain3 | snow1 | snow2 | snow3 | night2 | SEEN_ADVERSE | HELDOUT | ADVERSE_POOLED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 19 | 0.800 | 0.719 | 0.619 | 0.307 | 0.783 | 0.774 | 0.751 | 0.761 | 0.731 | 0.701 | 0.798 | 0.675 | 0.731 | 0.689 |
| Fixed weights | 19 | 0.807 | 0.731 | 0.635 | 0.314 | 0.790 | 0.783 | 0.762 | 0.770 | 0.740 | 0.712 | 0.806 | 0.686 | 0.740 | 0.700 |
| Confidence-only | 10579 | 0.814 | 0.737 | 0.643 | 0.330 | 0.796 | 0.788 | 0.765 | 0.775 | 0.745 | 0.718 | 0.809 | 0.694 | 0.746 | 0.709 |
| Per-frame weather classifier | 10195 | 0.802 | 0.737 | 0.641 | 0.319 | 0.785 | 0.776 | 0.754 | 0.764 | 0.732 | 0.703 | 0.804 | 0.687 | 0.733 | 0.699 |
| Per-frame oracle weather | 10259 | 0.802 | 0.727 | 0.641 | 0.321 | 0.782 | 0.776 | 0.759 | 0.769 | 0.740 | 0.711 | 0.804 | 0.689 | 0.740 | 0.703 |
| Per-frame features | 10643 | 0.782 | 0.699 | 0.599 | 0.314 | 0.767 | 0.761 | 0.750 | 0.746 | 0.723 | 0.700 | 0.787 | 0.669 | 0.723 | 0.685 |
| Per-object (ours) | 18195 | 0.866 | 0.804 | 0.714 | 0.405 | 0.856 | 0.843 | 0.822 | 0.848 | 0.821 | 0.787 | 0.859 | 0.759 | 0.819 | 0.777 |

## Harm-prediction AUROC (Car candidates, per present modality)

| arm | ADVERSE_POOLED | HELDOUT | SEEN_ADVERSE |
|---|---|---|---|
| Fixed weights | 0.620 | 0.606 | 0.624 |
| Confidence-only | 0.988 | 0.988 | 0.987 |
| Per-frame weather classifier | 0.642 | 0.603 | 0.645 |
| Per-frame oracle weather | 0.669 | 0.678 | 0.660 |
| Per-frame features | 0.673 | 0.659 | 0.672 |
| Per-object (ours) | 0.986 | 0.985 | 0.986 |

## H2 test (paired frame bootstrap)

| pool | Δ mAP obj − frame_cls | p | Δ mAP obj − frame_feats | p | Δ mAP obj − confidence | p | AUROC obj | AUROC frame_cls | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADVERSE_POOLED | 0.0781 | 0.001 | 0.0927 | 0.001 | 0.0687 | 0.001 | 0.986 | 0.642 | 0.988 |
| HELDOUT | 0.0867 | 0.001 | 0.0968 | 0.001 | 0.0738 | 0.001 | 0.985 | 0.603 | 0.988 |
| SEEN_ADVERSE | 0.0726 | 0.001 | 0.0902 | 0.001 | 0.0651 | 0.001 | 0.986 | 0.645 | 0.987 |

(Confidence-only comparison is reported for completeness; it is not part of the pre-registered H2 rule.)

**H2 refuted:** False

Weather classifier accuracy on test frames of training conditions: 0.992
