# Three-fold W32 MLS pooling analysis

This analysis combines the frozen out-of-fold predictions of W32 fold 0 and
fold 1 with the selector-AUC checkpoint from fold 2. It performs no model
inference and does not upload per-study or per-slice predictions.

## Robust profile across all three folds

- Family: `anchor_window`
- Anchor radius: `2` (five contiguous slices at most)
- Absolute selector gate: `0.9`
- Minimum active slices: `3`
- Aggregation quantile: `0.9`
- Probability weighting: `false`
- Heatmap guard ratio: `0.5`

| Fold | MAE mm | RMSE mm | Bias mm | F1@3 | F1@5 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.4337 | 2.6810 | -0.3869 | 0.8333 | 0.7805 |
| 1 | 1.4791 | 2.4282 | -0.4438 | 0.7059 | 0.8000 |
| 2 | 2.0224 | 3.3229 | -0.3197 | 0.8214 | 0.8696 |
| Mean | 1.6451 | 2.8107 | -0.3835 | — | — |

The previous two-fold `joint_component` profile obtains mean MAE `1.6691 mm`
when the AUC checkpoint is used on fold 2. The contiguous anchor-window
profile therefore improves mean MAE by `0.0241 mm` and worst-fold MAE by
`0.0813 mm`; the gain is small but directionally consistent with the anatomy.

## Generalization warning

The joint all-fold ranking is diagnostic because fold 2 participated in the
comparison. Leave-one-fold-out transfer gives held-fold MAEs of `1.5455`,
`1.6726`, and `2.1037 mm`. Fold 2 is materially harder, so the old two-fold
mean of `1.4519 mm` was optimistic. Future architecture decisions should use
fold-transfer or full OOF estimates and must not quote the best in-sample grid
row as untouched performance.

Raw profile metrics remain in `crossfold_pooling_grid.csv`; the compact audit
record is `crossfold_pooling_summary.json`.
