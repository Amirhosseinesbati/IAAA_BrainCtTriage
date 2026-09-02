# Exp20 preregistered plan: third-fold dual-selector replication

## Why this run is justified

The same narrow component hypothesis has now passed two held-out folds without
post-hoc retuning:

- fold1: 90% Exp09 plus 10% Exp18/epoch21 `mls_mm` improved MAE by
  `0.010951099 mm`, Boundary-F1 by `0.007305669` and objective by
  `0.025562438`;
- fold0: 90% Exp16 plus 10% Exp19/epoch21 `mls_mm` improved MAE by
  `0.021776801 mm`, Boundary-F1 by `0.009176587` and objective by
  `0.040129976` in the frozen independent test.

Both results use the baseline model's selector, peak/ranking probability and
heatmap. Only the regression output is blended, and the challenger weight is
fixed at 0.10. A third fold is required before this can be treated as a general
three-fold deployment recipe rather than two positive fold-specific results.

## Single training factor

Exp20 copies the normalized Exp19 training recipe exactly and changes only the
held-out competition fold from 0 to 2. It retains HRNet-W32, three 512px input
channels, the dual presence/peak selector, normalized selector-loss scale,
official 3484-row data contract, balanced sampler, seed 42, strict CUDA
determinism, all losses and augmentations, 23 epochs and snapshots
13/15/17/19/21/23. There is no warm start and no CPU model fallback.

## Frozen primary fold2 transfer test

This test is fixed before Exp20 training:

1. Baseline is Exp15r `epoch017`, the promoted fold2 candidate.
2. Exp20 `epoch21` is the only primary challenger checkpoint.
3. Exp15r supplies selector, peak/ranking probability and heatmap.
4. Only `mls_mm` is blended as `0.90 * Exp15r + 0.10 * Exp20`.
5. Production pooling remains `severity_window`, radius 3, selector gate 0.5,
   minimum three active slices, probability-weighted q0.75 and no heatmap
   guard.
6. No alpha, checkpoint, offset, threshold or pooling retune is allowed.

The authoritative same-runtime Exp15r baseline over 67 studies is:

- MAE `1.5483543317709396 mm`;
- Boundary-F1 `0.8925925925925926`;
- objective `1.7631691465857544`.

Primary replication passes only if all are true:

- hybrid MAE is no worse than `1.5483543317709396 mm`;
- hybrid Boundary-F1 is no worse than `0.8925925925925926`;
- hybrid objective is at most `1.7531691465857544`, an improvement of at
  least 0.01;
- all 67 fold2 studies complete CUDA inference with zero failures.

No other checkpoint can rescue the primary decision. Other snapshots may be
used only after the frozen result for failure analysis.

## Decision and resource rule

Peak VRAM for Exp18 and Exp19 was 4.648GB on this RTX3060, so the run fits the
12GB device. The complete third-fold experiment is justified by two positive,
mechanistically identical held-out results; it is not a seed or alpha sweep.

If the frozen fold2 gate passes, the 90/10 regression-only recipe has replicated
on all three competition folds and may advance to two-model packaged-runtime
parity, latency/memory validation and then a limited leaderboard submission. If
it fails, the general three-fold recipe is rejected; no post-hoc checkpoint or
alpha substitution may turn the primary result into a pass.

All model computation is CUDA-only. Aggregate metrics and reports go to MLflow;
raw medical data and per-study prediction CSVs do not. The Vast instance remains
active and is never stopped or destroyed without user coordination.
