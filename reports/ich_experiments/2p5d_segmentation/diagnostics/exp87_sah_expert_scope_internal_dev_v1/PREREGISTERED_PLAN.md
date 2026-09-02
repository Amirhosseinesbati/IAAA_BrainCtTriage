# Exp87 — patient-safe internal-dev SAH expert scope attribution

## Why this is necessary

Exp86 trained the complete copied decoder and binary SAH head for three epochs.
Its training Tversky reached `0.689`, yet calibration ranking collapsed near
chance (`expert raw AUC=0.511`, AP=`0.00387`) and was much worse than the frozen
incumbent raw margin (AUC=`0.615`, AP=`0.03178`). This could be a fundamental
failure of independent SAH representation or an optimization/scope failure from
updating all 2.84M decoder parameters and normalization state.

Exp87 distinguishes these causes without reading calibration or outer images.

## Locked split and execution

- Internal train uses manifest folds `0` and `3`; internal dev uses fold `4`.
- Calibration fold `1` and outer fold `2` are enumerated only for leakage checks
  and are not loaded or inferred.
- Every scope starts independently from the same Exp61 checkpoint and receives
  the same seeded sampler sequence.
- One fixed epoch, batch `8`, BF16, AdamW `1e-4`, weight decay `1e-4`.
- The Exp86 binary focal/Tversky target and loss weights remain unchanged so only
  trainable scope is attributed.
- All modules remain in evaluation mode during optimization; normalization
  running statistics are frozen while selected convolution/affine parameters can
  receive gradients.
- Scopes: binary head only (`145` expected parameters), final Unet++ block
  `x_0_4` plus head (about `7.1k`), and full decoder plus head (about `2.84M`).
- Threshold-free AP/AUC are evaluated before and after the epoch on internal dev.
  No threshold selection, checkpoint, row prediction, external reporting or
  model promotion is allowed.

## Locked scope gate

For any scope to advance, after-training expert metrics on recoverable incumbent
background/IPH pixels must simultaneously exceed the incumbent raw margin by:

1. at least `+0.005` average precision;
2. at least `+0.01` ROC-AUC;
3. at least `+0.01` average precision within a 15-pixel dilation of incumbent
   foreground.

Batch-identity hashes must also match across all scopes. If multiple scopes pass,
the highest pooled AP is selected. Passing authorizes a new fixed training run on
folds `0,3,4` with one final calibration screen. If none pass, the independent
SAH expert scope branch is closed; calibration and outer remain untouched.
