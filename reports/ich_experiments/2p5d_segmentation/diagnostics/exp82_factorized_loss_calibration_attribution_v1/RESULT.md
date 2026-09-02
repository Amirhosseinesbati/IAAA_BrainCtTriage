# Exp82 result — shared spatial representation caused the early collapse

## Decision

`shared_foreground_or_decoder_pressure_primary_suspect`

The exact four-update Exp79/80 calibration failure was reproduced. Removing
either conditional subtype component did not meet the preregistered rescue
thresholds, while the foreground-only update still caused a large held-out
decline. The next experiment must freeze the shared decoder and legacy
segmentation head before changing conditional loss weights.

## Reproducibility and safety

- Execution commit: `add56b324f8a2c9b43f41f2c512176171b401670`
- Exp61 checkpoint SHA-256:
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`
- Schema4 manifest SHA-256:
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`
- Aggregate result SHA-256:
  `c9e4c9150c488eb55395a67d1f753ac5354e3b24f9080303d1c78cea7b8b8c02`
- Four BF16 AdamW updates, batch 16, learning rate `5e-5`, identical ordered
  batches and zero-residual initialization for every variant.
- Local-only execution: no metric or artifact was sent to DagsHub/MLflow or
  Telegram; no row-level predictions were persisted.
- Outer fold 2 was not read, and no model/checkpoint was written or promoted.

## Aggregate calibration results after four updates

| Variant | Checkpoint score | Selection | Mean Dice | SAH | SDH | EDH | Volume MAE (mL) | Bias (mL) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp61 identity | 0.586668 | 0.666162 | 0.459106 | 0.053024 | 0.381665 | 0.537976 | 10.7627 | -6.2364 |
| Full Exp80 | 0.555336 | 0.646386 | 0.423149 | 0.074996 | 0.242223 | 0.485361 | 12.5135 | -9.0380 |
| Without conditional Dice | 0.552654 | 0.645659 | 0.421827 | 0.070294 | 0.236767 | 0.479577 | 12.8104 | -9.5091 |
| Without conditional focal | 0.561404 | 0.650226 | 0.430130 | 0.075038 | 0.264554 | 0.501607 | 12.2158 | -8.2987 |
| Foreground only | 0.559806 | 0.650048 | 0.429807 | 0.051512 | 0.268233 | 0.495560 | 12.3994 | -8.8006 |

The full path lost `0.03596` mean Dice and `0.03133` checkpoint score. Its mean
SAH/SDH Dice drift was `-0.05874`. Removing conditional focal rescued only
`0.00698` mean Dice and `0.01119` diffuse mean Dice, below the locked `0.01` and
`0.02` thresholds. Removing conditional Dice slightly worsened both measures.

Most decisively, foreground-only updates still lost `0.02930` mean Dice,
reduced SDH by `0.11343`, reduced EDH by `0.04242`, increased volume MAE by
`1.64 mL`, and moved total bias another `-2.56 mL`. IVH/IPH gained only
`0.00720`/`0.00366` Dice. Thus the failure does not require either conditional
subtype loss.

## Interpretation

The algebraic output factorization separates foreground and conditional subtype
residual logits, but Exp80 also trained the shared decoder and legacy six-class
segmentation head. Foreground loss can therefore change the shared features and
legacy subtype logits before the factorized residual composition. This hidden
coupling explains why a foreground-only objective altered SDH/EDH morphology
and volume despite the nominally separated output branches.

The next technical hypothesis is residual-head-only factorized training:
freeze encoder, classifier, decoder and legacy segmentation head, and train only
the zero-initialized one-channel foreground residual and five-channel centered
subtype residual. A four-update calibration gate must first prove that this
removes the shared-representation collapse before any three-epoch run.
