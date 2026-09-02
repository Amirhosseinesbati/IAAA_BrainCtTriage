# Exp79 — BF16-exact factorized hierarchy four-step smoke

## Locked purpose

Exp79 repeats Exp77 Phase A after the sole Exp78 numerical correction. It tests
the real training/evaluation pipeline and epoch-zero metric identity; it is not a
hyperparameter experiment. The partial-update performance may not alter the
already locked full calibration recipe.

## Locked recipe

- Same Exp61 checkpoint SHA, schema4 manifest SHA, outer fold `2`, calibration
  fold `1`, seed `42`, `unetplusplus/tu-efficientnetv2_rw_s`, 384 resolution.
- Factorized output head; encoder and auxiliary classifier frozen; decoder,
  legacy segmentation head and both residual heads trainable with frozen decoder
  BatchNorm running statistics.
- Hierarchical weights: foreground Dice `0.325`, foreground focal `0.175`,
  conditional subtype Dice `0.325`, conditional class-weighted focal CE `0.175`
  with gamma `2`; OVR and classification losses zero.
- Pixel class-weight power `1`, cap `8`; hard-empty penalty `0.05` on top
  `0.001`; `fpr_volume_penalized` checkpoint selection.
- AdamW `5e-5`, weight decay `1e-4`, cosine schedule, batch `16`, workers `4`,
  BF16, at most four optimizer steps, no outer evaluation and no pretrained
  download.
- The pipeline must persist a dedicated epoch-zero aggregate summary so the gate
  never depends on CSV precision or whichever epoch becomes the best checkpoint.

## Preregistered gates

- Exactly one epoch-zero row and at least one partial-update row.
- Epoch-zero selection, Dice, Any AUC, macro subtype AUC, total-volume MAE and
  bias differ from Exp61 by at most `1e-6`; FPR and F1 are exactly equal using the
  lossless JSON summaries.
- All required primary metrics and all observed train-loss components finite;
  missing metrics for volume strata with zero positive studies are ignored.
- Peak VRAM below 20 GiB, checkpoint/aggregate artifacts written, four-step
  config and factorized scope preserved, outer evaluation absent.
- Evaluator decision must be
  `authorize_locked_three_epoch_calibration_screen` and be attached to MLflow.

Passing authorizes Exp80: the same already locked three-epoch calibration-only
recipe and original promotion gates from Exp77. Failure requires another causal
diagnosis; no gate may be relaxed after observing the result.
