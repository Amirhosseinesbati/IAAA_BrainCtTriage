# Exp82 — factorized loss calibration-attribution probe

## Rationale fixed before execution

Exp81 found positive short-horizon SAH/SDH soft-Dice drift on held-aside train
batches even though Exp79/80 showed immediate calibration collapse. The large
train-to-calibration gap means train-gradient attribution alone is insufficient.
Exp82 therefore repeats the exact four-update Exp79 horizon with component
ablations and evaluates only the already-designated calibration fold.

## Locked scope

- Same Exp61 checkpoint and Schema4 manifest hashes as Exp80/81.
- Patient-safe outer fold `2` remains completely untouched; calibration fold is
  `1`; seed `42`.
- Same first four sampler batches, BF16, batch `16`, AdamW `5e-5`, weight decay
  `1e-4`, and zero-residual factorized initialization for every variant.
- Variants: `full_exp80`, `without_conditional_dice`,
  `without_conditional_focal`, and `foreground_only`.
- Full calibration aggregate metrics are computed after four updates. Slice- or
  study-level predictions remain in memory and are never saved or logged.
- Diagnostic only: no outer/OOF inference, checkpoint promotion, or model file.
- The first execution is `local-only`: the aggregate JSON stays in the project
  workspace and no derived metric is sent to MLflow/DagsHub or Telegram unless
  the user separately authorizes those exact external destinations.

## Locked attribution rule

The Exp79 failure is considered reproduced if the full objective loses at least
`0.01` mean foreground Dice on calibration. Removing a conditional component
identifies it as a primary suspect only if, relative to the full objective, it
rescues both mean Dice by at least `0.01` and mean SAH/SDH Dice by at least
`0.02`, and its mean-Dice rescue exceeds the other removal by at least `0.005`.
If both removals pass their rescue thresholds, classify an interaction. If
foreground-only still loses at least `0.01` Dice, classify shared
foreground/decoder pressure. Otherwise attribution is inconclusive.

This diagnostic may select the structure of a subsequent preregistered
calibration recipe. It cannot itself promote a checkpoint or authorize outer.
