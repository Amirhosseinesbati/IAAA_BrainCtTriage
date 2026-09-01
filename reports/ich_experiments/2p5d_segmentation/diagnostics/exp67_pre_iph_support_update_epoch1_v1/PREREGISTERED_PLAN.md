# Exp67 pre-calibration train-only IPH→SAH selectivity probe

## Question

Can the frozen 3,217-parameter SAH residual head recover true-SAH pixels that
the audited exp61 checkpoint predicts as IPH, without materially relabelling
correct true-IPH pixels or increasing background pressure?

The preceding calibration-only margin audit found a real opportunity
(36.37% of true-SAH pixels were predicted as IPH) but also a large theoretical
risk. A uniformly saturated cap of 8 could reach only 16.07% of those missed
SAH pixels while exposing 6.82% of correctly predicted true-IPH pixels. This
probe therefore tests learned selectivity before any further calibration use.

## Locked recipe

- Data scope: train folds 0/3/4 only; calibration fold 1 and outer fold 2 are
  not loaded for evaluation.
- Base: audited exp61 checkpoint and schema4 manifest, verified by SHA-256.
- Adapter support: incumbent argmax background or IPH only.
- Trainable parameters: residual head only (expected 3,217).
- Initialization: zero final convolution; exact exp61 identity before update.
- Optimizer: AdamW, one complete train epoch, lr=5e-4, weight decay=1e-4.
- Objective: unchanged main segmentation loss plus true-SAH-pixel NLL weight
  0.03; legacy positive-row SAH Tversky weight 0.
- Residual cap: 8 logits. No cap/weight/epoch sweep is allowed.
- Probe: 12 SAH-positive batches, 12 SAH-negative IPH-control batches and 12
  SAH/ICH-negative background-control batches from a separately seeded train
  loader.
- Persistence: aggregate JSON and MLflow metrics only; no row-level medical
  predictions and no checkpoint promotion.

## Preregistered gates

All four conditions must pass:

1. At least 5% of eligible true-SAH pixels predicted as IPH convert to SAH.
2. At most 0.1% of correctly predicted true-IPH pixels convert to SAH.
3. At least 50% of all incumbent-IPH→SAH conversions are true-SAH pixels.
4. At most 0.01% of eligible true-background pixels convert to SAH.

Passing only authorizes one locked calibration-only experiment. It is not a
promotion, OOF, outer-fold or leaderboard claim. Failure closes the frozen
background/IPH relabel branch; the next architecture must be an independently
supervised multi-label or two-stage subtype head with patient-safe evaluation.
