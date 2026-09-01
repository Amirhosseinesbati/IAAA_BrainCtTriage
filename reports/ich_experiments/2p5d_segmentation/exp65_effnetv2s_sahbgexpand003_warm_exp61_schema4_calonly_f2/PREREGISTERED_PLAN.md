# Exp65 preregistration — frozen background-to-SAH residual adapter

## Question

Can a small, zero-initialized residual head recover SAH pixels that exp61 calls
background without repeating exp63's SDH/SAH interference or worsening normal
false positives and physical-volume error?

## Evidence before training

- The patient-disjoint training folds 0/3/4 contain 16 SAH-positive studies.
- Calibration fold 1 contains four SAH-positive studies and none is isolated
  SAH, so a flexible head or a hyperparameter sweep is not defensible.
- Exp63 briefly raised SAH Dice from 0.05302 to 0.06556 but reduced SDH Dice
  from 0.38166 to 0.28151 and worsened volume MAE.
- Exp64 reduced total-volume MAE by only 0.13561 mL, excluding hard-volume
  readout as the dominant failure mechanism.
- Generic study balancing p0.50 and p0.75 was previously rejected; exp65 keeps
  `sampler_study_balance_power=0.0`.

## Locked architecture

- Warm start: exp61 checkpoint, same outer=2/calibration=1 split and manifest.
- Base encoder, decoder, segmentation head, classification head and BatchNorm
  are frozen and evaluated without gradients.
- A 16-channel Conv3x3 + GroupNorm + SiLU + Conv1x1 head reads detached decoder
  features and incumbent mask logits.
- The final convolution is initialized to zero.
- A bounded residual (`8*tanh`) is added only to the SAH logit where the
  incumbent argmax is background. Existing SAH and IVH/IPH/SDH/EDH argmax
  pixels cannot change by construction.

## Locked optimization

- epochs=6, patience=2, batch_size=16, lr=5e-4, weight_decay=1e-4.
- classification loss=0 because its frozen logits are exact invariants.
- Existing pixel-basis subtype weighting and hard-empty loss are preserved.
- SAH-only positive Tversky weight=0.03; diffuse SDH/SAH Tversky=0.
- No outer evaluation. First a four-step smoke run, then one calibration-only
  run. No temperature, threshold, loss-weight or architecture sweep is allowed.

## Promotion gate

All conditions must pass simultaneously:

1. Best checkpoint epoch is at least one and outer was not evaluated.
2. Warm-start and manifest SHA256 provenance match exp61.
3. SAH Dice improves by at least 0.01 absolute.
4. SAH MAE improves by at least 0.10 mL.
5. FPR, presence F1, total-volume MAE and absolute bias do not worsen.
6. EDH/IPH/IVH/SDH Dice, AUC, MAE and bias are exact invariants to 1e-10.
7. Any-ICH and macro-subtype AUC are exact invariants to 1e-10.
8. FPR/volume-penalized checkpoint score improves by at least 0.001.

Passing authorizes a locked patient-disjoint OOF experiment. It is not final
promotion or leaderboard evidence. Any failed condition yields
`reject_before_outer` and the checkpoint is not copied into `checkpoint/ich`.
