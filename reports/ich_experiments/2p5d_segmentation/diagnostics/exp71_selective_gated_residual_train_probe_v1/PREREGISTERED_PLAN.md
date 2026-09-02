# Exp71 preregistration: selective gated residual train probe

Status: recipe locked before execution

Scope: ICH only; train folds only; no MLS, fracture, calibration, outer, test,
triage fusion, row-level export, checkpoint promotion, or leaderboard inference

## Motivation

Exp70 removed the Exp69 collapse and recovered 28.448% of train SAH-to-IPH
errors with very low collateral harm, but it still changed more correct pixels
than errors it fixed. Its conditional accuracy and macro recall fell by 0.07671
and 0.00372 percentage points, respectively. Shrinking the same update would
shrink benefit and harm together without fixing that selectivity ratio.

Exp71 changes the decision mechanism. A supervised selection gate predicts
whether the frozen incumbent is wrong at each incumbent-foreground pixel. The
bounded residual is soft-gated during training but is applied at evaluation
only where the fixed gate probability threshold 0.5 is crossed. This follows
the selective-prediction principle of jointly learning prediction and an
accept/reject function rather than relying only on incumbent confidence.

Primary research basis: Geifman and El-Yaniv, *SelectiveNet: A Deep Neural
Network with an Integrated Reject Option*, ICML 2019:
https://proceedings.mlr.press/v97/geifman19a.html

## Locked recipe

- Incumbent: Exp61 Unet++ / `tu-efficientnetv2_rw_s`, outer 2, calibration 1.
- Data: official schema4 train folds 0/3/4 only.
- Frozen: complete incumbent and its classification logits/support.
- Trainable budget: at most 5,000 parameters.
- Shared `3x3 Conv -> GroupNorm -> SiLU` stem, five-channel zero-initialized
  residual output, and one-channel selection gate.
- Residual cap: `4*tanh`; conditional foreground margin 1.0.
- Initial gate probability: 0.01; hard evaluation threshold: 0.5.
- Population-normalized correction CE coefficient 4.0 and soft-KL
  preservation coefficient 1.0.
- Gate target: one only for known supported true-foreground pixels whose
  incumbent subtype is wrong; every other incumbent-foreground pixel is zero.
- Gate BCE positive weight 200 and total gate coefficient 0.25.
- Weak foreground class weights: power 0.25, maximum 2.
- AdamW, learning rate `5e-4`, weight decay `1e-4`, one epoch, batch 16, BF16,
  seed 42.
- No checkpoint is saved by this probe.

## Locked gates

All gates must pass:

1. Trainable parameters at most 5,000.
2. Exact initial hard-mask identity.
3. Exact final foreground-support lock.
4. At least 100 SAH-to-IPH incumbent error pixels.
5. At least 10% SAH-to-IPH recovery.
6. Correct IPH harm at most 0.3%.
7. Correct IVH/SDH/EDH harm at most 0.3%.
8. True-background subtype changes at most 0.3%.
9. Conditional accuracy non-decreasing.
10. Conditional macro recall non-decreasing.
11. Gate error precision at least 10%.
12. Gate error recall at least 10%.
13. Gate coverage at most 5% of incumbent foreground.

Passing authorizes one locked patient-safe calibration screen. Failure rejects
this recipe before calibration/outer; thresholds are not relaxed after seeing
the result.
