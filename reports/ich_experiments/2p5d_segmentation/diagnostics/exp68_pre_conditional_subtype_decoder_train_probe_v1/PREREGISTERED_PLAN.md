# Exp68 train-only two-stage conditional subtype decoder probe

## Question

Can a materially expressive second-stage decoder/head correct the dominant
SAH↔IPH subtype confusion while the audited exp61 model retains exact ownership
of foreground/background support and all auxiliary classification scores?

Exp67 proved that a 3,217-parameter frozen residual can make safe local
relabels, but it improved SAH MAE by only 0.0216 mL and failed its locked gate.
No further cap, weight or epoch sweep of that frozen residual branch is allowed.

## Locked architecture and recipe

- Data scope: train folds 0/3/4 only. Calibration fold 1 and outer fold 2 are
  not used for image inference, loss, checkpoint selection or quality claims.
- Base: audited exp61 checkpoint and schema4 manifest, verified by SHA-256.
- Stage 1: the complete incumbent model is frozen in eval mode. Its hard
  foreground/background support and six auxiliary classification logits are
  returned unchanged.
- Stage 2: a copy of the incumbent decoder and six-channel segmentation head is
  initialized from exp61 and trained; the encoder is frozen and shared through
  detached incumbent features.
- Deployment rule: outside incumbent foreground, all mask logits are exactly
  incumbent logits. Inside foreground, stage 2 may choose only among IVH, IPH,
  SDH, EDH and SAH; a fixed +1 logit margin guarantees foreground support cannot
  disappear.
- Supervised objective: five-way cross-entropy on spatially known true-
  foreground pixels already supported by the incumbent. Pixel-frequency class
  weights reuse the exp61 rule (power 1, maximum 8).
- Stability objective: weight 0.25 hard-distillation to the incumbent subtype
  on incumbent-foreground pixels outside the supervised set (known background
  or spatially unknown rows), preventing arbitrary false-positive redistribution.
- Optimizer: AdamW, one complete train epoch, lr=1e-4, weight decay=1e-4,
  batch=16, seed=42. No learning-rate, stability-weight or epoch sweep.
- Probe: every train row, deterministic and without augmentation. It is a
  capacity/selectivity proof on training data, not a generalization estimate.
- Persistence: aggregate JSON and MLflow metrics only. No row-level medical
  prediction, checkpoint, calibration result or outer result is retained.

## Preregistered gates

Every condition must pass:

1. Exact hard-mask identity at initialization.
2. Exact foreground/background support identity after training.
3. At least 20% of true-SAH pixels initially predicted as IPH become SAH.
4. At most 1% of correctly predicted true-IPH pixels lose IPH.
5. At most 1% of correctly predicted IVH/SDH/EDH pixels change subtype.
6. At most 2% of true-background incumbent-foreground pixels change subtype.
7. Conditional foreground pixel accuracy improves by at least 0.5 percentage
   points.
8. Conditional macro subtype recall improves by at least 1 percentage point.

Passing authorizes exactly one locked patient-safe calibration screen after a
separate calibration plan and integration tests are committed. It is not a
promotion, OOF, outer-fold or leaderboard claim. Failure triggers causal review
of capacity, support coverage and stability—not post-hoc threshold relaxation.
