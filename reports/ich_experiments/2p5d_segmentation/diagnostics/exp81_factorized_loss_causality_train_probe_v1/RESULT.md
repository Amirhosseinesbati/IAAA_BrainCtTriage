# Exp81 result — train-only attribution was insufficient

## Decision

`short_horizon_drift_not_reproduced`

The exact Exp80 four-component objective did not reproduce its calibration
collapse on fixed, non-updated train-distribution probe batches. This diagnostic
therefore does not authorize a loss change, calibration recipe, outer
evaluation, or model promotion.

## Reproducibility

- MLflow run: `3a2da4399082437e92356cb94ae8c115`
- Git commit: `c1e312ab1d02168d0da8f10b0ec4982c228f4a36`
- Exp61 checkpoint SHA-256:
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`
- Schema4 manifest SHA-256:
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`
- Causality aggregate SHA-256:
  `9b48d366e4e41ef97a812d49c4d260ba47c8f8de99180beb0b48abecf97b3ad7`
- BF16; batch 16; 24 gradient batches; eight updates per variant; four
  fixed train-distribution probe batches.
- Aggregate-only output; no calibration, outer, OOF, row-level predictions, or
  checkpoint artifact.

## Main evidence

Mean SAH/SDH soft-Dice drift after eight identical updates was:

| Variant | Diffuse mean soft-Dice delta |
|---|---:|
| Full Exp80 objective | +0.00809 |
| Without conditional Dice | +0.02692 |
| Without conditional focal | +0.01901 |
| Foreground only | +0.02694 |

The full objective therefore did not satisfy the locked `-0.001` failure
reproduction threshold. Although removing conditional Dice produced the largest
conditional-component rescue, assigning causality from a non-failing setting
would be invalid.

The weighted parameter-gradient evidence was also mixed:

- foreground support vs conditional focal cosine: `-0.0733`;
- foreground support vs conditional Dice cosine: `-0.1456`;
- conditional focal vs conditional Dice cosine: `+0.4367`;
- decoder gradient norm: foreground `0.4040`, conditional focal `0.6683`,
  conditional Dice `0.3776`.

Thus both conditional components oppose the foreground objective in shared
parameters, while agreeing moderately with one another. The focal component is
the strongest decoder update, but this alone does not prove it caused Exp80's
held-out failure.

## Important interpretation

The fixed train probe started at SAH Dice `0.6923` and SDH Dice `0.8238`, versus
Exp61 calibration SAH `0.0530` and SDH `0.3817`. These are not comparable
generalization estimates, but the scale of the gap explains why favorable train
drift cannot resolve the held-out collapse.

Foreground-only updates also reduced mean foreground probability on true
foreground by `0.0567`, including large target-probability reductions for SAH
and SDH, while soft Dice rose because predicted probability mass contracted.
Consequently a short train soft-Dice gain may represent pruning/underprediction,
not better diffuse-subtype transfer.

## Consequence

Exp82 must repeat the exact four-update Exp79 horizon with the same ablations and
measure full aggregate calibration metrics. Outer fold 2 remains untouched.
