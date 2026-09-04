# A9 speed evidence and 3-mm threshold diagnostic

## Status

Completed on the remote RTX 3090 on 2026-09-05.  This document contains only
aggregate evidence.  No study identifier, prediction, image, checkpoint, or
private row was copied to this repository.

## Proven safe speed improvement

The paired `A9-speed-equivalence` harness ran four one-epoch arms in the order
reference → optimized → optimized → reference, with the original A9 data order,
seed, strict FP32/no-AMP/no-TF32 setting, 169 optimizer steps, and frozen
baseline.  It required bitwise equality of the input digest, every loss in the
trace, mean loss, refiner/optimizer/scheduler/RNG states, and all frozen base
state before declaring an optimization usable.

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| Median timed epoch | 40.752565 s | 32.103864 s |
| Speed ratio | \- | 1.269398× |
| Relative improvement | \- | 26.94% faster |

All equivalence gates passed.  The only approved changes for a **new** trainer
are (1) zero-copy byte views for the already-required input digest, and (2) one
post-epoch host transfer for the loss trace instead of a scalar transfer each
step.  This does not alter A9 itself and is not a model-quality result.

Evidence: `A9_SPEED_EQUIVALENCE_RESULT_20260905.json`, SHA-256
`3c4e4f4957c10415eeed681f5fb7ab80ff8f9af3f7671ea5cb24b5f44bbcc508`.
The corresponding MLflow run is `d616931d222f4e14a9a9c471804ca2e4`, tagged as
`speed_equivalence` / non-candidate / promotion-ineligible.

## Why A9 was rejected

The A9 refiner improved the aggregate MLS objective but failed the hard
3-mm-boundary gate against the qualified runtime baseline:

| Metric | Qualified baseline | A9 refiner | Delta |
| --- | ---: | ---: | ---: |
| MAE (mm) | 1.470565 | 1.448133 | -0.022432 |
| F1 at 3 mm | 0.819672 | 0.786885 | -0.032787 |
| F1 at 5 mm | 0.736842 | 0.810811 | +0.073969 |
| Boundary F1 | 0.778257 | 0.798848 | +0.020591 |
| Selection objective | 1.914051 | 1.850437 | -0.063614 |

At 3 mm, four classifications changed: one true positive was gained, two true
positives were lost, and one false positive was introduced.  Thus the regression
is a concrete net loss of one TP plus one added FP, not an evaluator mismatch or
an input-alignment issue.  At 5 mm the direction was favorable (one TP gained,
two FPs removed).  The candidate's mean correction was -0.070054 mm, consistent
with the observed tendency to push some borderline 3-mm positives downward.

The aggregate-only comparison was performed over 70 exactly aligned studies;
both private prediction file SHA-256 values were pinned before parsing and the
same verified bytes were parsed.  Alignment checks for coverage, input
fingerprints, and ground truth all passed.  The public report checksum is
`1164ad0a7a2490d468743cae5e38da3619df7a1318c79690353f76fffe5650a6`.

## Consequence for A10

No blend, threshold, checkpoint, or pooling parameter may be tuned from this
screen.  A10 must be a single pre-registered hypothesis targeted at protecting
the 3-mm decision boundary while retaining the frozen qualified baseline and
the approved speed-only implementation changes.  It remains
promotion-ineligible until it clears a fresh, deploy-aligned evaluation gate.
