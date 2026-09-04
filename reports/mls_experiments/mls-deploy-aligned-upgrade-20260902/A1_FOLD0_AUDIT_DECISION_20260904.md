# A1 ordinal auxiliary head — fold-0 decision record

## Protocol and integrity

This was the pre-registered A1 ablation: a training-only ordinal head for MLS
at 1/3/5 mm. Predictor, held-out fold, epoch 15, seeds 42/2026/3407, and
median-of-three aggregation were fixed. CUDA-only inference completed all 70
fold-0 studies. Checkpoint SHA-256 values matched the immutable A1 manifest.
Raw per-study predictions remain private and were not uploaded to MLflow.

## Exact median comparison

| Metric | locked baseline | A1 ordinal | delta |
|---|---:|---:|---:|
| MLS MAE (mm; lower is better) | 1.3871 | 1.7029 | +0.3158 |
| Boundary-F1 | 0.8184 | 0.7417 | -0.0767 |
| F1 at 1 mm | 0.7945 | 0.7949 | +0.0004 |
| F1 at 3 mm | 0.7797 | 0.7333 | -0.0463 |
| F1 at 5 mm | 0.8571 | 0.7500 | -0.1071 |
| selection objective (lower is better) | 1.7503 | 2.2196 | +0.4693 |

## Decision

**Reject A1.** It fails non-inferiority at both 3 mm and 5 mm and regresses
in MAE and Boundary-F1. Do not add A1 seeds, epochs, or folds and do not use
it in a submission. This rejects the fixed independent ordinal-head ablation
only, not every possible ordinal formulation.

The `exit_code.txt` in the artifact directory contained `1` from an earlier,
interrupted resume attempt. It is stale: the new audit log reached 70/70 and
the atomically written `status.json` and `aggregate_summary.json` both report
`completed`. It must not be interpreted as the terminal result.

## Next hypothesis

The existing absolute MLS objective cannot distinguish a geometrically mirrored
outermost-falx point. The annotation-defined signed offset is well-defined and
nearly balanced: 894 positive, 885 negative, and 2 zero among 1,781 target
slices. A2 adds only a low-weight signed perpendicular-offset loss to the
existing keypoint pathway; inference remains unchanged. This is motivated by
structural-midline literature, not a post-hoc pooling change.
