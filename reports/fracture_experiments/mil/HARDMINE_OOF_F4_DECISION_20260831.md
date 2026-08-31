# OOF Hard-Mining Fold-4 Decision — 2026-08-31

## Decision

Do not replicate hard-negative/false-negative mining v1 across the remaining
folds. The experiment produced an interesting early detector checkpoint, but
the fully deployable detector/MIL blend underperformed the current incumbent
on the same outer fold.

## Audit and recovery

- Training used a patient-disjoint outer-fold mining gate: no Fold-4 validation
  study entered the mining pool.
- The initial post-training evaluator exposed broken relative `images` and
  `labels` links in the derived dataset. Training checkpoints were intact.
- Commit `d4bf769` changed derived-dataset links to verified absolute targets.
- The recovered run stopped after epoch 13 and retained nine checkpoints.
- All checkpoints were screened at Study level; Ultralytics `best.pt` was not
  trusted as the selection criterion.
- Per-study predictions and calibration arrays remain private on the server.
  Only aggregate summaries were logged to MLflow.

## Detector screening

| Model | Fold-4 adjacent-pair AUC |
|---|---:|
| Incumbent detector epoch10 | 0.791569 |
| Hard-mine Ultralytics `best.pt` | 0.791569 |
| Hard-mine selected `epoch2.pt` | **0.838407** |

The selected detector improved observed AUC by `+0.046838`, but the
50,000-iteration paired bootstrap interval was
`[-0.149883, 0.049180, 0.229508]`; the estimated probability that it was not
better was `0.30634`. This was promising but insufficient to justify five-fold
replication without the downstream MIL gate.

## Cache and MIL integration

- The epoch2 feature cache contains 7,683 slices from 338 studies and 320
  patients.
- Matching the direct evaluator's batch size (`16`) produced exact AUC parity
  for every pooling rule; maximum score differences were approximately
  `1e-16`.
- Nested SA-MIL selected `alpha=0.1` and 12 final epochs across seeds
  `42,43,44`.
- Outer Fold-4 MIL AUC was `0.765808`; MIL alone did not replace the detector.

## Pre-specified deployable blend gate

The blend used the incumbent pre-specified MIL weight `0.45`. Detector and MIL
scores were mapped through empirical CDFs fitted only on the 270 outer-training
studies.

| Candidate | Fold-4 AUC |
|---|---:|
| Hard-mine detector epoch2 | 0.838407 |
| Hard-mine detector + new MIL, fixed 0.45 | 0.845433 |
| Current incumbent detector + MIL, fixed 0.45 | **0.873536** |

The new blend gained only `+0.007026` over its detector, with bootstrap
interval `[-0.100703, 0.007026, 0.117096]` and probability-not-better
`0.45842`. It was `-0.028103` below the incumbent Fold-4 blend.

## Interpretation

Aggressive OOF hard-example repetition changed the detector ranking signal,
but did not preserve the complementary sequence signal that made the incumbent
SA-MIL blend strong. The experiment is rejected as a production candidate,
not erased: its MLflow runs and private server artifacts remain useful evidence
against repeating the same configuration.

## MLflow

- Detector/hard-mine run: `64d8f2b5884447adb1e3a8671c695fd5`
- Integrated SA-MIL run: `07ed25c910184696946c7515ea663b45`

