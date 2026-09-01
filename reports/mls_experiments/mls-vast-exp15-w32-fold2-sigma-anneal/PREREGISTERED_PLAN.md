# Preregistered plan: MLS Vast Exp15 sigma annealing

- Status: `not_started`
- Parent anchor: `mls-vast-exp14-w32-fold2-hybridsoft-repro`
- Historical reference: `mls-local-v2-exp10-w32-fold2-hybridsoft-transfer`
- Manifest: `config/experiments/mls-vast-exp15-w32-fold2-sigma-anneal.yaml`
- Compute policy: model forward/backward/validation are CUDA-only.

## Why this experiment exists

The fixed-sigma baseline asks the network to learn the same target precision
from the first update to the last. Exp15 tests a coarse-to-fine curriculum:
broader spatial supervision early, the frozen baseline width at the schedule
midpoint, and a sharper target late. The 23-epoch arithmetic mean remains
exactly 3.0 px, so the experiment changes target-precision ordering rather
than average target width.

The feature is opt-in. A missing `heatmap_sigma_anneal_end` reproduces the
historical fixed-sigma behavior, including Exp14.

## Launch gate

Exp15 must not start unless all of the following are true:

1. Exp14 reaches a terminal `completed` state with finite CUDA metrics and a
   durable MLflow run or a fully replayable deferred MLflow queue.
2. Exp14 used the committed manifest without an uncommitted server override.
3. The fixed epoch-15 Exp14 metrics remain within these gross-drift guards of
   local Exp10: study MAE within 0.35 mm, study boundary-F1 within 0.06, and
   selector AUC within 0.03.
4. Raw-data and DVC integrity remain clean and no ICH checkpoint is used.

These tolerances are operational reproducibility guards, not a statistical
claim. Failure sends the workflow to environment/data diagnosis rather than
to a new architecture experiment.

## Frozen control and changed factor

All validated training fields are equal to Exp14 except:

```text
heatmap_sigma_anneal_end: null -> 2.0
```

With `heatmap_sigma=3.0`, this resolves deterministically to:

- epoch 1: 4.0 px
- epoch 12: 3.0 px
- epoch 23: 2.0 px
- validation: fixed 3.0 px at every epoch

Fold, seed, HRNet-W32 architecture, sampler, selector targets, losses,
optimizer, learning-rate schedule, augmentation, aggregation, and snapshot
epochs are frozen.

## Evaluation and decision rule

Primary comparison is fixed epoch 15 on fold 2 under the same production
aggregation profile. Exp15 earns cross-fold transfer only if it improves
study MAE by at least 0.10 mm without reducing study boundary-F1 by more than
0.02. Slice MLS MAE may not regress by more than 0.15 mm and selector AUC may
not regress by more than 0.02.

Snapshots 13/15/17/19/21/23 are retained for a CUDA-only full-study audit,
but best-snapshot results are secondary and cannot replace the fixed-epoch
gate. If the gate passes, the identical schedule is transferred to folds 0
and 1 without fold-specific tuning. If it fails, no cross-fold GPU resources
are spent and the hypothesis is rejected or revised from error analysis.

## Persistence contract

The trainer writes an atomic recovery checkpoint after every epoch, appends
`epoch_metrics.jsonl`, refreshes `report.md`, logs to MLflow through the
resilient queue, and never auto-destroys the Vast instance. Any deployable
candidate must later be copied into a distinct local `checkpoint/mls/<run>`
directory with a short provenance README.
