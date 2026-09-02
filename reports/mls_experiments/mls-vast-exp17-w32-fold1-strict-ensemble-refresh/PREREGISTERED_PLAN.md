# Exp17 strict fold-1 ensemble refresh

## Question

Does the strict recipe that passed the frozen production gate on folds 2 and 0
also improve held-out fold 1 enough to replace the historical Exp09 epoch-15
member of the deployable three-fold ensemble?

## Single changed factor

Relative to Exp16, only `fold` changes from `0` to `1`. The HRNet-W32
architecture, seed, immutable 3484-row processed-data contract, slice-balanced
sampler, loss weights, optimizer, augmentations, 23-epoch schedule, snapshots
and strict determinism remain frozen. No warm start, data rebuild, loss or
architecture change, pooling retune, ICH input, checkpoint averaging or CPU
model fallback is allowed.

## Preconditions

1. The Exp16 two-fold package must pass exact slice-level parity and a full
   package CUDA smoke test before this training run starts.
2. The processed-data contract remains 3484 rows, 338 studies, 1781 positive
   and 1703 negative slices, with all paths resolved.
3. The server must be on the exact commit containing this manifest; the host
   driver library must precede the stale CUDA compatibility library.
4. No live GPU process, tmux session or durable run directory may already exist
   under the Exp17 name.
5. MLflow secrets remain root-only and the Vast instance remains persistent.

## Evaluation protocol

- Train all 23 epochs unless a non-metric safety failure stops the run.
- Preserve epochs 13, 15, 17, 19, 21 and 23 plus all named selection
  checkpoints.
- Run GPU-only inference for every candidate on all 67 held-out fold-1 studies.
- Apply the already locked production profile: severity window radius 3,
  selector gate 0.5, at least 3 active slices, weighted q0.75, guard 0.
- Treat every pooling grid fitted on fold 1 as diagnostic only.

## Decision rules

Infrastructure success requires terminal exit code 0, a nonempty MLflow run,
strict deterministic flags, no CPU model fallback, no NaN/Inf and 67/67
successful studies for every audited checkpoint.

The historical packaged reference is Exp09 fold1 epoch15 under the same locked
profile: MAE `1.258665 mm`, RMSE `1.976092 mm`, bias `-0.182218 mm`, boundary
F1 `0.823729` and objective `1.611207`. A new checkpoint is eligible to replace
it only if:

- MAE is no worse than `1.258665 mm`;
- boundary F1 is at least `0.82`; and
- selection objective is no worse than `1.611207`.

All three checks are mandatory. If no preregistered candidate passes them,
retain Exp09 epoch15. Do not promote from the online validation proxy or an
in-fold retuned pooling rule.

No server stop or destroy action is authorized after completion.
