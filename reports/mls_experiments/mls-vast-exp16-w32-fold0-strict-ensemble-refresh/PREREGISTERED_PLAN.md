# Exp16 strict fold-0 ensemble refresh

## Question

Does the Exp15r strict recipe transfer to held-out fold 0 strongly enough to
replace the historical fold-0 epoch-15 model in the deployable three-fold
ensemble?

## Single changed factor

Relative to Exp15r, only `fold` changes from `2` to `0`. The backbone, seed,
3484-row processed-data contract, sampling mode, loss weights, optimizer,
augmentations, 23-epoch schedule, snapshots and strict determinism remain
frozen. No warm start, data rebuild, loss or architecture change, pooling
retune, ICH input, model averaging or CPU model fallback is allowed.

The data preparation flag is disabled because the already validated immutable
processed dataset is reused byte-for-byte; this is an orchestration choice, not
a changed training-data factor.

## Preconditions

1. Exp15r epoch17 has passed its 67-study promotion gate and exact packaged
   runtime parity test.
2. The processed-data contract remains 3484 rows, 338 studies, 1781 positive
   and 1703 negative slices, with all paths resolved.
3. The server is on the commit containing this manifest; the host driver
   library precedes the stale CUDA compatibility library.
4. No live MLS training session or run directory exists under the Exp16 name.
5. MLflow secrets remain root-only and the Vast instance remains persistent.

## Evaluation protocol

- Train all 23 epochs unless a non-metric safety failure stops the run.
- Preserve epochs 13, 15, 17, 19, 21 and 23 plus all named selection
  checkpoints.
- Run GPU-only inference for every candidate on all 70 held-out fold-0 studies.
- Apply the already locked production profile: severity window radius 3,
  selector gate 0.5, at least 3 active slices, weighted q0.75, guard 0.
- Treat any pooling grid fitted on fold 0 as diagnostic only.

## Decision rules

Infrastructure success requires terminal exit code 0, a nonempty MLflow run,
strict deterministic flags, no CPU fallback, no NaN/Inf and 70/70 successful
studies for every audited checkpoint.

The historical packaged reference is Exp08 fold0 epoch15 under the same locked
profile: MAE `1.664553 mm`, boundary F1 `0.822263`, objective `2.020027`.
A new checkpoint is eligible to replace it only if:

- MAE is no worse than `1.664553 mm`;
- boundary F1 is at least `0.82`; and
- selection objective is no worse than `2.020027`.

If no preregistered candidate passes all three checks, retain the historical
fold0 model. Do not promote based solely on an in-fold retuned pooling rule.

No server stop or destroy action is authorized after completion.
