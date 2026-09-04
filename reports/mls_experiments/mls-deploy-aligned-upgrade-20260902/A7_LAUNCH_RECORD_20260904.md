# A7 paired translation trial: implemented, tested and running

The previous goal turn made progress by measuring a training-only translation
invariance violation and verifying the corrected runtime comparator. This turn
implements and starts the matched intervention, not another diagnostic-only
or architecture/pooling sweep. The original triage-improvement goal is not met.

## Fixed comparison

Two sequential arms: paired-view supervised control, then the identical
paired-view training plus overlap-normalized heatmap Jensen-Shannon consistency.
Both start from the same checksum-bound ImageNet backbone/random heads, not an
MLS warm start. One forward contains5 originals and5 translated images; one
optimizer step per5 underlying examples preserves541 steps/epoch. Both arms
have the same10-image BN batch and stochastic operations. Consistency imposes
agreement under translation AND independently sampled dropout; do not attribute
any future improvement exclusively to deterministic geometric equivariance.

Four fixed translations (+/-8px horizontally/vertically), existing ordinary
augmentation retained. Regenerate translated Gaussian targets; exclude invalid
positive translated annotations without converting them to negative examples.
The criterion reaches both views, only on valid positive anatomy. No selector
consistency, new head, primary geometry decoder, inference or pooling change.

Weight10, linear3-epoch ramp, fixed before training. Seed42/fold0,
AdamW1e-4/weight_decay.001, FP32/TF32off/strict deterministic CUDA. Stop at actual
epoch15 while preserving the original23-epoch cosine schedule/two-epoch warmup.
No15-to23 tail is computed and no internal validation/epoch selection occurs.
Each arm has8115 optimizer steps if complete. Per-epoch hashes of all augmented
inputs/coordinates must match before interpreting a paired effect.

## Verification and launch evidence

-17 server tests passed: six new paired-loss tests, four translation-probe
 tests, seven runtime-reference tests. No model was run on the user's PC/CPU.
- Real disposable CUDA preflight: batch10, one finite backward/optimizer step,
  peak allocated8.305997GiB, runtime4.00s, five valid second-view samples,
  three valid positives. Initialization saved BEFORE optimizer mutation.
- Weighted consistency/supervised gradient-norm ratio at output heatmaps:
  .000244463. This only excludes dominance at initialization; it does not prove
  an effective regularization strength later. Nearly uniform initial heatmaps
  can have tiny consistency gradients. Weight was not tuned on held-out data.
- Local preflight copy matches server SHA256:
  b3a556892c1673ce57165415cff3f03bab91328dea72943152b14127e9bdda04.
- Shared initialization SHA256:
  99996c103d672bbbdc3f589a38b6555e4f54217769c03fef5922ea6d7e15367a.
- Protocol SHA256:
  051691b8600fd965ec94ca6c30eb72b34c6a4aa40ecaa1395b4053e1f78c2f23.
- Training source commit e3854ad; finite launcher commit e735a5e.
  SCP uploads verified; no existing inference/training source overwritten.
  No GitHub push attempted because its authentication is unavailable.
- Supervisor program `mls_a7_pair`: RUNNING, parentPID47176; control
  childPID47321 independently verified live. GPU report showed child47321,
  93%utilization and9898MiB process memory (9908MiB total used).
- Control MLflow run:0f5b17c509714f7fa3da96726c59cfb6, status training.
  The consistency run ID is created only when that arm starts.

## Durable files and continuation

Server root:
/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a7_paired_translation_20260904

`sequence_status.json` records supervisor-child IDs and the active arm.
`control/status.json` and `consistency/status.json` record process/run identity.
Process existence or supervisor state, not a stale status/lock file, establishes
liveness. Do not read private process logs or per-epoch metrics to select epochs.

Each arm writes atomic`recovery.pth` (model/optimizer/scheduler/RNG/history),
then`mls_multitask_epoch_015.pth` and`training_summary.json` at completion.
MLflow receives training aggregates and final checkpoint. Raw study prediction
rows are never uploaded. The supervisor program is finite, autostart=false,
autorestart=false; no accidental repeated training. It does not stop the server.
The campaign GPU lock covers each child. The15-minute monitor remains disabled.

After completion, read`pair_completion_summary.json`: verify common
initialization, all15 exposure hashes and8115 optimizer steps for BOTH arms.
Run the unchanged canonical evaluator once per checkpoint using qualified
runtime reference SHA59743c79a788839940f73e6d9e81cbd564c0fae9421239e2382debf3b56e5b19.
Do not use old A2-A6 pooling launchers. Consistency needs all strict resource
gates AND lower objective than paired control before replication. If the control
alone improves, that is evidence for paired supervision, not consistency.
Neither can be released from this fold alone. Full multi-seed/cross-fold,
frozen-Champion/oracle final triage checks and artifact verification remain.

If interrupted: inspect live process/supervisor first; never restart because a
poll timed out. Explicit arm resume is supported by the training script's
`--mode train --arm <arm> --resume`, with manifest and exact recovery checks.
The sequence intentionally refuses automatic restart. Do not remove status or
lock files to force a restart; reconcile a truly terminated job first.

No improved model or leaderboard gain has yet been demonstrated; no release
checkpoint was copied into the user's best-model directory and no ZIP was built.
