# Real training-batch preflight completed; fixed next experiment design

Previous goal turn was progress: integration plus checkpoint tests. This turn exercised the actual multitask loss on a fixed mixed training batch and examined initialization fallback on the frozen positive training sample. Session 37428 exited 0; no validation images used. Module state was discarded afterward, not saved as an experimental checkpoint.

## Evidence

Result JSON copied locally with equal local/server SHA256 `4dff6c89b3a7c8cf09b2cbed254fbd5547b5010dcd9ab5df7ea0901598bcb446` (`REFERENCE_REAL_BATCH_PREFLIGHT_20260904.json`).

- Common base initialization is the existing immutable ImageNet-backbone/random-head state saved BEFORE A7's disposable preflight step, SHA256 `99996c103d672bbbdc3f589a38b6555e4f54217769c03fef5922ea6d7e15367a`. It is not A7's trained model.
- All shared state loaded with only explicitly enumerated refinement keys missing; unexpected/missing base keys were rejected.
- Fixed 128 positive training slices reproduce sample hash `b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1`.
- Initial eval reference valid: 119/128; fallback: 9/128. These are untrained initialization counts, not learned-model localization accuracy or long-run exposure.
- Real mixed unaugmented batch: three positives plus two negatives, chosen by row metadata before outputs. Four of five references valid in train mode.
- One actual AdamW update with multitask loss, strict determinism and finite gradients updated the refinement final conv. Peak allocated memory including optimizer state: 4.278028 GiB. This is one batch, not a long-run memory guarantee.
- Source script local/server SHA256: `bde8b2a3d42f26533a33a1f80763cc999c7bb0495ced1c9293e9cc4cd69bfc7c`. Updated model/config/trainer hashes are recorded in result JSON.

## Bounded prospective comparison (A8 design; executable manifest still required)

Two arms: original model control and reference-refinement candidate. Shared parameters start from exactly the immutable untrained state above; candidate-only parameters are seeded explicitly and final conv zero. Both train from scratch with respect to MLS, not from rejected A7 weights or the disposable preflight state. Keep existing baseline losses/selector, ordinary augmentation (NO A7 paired-view/consistency), batch5/accumulation1, 541 steps per epoch, seed42/fold0, fixed epoch15 with the 23-epoch LR horizon and two warmup epochs. Reset epoch RNG identically and verify tensor exposure hashes; no validation during training and no epoch selection.

This contrasts the whole refinement package against original architecture, NOT coordinate conditioning independently of extra convolutional capacity. Do not claim a positive result isolates that mechanism. A third capacity-matched arm is not automatic; first determine whether the package clears the actual resource/triage gates.

Use the same unchanged numerical resource gates and an updated-source canonical baseline qualification. All70 study outputs must match the prior qualified baseline (or the discrepancy must be explained before training). Never silently overwrite old source pins. Both final arms are evaluated with identical frozen pooling/runtime; candidate must pass all baseline gates and improve objective over control before any replication review. Fold0 is repeatedly observed exploratory screening, not final generalization evidence; subsequent independent folds/seeds and full frozen triage gates remain mandatory.

## Immediate next action

Implement a versioned updated-source qualification wrapper and a pinned executable A8 manifest/launcher; reuse the existing managed-job pattern, MLflow logging, full-state recovery and GPU lock. Do not run more exploratory architecture probes unless qualification exposes a concrete failure. Three refinement technical reports (prototype, integration, this real-batch check) still need MLflow archival. No training or submission has been launched from this report.
