# A8 launched after exact70-study source qualification

Previous turn created and launched qualification (progress). This turn confirmed supervisor job `mls_reference_qualification` exited and a valid successful receipt exists. Every complete study/slice record matched the previous qualified baseline exactly. Source migration therefore passed without changing any numerical gate. No model improvement is claimed by this check.

Receipt `REFERENCE_REFINEMENT_RUNTIME_QUALIFIED_20260904.json` copied locally; local/server SHA256 `9255e8387977c97bba19b77aa454403538abaa3ea03ddf184e89b87f136e3b96`. Baseline aggregate SHA256 `e91d240619c4b9d81818f3987315b8f7b9eaad0e13f2f6c58ab84d82cb55efc3`; all private records remain server-only.

## Training protocol and launch

Executable manifest: `A8_TRAINING_PROTOCOL_20260904.json`. Source/input hashes, immutable initialization and qualified-runtime receipt are enforced by setup. `--validate-only` completed successfully (session64057, 1938 shared initialization keys); both new Python files compiled on the server. Uploaded source tar local/server SHA256 matched `8c7773d56431a6e6023e33efe18d5fe916490496aea9c23c0f3fde40a39a6a05`. Approximately56GB disk was available before launch.

Supervisor job `mls_a8_pair` started and was RUNNING, initial parent PID54184. autostart=false, autorestart=false. It sequentially launches ordinary control then refinement, each with fixed15 epochs/8115 updates, batch5,23-epoch schedule, and no validation selection. Epoch input exposure hashes and shared initialization must match before paired interpretation. This is not A7 paired-view training: no translated extra view or consistency loss. It tests the whole refinement package, not isolated coordinate conditioning versus equal parameter capacity.

One startup verification at parent uptime1m24s confirmed control worker PID54427 alive, CUDA process54427 using5414MiB, and `status=training` with MLflow run `451470b102064180add9d0d21f2e45fe`. PIDs54724/54725 also appeared with the inherited worker command (data-loader subprocesses); only54427 appeared in CUDA compute processes. No epoch metric was inspected. This proves startup/tracking run creation, not final checkpoint delivery or completed learning.

MLflow setup is performed by each worker using the existing protected server environment. Manifest, aggregate training metrics, history and final checkpoint are logged; full-state recovery is written each epoch. No private prediction rows are uploaded. A training summary's existence alone is not proof that final MLflow upload succeeded; require terminal status and verified artifacts. The worker intentionally fails visibly on tracking errors rather than silently running untracked. An upload failure after epoch15 must not trigger redundant retraining.

## Monitoring / recovery

- Parent job: `mls_a8_pair`, initial PID54184.
- Log: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a8_pair.process.log`.
- Work: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a8_reference_refinement_20260904`.
- Each arm: `status.json`, `recovery.pth`, `training_summary.json`, final `mls_multitask_epoch_015.pth`.
- Pair receipt: `pair_completion.json`, only after both terminal successes and exact exposure verification.

Observe real process state, not merely lock/status files. Do not restart on timeout. No frequent epoch polling, no15-minute automation and no instance stop/destroy. Resume is explicit and verifies manifest/arm/config; full interrupted-run resume has not been tested in this campaign. Initial worker/GPU/MLflow status should be checked once to confirm startup, then wait for a meaningful milestone. Never read epoch loss values or private predictions into model context.

No automatic evaluation, replication, promotion or submission ZIP occurs. After pair completion, audit fixed checkpoints with the versioned evaluator and checksum-bound new runtime reference; unchanged resource and final frozen-Champion triage gates remain. Technical preflight reports and this receipt still require MLflow archival; they are not claimed archived yet.

## Subsequent verified pretraining-evidence archival

The following continuation completed archival (session32351 exited0) without polling epoch progress, changing training-pinned files, running GPU diagnostics or touching worker state. Five checksum-pinned aggregate JSONs are now stored in control run `451470b102064180add9d0d21f2e45fe` under `reports/a8_pretraining_evidence`: prototype preflight, integration test, real-batch preflight, source-migration qualification receipt, and canonical baseline aggregate. All five were downloaded again and matched source SHA256. No private prediction rows were uploaded. These replace the pending-archive status in earlier prototype/integration/preflight notes.

Receipt copied locally as `A8_PRETRAINING_ARCHIVE_RECEIPT_20260904.json`, local/server SHA256 both `664075f044951a78d03150a8018cfc3360008fa4a97fbc23b8cf39dfd1332b80`. Baseline aggregate copied as `REFERENCE_BASELINE_AGGREGATE_20260904.json`, SHA256 `e91d240619c4b9d81818f3987315b8f7b9eaad0e13f2f6c58ab84d82cb55efc3`, matching the qualification/MLflow-verified source. This proves evidence delivery, not final training completion or model superiority. Next useful local work while the existing pair runs is a fixed-checkpoint paired evaluator that rejects incomplete/mismatched training and never selects epochs by validation.
