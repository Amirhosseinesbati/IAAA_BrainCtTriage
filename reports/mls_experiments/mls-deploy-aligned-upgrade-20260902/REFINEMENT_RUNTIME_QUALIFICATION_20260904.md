# Versioned source-migration qualification, before A8 training

Previous turn made progress through real-batch CUDA preflight. This turn created a separate canonical evaluator and strict qualification helper; the original evaluator, qualifier and historical receipts are unchanged. The new evaluator differs only in explicit updated-source hashes, binding the refinement module, recording its enable flag, and using the new source-migration receipt loader. Pooling, clipping, preprocessing, scoring and gate logic remain unchanged.

## Acceptance rule

The pinned historical baseline checkpoint is inferred on all70 fixed fold0 studies, with refinement disabled, using the same IEEE runtime/batch6. The helper reconstructs and verifies the old qualified reference using its original receipt checksum, then requires exact equality of every private study record, including all decoded slice predictions, truth and raw input/SOP fingerprints. It also requires identical non-source inference signature, hardware and aggregate metrics. Newly qualified source file hashes are checked against actual files. No numerical tolerance is relaxed. Prospective gate bounds are copied unchanged from the verified prior qualification.

The new base evaluator can retain `failed_baseline_reproduction` relative to the older historical cross-runtime anchor; this known distinction is not concealed. The new qualification must prove exact agreement with the already qualified same-runtime reference, not simply tolerate a mismatch. A new receipt is emitted only after all checks pass. This qualifies source migration, not a model upgrade.

## Execution and recovery

- Local/server source bundle SHA256 matched `48b22b3ab6a374c5971928d87d2eeb37ee519f43cd3d1fb72ab0628c1f30860f`.
- Server py_compile passed before launch.
- Supervisor job: `mls_reference_qualification`, initial PID53645; RUNNING confirmed immediately after launch. autostart=false, autorestart=false. No heartbeat or15-minute schedule was created.
- Wrapper: `scripts/run_reference_runtime_qualification.sh`.
- Log: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/reference_runtime_qualification.process.log`.
- Output directory: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/reference_refinement_baseline_qualification_20260904`.
- Successful final receipt: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/reference_refinement_runtime_qualified_20260904.json`.

Do not restart based on an observation timeout or the older-anchor reproduction flag alone. Inspect supervisor/process state and final receipt; raw predictions stay server-only. At this note's creation qualification is running and no result is claimed. No A8 training, replication, promotion or submission ZIP has been launched.

Next: consume the completed receipt, preserve aggregate evidence locally/MLflow, then finish the executable A8 paired protocol/launcher already specified in REFERENCE_REAL_BATCH_REVIEW_20260904.md. A qualification failure must be investigated before launching training.
