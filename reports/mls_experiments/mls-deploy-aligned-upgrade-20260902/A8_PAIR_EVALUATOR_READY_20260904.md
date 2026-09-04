# Fixed A8 paired evaluator ready; ten logic tests passed

Previous turn progressed verified evidence archival. This turn prepared the paired evaluator while leaving all training-pinned files and worker state untouched. No epoch polling, new GPU workload, training restart or changed gate occurred.

`scripts/evaluate_mls_a8_pair.py` binds the A8 manifest, versioned canonical evaluator and exact source-migration qualification receipt by SHA256. It refuses to start before pair completion and refuses to overwrite an existing audit directory. It independently checks all15 epoch records,541 updates/epoch,8115 total updates, matching input exposure digests and immutable base initialization. Final epoch15 checkpoint bytes and embedded manifest/arm/initialization/MLflow identity must match their summaries. Only the declared boolean `use_reference_refinement` may differ across model configurations. Each audit must report the correct model variant.

Both checkpoint audits run sequentially with the same frozen canonical pooling/runtime/reference. Saved gates are recomputed from finite metrics and exact unchanged bounds; the objective must equal MAE+2*(1-boundary_F1). Refinement only becomes eligible for replication review if it passes all resource gates and has lower objective than its paired control. Review eligibility is not automatic replication, promotion, final triage success or permission to build a submission ZIP.

## Tests

On-server command: `.venv/bin/python -m unittest discover -s tests -p test_mls_a8_pair_audit.py -v`.
Session3494 exited0; all10 tests passed. Tests use synthetic dictionaries only, no model forward, patient data or GPU allocation. Coverage includes matched/mismatched exposure, undeclared configuration differences, wrong variant/checkpoint/runtime/coverage, false pass flags, nonfinite metrics, fabricated objective, gate failure, and identical outcomes not counting as a refinement benefit. This does not constitute full end-to-end evaluation of unfinished models or a test of every filesystem failure.

Uploaded archive local/server SHA256 `38e39ea896da0690240fe449db78b4cb6b35029a8f07b96cd8ff3d679cbe0087`. Only the new evaluator and new tests were added server-side; no training manifest/source changed. Evaluation has NOT been launched. At the next meaningful training milestone inspect the existing supervisor/worker handles, not loss values; after pair completion run this evaluator once under a managed one-shot job. Preserve partial audit outputs if a failure occurs rather than silently rerunning.
