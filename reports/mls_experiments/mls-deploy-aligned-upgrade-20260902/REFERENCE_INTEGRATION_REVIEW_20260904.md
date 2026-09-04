# Opt-in reference refinement integration: eight CUDA checks passed

The previous response was a requested status report rather than model progress. This continuation completed the partial model edit, connected an explicit `use_reference_refinement=false` default to multitask configuration, training construction, inference construction and resume configuration checks. The heatmap-only loader rejects refinement checkpoints rather than silently discarding their weights. The single-task trainer rejects this option. All three ordinary, multitask and extended forward paths use the same refinement function. Historical model parameter keys remain unchanged when the option is false.

## Preservation and deployment

Before changing server code, all six affected historical source files matched recorded hashes. Exact originals were copied under `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/pre_reference_integration_source_20260904`, preserving relative paths. No trained checkpoint, data or prior evidence was overwritten. New source bundle local/server SHA256 matched `4396676c055930a99885073004f0ec9a0d0cecf43fe6eaec3a6a095372da24fc`.

Important: old source-pinned audit scripts now intentionally reject the changed source tree. Do not silently replace their pins and claim equivalence. Historical source snapshots and reports remain valid evidence of their original runs; the next candidate protocol must explicitly qualify its updated runtime on the same data/reference. This integration test is not that full canonical inference qualification.

## Completed test

`scripts/test_mls_reference_integration_cuda.py`, session 93169, exited 0 on server CUDA. Eight checks passed:

Result JSON local/server checksum verified equal: `b2646da4ac1bf47f0083db699d65933e15ac76093a1e0831295c2be08ff7f459` (`REFERENCE_INTEGRATION_CUDA_20260904.json`).

- Historical config defaults to refinement disabled.
- Exact outputs against the backed-up historical model source with the same pinned trained baseline weights and synthetic inputs.
- Ordinary, multitask and extended forward outputs match in eval mode.
- Actual multitask training loss has finite gradients.
- One synthetic AdamW step changes the final refinement layer.
- Refined checkpoint reload reproduces outputs exactly.
- Heatmap-only legacy loader rejects refined checkpoints.
- Missing refinement weights cause strict loading failure.

Batch2 at 512 pixels, one synthetic optimizer step, no patient images. Peak allocated CUDA memory including the Adam state was 3.849952 GiB; this is not batch5 full training memory. Temporary synthetic checkpoints were confined to a dedicated temporary directory and removed automatically; no production checkpoint was touched. No full training started. Report archival in MLflow remains pending.

## Remaining before controlled training

One actual training batch5 smoke/fallback-frequency check, historical-baseline canonical runtime qualification for the updated source, and a fixed experiment protocol with identical base initialization and optimizer-update exposure across control/candidate. No threshold tuning, data exclusion, A7 winner claim, or baseline replacement. Resume flag is checked in code; full interrupted-job resume and extended ordinal-enabled training have not been exercised by this narrow test.
