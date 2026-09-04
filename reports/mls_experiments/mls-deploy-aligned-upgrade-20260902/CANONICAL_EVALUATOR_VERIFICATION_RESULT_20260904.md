# Canonical evaluator implementation: numerical reproduction not yet certified

## Implemented protection

`scripts/evaluate_mls_canonical_resource_cuda.py` replaces the legacy resource
entry point for prospective fixed fold0/seed42/epoch15 evaluation. It checks
all pooling fields, clipping, preprocessing, source fingerprints, input content
and ordered SOP identifiers. A candidate requires a checksum-bound successful
baseline self-test under the same evaluator. No default can silently change
pooling; only the precisely SHA-bound historical baseline is allowed its
documented missing-single-selector schema migration.

23 target-server tests passed, including missing/mutated inference fields,
legacy-profile rejection, clipping parity with the three-seed evaluator,
input/UID fingerprints, coverage, and per-study failure despite a small mean.
This is evidence about the contract, NOT proof of production numerical parity.

## Full70-study CUDA baseline reproduction

The full diagnostic was terminal with `failed_baseline_reproduction`, as
required by the fixed tolerances. All70 studies produced finite outputs.
No new model was trained and the original reference was not overwritten.

| Metric | Immutable reference | Fresh baseline, batch6 |
|---|---:|---:|
| MAE mm | 1.470958639 | 1.466848264 |
| F1 at3mm | 0.819672131 | 0.819672131 |
| F1 at5mm | 0.736842105 | 0.736842105 |
| Boundary-F1 | 0.778257118 | 0.778257118 |
| Objective | 1.914444403 | 1.910334027 |

-42 studies exceeded the unchanged1e-5mm per-study reproduction tolerance.
-Mean absolute prediction difference:0.006000233mm; maximum:0.316395283mm.
-No change in F1 at1/3/5mm; MAE differs by0.004110375mm.
-Fresh inference:50.720s, peak allocated0.452GiB; PyTorch2.10.0+cu128.

The apparently better fresh MAE is NOT a model gain: identical checkpoint
bytes were evaluated. The maximum residual must not be dismissed as tiny
rounding noise without investigation. The current default flags observed in
the same project environment are matmul TF32 off, cuDNN TF32 on, cuDNN
benchmark/deterministic off. They are hypotheses, not a demonstrated cause.

## Evidence and current disposition

Evaluator commit`ae02fab`, SHA256
`274243acd0e3b3fc2b3876f89d6daec59f6fde8b72edc7b633d7032c04ec7970`.
Full baseline aggregate retained locally as
`CANONICAL_BASELINE_RESIDUAL_AGGREGATE_20260904.json`, SHA256
`d920c3209464c4ab09eccbfefd680b010ed66f9743fb2f694ad18ccfc566b443`.
Server original:
`/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/canonical_baseline_residual_diagnostic_20260904/aggregate_summary.json`.
Private records, raw-file/SOP fingerprints and per-slice predictions remain
only on the server; private SHA256
`fdfca255e5558459ae3c57f574acc7090bb0595eb9c19ec3e2a44d8f004becd3`.
The first fail-fast attempt is preserved separately in
`canonical_baseline_verification_20260904/status.json` (failed, no retained rows).

Do not use either failed artifact as a verified baseline. Do not change
reproduction tolerances or redefine this as a successful test. No candidate
training, replication, model release or submission ZIP is authorized by it.

Follow-up is the fixed baseline-only batch/precision diagnostic documented in
`BASELINE_RUNTIME_PARITY_PROTOCOL_20260904.md`, commit`4ac01e4`. It includes a
repeat-default control and two one-factor changes, and compares identical
input bytes. No threshold or accuracy optimization is involved. The15-minute
monitor remains paused.

## Fixed runtime diagnostic: completed

Same model, same70 studies, unchanged raw bytes and ordered SOP fingerprints:

| Mode | Mean difference vs old reference mm | Max difference mm |3/5mm changed decisions vs old|
|---|---:|---:|---:|
| Default repeat, batch6, convolution TF32 on |0.006000233|0.316395283|0 / 0|
| Batch16, convolution TF32 on |0.006144016|0.315499306|0 / 0|
| Batch6, convolution TF32 off |0.001307337|0.007967949|0 / 0|

The default repeat equals the preceding fresh run EXACTLY for all70 predictions.
Thus the measured default discrepancy is repeatable, not evidence of random
run-to-run instability. Changing batching alone does not explain the large
reference residual. Disabling convolution TF32 substantially reduces it, showing
an important precision-mode contribution. The remaining difference is real and
unexplained; neither a complete causal account nor reproduction pass is claimed.
All42 previously sensitive studies still exceed1e-5mm in IEEE mode.

IEEE-mode MAE1.470565135, F1@3=0.819672131, F1@5=0.736842105 and objective1.914050899
are numerical execution diagnostics of the SAME weights, not a new model.
The fixed three-mode test took114.674s and did no training. Aggregate
`baseline_runtime_parity_diagnostic_20260904.json` SHA256:
`2bcec552e6c895e45c0e6423b2bcf32027df0487056c8e99f4f76f5a2ad365a3`.
Diagnostic source SHA256:
`f1f5d27aabf00ee82db45411eb8c62bfa57e7c85a1fd558be4cc98ddb8f6385b`.

## Decision for continuation

The prospective evaluator now explicitly fixes and records IEEE convolution
and matmul precision; test/source hashes must be reverified after this amendment.
It also suppresses identifier-bearing UID format warnings without weakening
identity validation. The original failed tests stay failed and their evidence
is immutable. Do not chase an unrestricted batch/precision grid to obtain a
green historical comparison, and do not claim that IEEE alone fixed parity.

Next: define and validate a paired baseline on the current server with identical
explicit runtime for all candidate comparisons, preserving the old reference
and its failed cross-runtime parity. Require independent repeated prediction
reproduction, exact raw-content/UID parity and all unchanged final triage gates.
Any runtime-reference migration must be explicit, not an implicit loosening of
the old1e-5/1e-6 tolerances. This qualification remains unfinished; training has
not resumed. Once qualified, return to a distinct model-improvement hypothesis.
