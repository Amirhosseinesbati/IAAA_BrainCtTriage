# Explicit same-runtime baseline qualification (not a historical parity pass)

## Why and scope

The old reference retains its original predictions and failed cross-runtime
reproduction checks. Do not loosen1e-5/1e-6 tolerances or relabel those failures.
The fixed diagnostic showed exact repeatability on3090 and a substantial TF32
contribution to its difference from the old runtime. Prospective candidate
comparisons need a baseline measured with exactly the same explicit runtime.
This qualification addresses that requirement; it is not evidence of MLS or
triage improvement and cannot authorize a submission.

## Fixed controls before execution

Run the SAME immutable Exp16 epoch15 checkpoint twice in separate processes,
each loading it independently. Use all70 fold0 studies, seed42 checkpoint,
image512/channels3, fixed canonical pooling/clamp, inference batch6, no autocast,
IEEE convolution and matmul, matmul precision highest, cuDNN benchmark and
deterministic false. No mode search or choice based on the lower MAE.

Generate fresh directories `ieee_baseline_independent_a_20260904` and
`ieee_baseline_independent_b_20260904` under the campaign. Use the evaluator's
existing baseline-self-test mode, which still reports failure if the OLD
reference is not reproduced. Preserve both outcomes unchanged. The following
separate qualification must then satisfy ALL conditions:

1. Distinct recorded OS process IDs and execution UUIDs, same source/runtime/
   hardware/checkpoint/fold/truth/reference fingerprints.
2. Identical study coverage and EXACT equality of every recorded local decoded
   slice output, study MLS, truth and input fingerprint between executions.
   No numeric tolerance replaces this requirement.
3. Exact raw-file-content and ordered SOP fingerprint agreement with the
   previously pinned70-study input anchor; official truth unchanged.
4. Every original baseline decision at1/3/5mm is unchanged, study by study.
   Identical F1 averages alone are insufficient.
5. Recompute aggregate metrics from private records; both recorded vectors
   must match exactly.

Failure of any requirement means no qualified reference. Preserve the result
and investigate; do not select a favorable execution or change tolerances.

## Candidate resource gates cannot get easier

Use the stricter bound from the old frozen resource screen AND the freshly
qualified baseline: MAE upper bound=min(old bound,new baseline MAE); F1@3,
F1@5 and Boundary-F1 lower bounds=max(old bound,new baseline value); objective
upper bound=min(old bound,new baseline objective minus0.01). Retain the original
1e-8 comparison rounding tolerance. Tests cover both favorable/unfavorable
runtime drift. A baseline control never counts as a model improvement.

Candidate evaluator requires the qualification JSON's explicit SHA and rebuilds
the qualification from both checksum-bound audit artifacts on each invocation.
It verifies current source/runtime/hardware and per-study raw/SOP fingerprints
before candidate inference. No automatic training/replication, no promotion.

## Final goal remains unchanged

This qualifies only the inexpensive fold0/seed42 resource comparison. Future
three-seed/cross-fold baseline and candidate audits must share the explicit
runtime policy; old mixed-runtime median caches cannot silently stand in for
matched controls. Full frozen-Champion/oracle triage evaluation, five-fold
coverage, bootstrap and all final hard gates remain required. No claim of
leaderboard progress until those checks and actual submission evidence exist.

Only aggregate reports/receipts go to local storage or MLflow; private rows
stay on the server. Both CUDA jobs run sequentially under the campaign lock.
No CPU model execution. The cancelled15-minute monitor stays paused.
