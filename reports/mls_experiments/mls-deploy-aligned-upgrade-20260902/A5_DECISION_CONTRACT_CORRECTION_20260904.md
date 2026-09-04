# A5 decision-publication correction

## Evidence and scope

Static review while A5 training was active found two output-contract defects:

- The A5 wrapper inherited A2's
  `can_start_only_seeds_2026_and_3407_on_fold0` flag, which became `true` on a
  passing screen despite A5 requiring a separately registered replication plan.
- The shared evaluator published an intermediate A2 decision to the final A5
  output path before the wrapper replaced it with A5 metadata.

The shared evaluator now accepts a keyword-only `publish` option, defaulting
to `true` for existing callers. A5 sets it to `false`, forces the inherited
replication flag to `false`, and atomically publishes the A5 result once.
Metric calculations, all five numeric gates, epoch, fold, study population,
checkpoint selection, pooling, and training code are unchanged.

## Verification limits

The A5 regression tests now exercise both passing and failing synthetic JSON
inputs, require exactly one A5 publication and no shared A2 publication, compare
the persisted document to the returned result, and prohibit the inherited
replication permission in either outcome. These use dummy checkpoint bytes and
perform no model computation.

The local test command did not execute: the sandboxed uv Python launcher failed
with permission denied, and the escalation request was rejected. The user then
clarified that tests and all other execution must take place on the target
server; the local machine is for source editing, version control, and artifact
storage. No local test ran. Static review and `git diff --check` passed locally.

The target server's project venv then ran
`python -m unittest tests.test_evaluate_mls_a2_fold0_resource_screen tests.test_evaluate_mls_a5_fold0_resource_screen`:
**8 tests passed in 1.655 seconds**. These tests performed no model computation
and did not access training logs or candidate metrics.

## Verified deployment

Before replacement, absence of the A5 resource-screen launcher status and tmux
session was checked. The three old files were backed up under
`/workspace/iaaa_artifacts/server_source_backups/a5_decision_publication_20260904/`.
The replacements in `/workspace/IAAA_BrainCtTriage_mls_da` passed SHA-256 checks:

| File | SHA-256 |
|---|---|
| `scripts/evaluate_mls_a2_fold0_resource_screen.py` | `7207be1ada919771e9a7013ebb59cb320532587cf72434ed954745193ca3a685` |
| `scripts/evaluate_mls_a5_fold0_resource_screen.py` | `640ffc2b682e4243028820a0ce02a35ef3ef9b6ef493ad1c421cd5e83dc6a3da` |
| `tests/test_evaluate_mls_a5_fold0_resource_screen.py` | `477b8c0285573571f16698fa3d4fa75af030dd970e721d3c5df35faeee675596` |

Deployment and target-server regression tests are complete. The next terminal
resource-screen result must retain
`can_start_only_seeds_2026_and_3407_on_fold0: false`,
`promotion_eligible: false`, and `submission_zip_allowed: false`.
