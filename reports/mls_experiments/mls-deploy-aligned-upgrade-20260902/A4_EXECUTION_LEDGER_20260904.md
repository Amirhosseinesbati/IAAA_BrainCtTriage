# A4 execution ledger — fold 0 / seed 42

- Status at recording: `running`.
- Start UTC: `2026-09-04T06:05:24Z`.
- Run name: `mls-vast-da-a4-pair-rank-fold0-seed42`.
- Compute policy: `cuda_only_no_cpu_fallback`.

## Reproducible source and transfer

- A4 implementation commit: `6ddd738244cc8b5d702235e64c88b1c8608a93f3`.
- Launcher-contract follow-up commit: `e189153d57e53c1a87517121ea44a92472afefcb`.
- Training manifest SHA-256: `c6f87f4a1519f565ab876658f69db198d5165eba47d848e924ec38c8b9c29c58`.
- The ten A4 implementation/manifest/test files were independently SHA-256
  matched after SCP to `/workspace/IAAA_BrainCtTriage_mls_da`.
- Existing overwritten source files were preserved before transfer under
  `/workspace/iaaa_artifacts/server_source_backups/a4_pair_rank_pre_6ddd738`.
- The revised launcher test was preserved under
  `/workspace/iaaa_artifacts/server_source_backups/a4_pair_rank_pre_e189153`.

## Preflight evidence

The actual server was an idle `NVIDIA GeForce RTX 3090` with `24,124 MiB` free
VRAM and `60 GiB` project filesystem free. The exact A4 synthetic primary
batch plus same-study pair forward/backward preflight succeeded under strict
determinism:

```json
{
  "status": "ok",
  "primary_batch_size": 10,
  "pair_size": 2,
  "peak_vram_gb": 8.2958612442,
  "pair_rank_loss": 0.6898068190,
  "compute_policy": "cuda_only_no_cpu_fallback"
}
```

Server unit tests passed: five pair-ranking tests and five A4 resource-screen/
launcher tests. Shell syntax checks passed for both A4 launchers. No raw
per-study prediction has been copied or registered for tracking.

The final-promotion checker was separately exercised with its five unit tests,
as was the existing three-seed triage-contract checker (three tests). The final
checker refuses packaging unless all five folds / 338 studies, frozen-Champion
checksum, Macro-F1 and Urgent-F1 hard gates are present and true.

## Terminal protocol

No training log or mid-epoch metrics will be read. At terminal training state,
the only permitted next GPU work is the fixed epoch-15 CUDA audit and the
unchanged five-gate A4 resource screen. Any failed gate ends A4 expansion;
passing still does not promote a model or authorize a submission.

At `2026-09-04T06:13:37Z`, after A4 had started but before any A4 validation,
CUDA-audit, resource-screen, or private-prediction outcome was read, the exact
three-seed continuation contract was locked in
`FOLD0_A4_TRIAGE_SCREEN_PREREGISTRATION.json` and exercised by four unit
tests. It permits only the two remaining fold-0 seeds after a resource-screen
pass; it still forbids automatic cross-fold expansion, pooling/threshold
rescue, promotion, and submission. This timing limitation is explicit in the
preregistration and must remain visible in every later A4 interpretation.

## Triage evaluator portability correction

The initial A4 triage evaluator imported helper functions from historical A1
and A3 scripts. Those scripts are deliberately absent from the clean server
worktree, so this was a hidden dependency in the gate implementation, not a
model or data failure. Before any A4 outcome was read, the evaluator was made
self-contained with the same SHA-256, JSON-atomic-write, single-fold-source,
three-distinct-checkpoint, protocol, and eight-gate checks. Its five local
unit tests include an explicit assertion that it no longer imports either
historical gate. The corrected evaluator must be independently SHA-256 matched
and its five unit tests rerun on the server before it is allowed to decide the
three-seed continuation.

That transfer is now complete. The pre-correction server copies of the
evaluator and its test were preserved under
`/workspace/iaaa_artifacts/server_source_backups/a4_triage_selfcontained_pre_cd537aa`
with SHA-256 values `49b99ff68a7592b20bbeaa392fd2dd581ff92fb5388c99ef2a9fd3859d4033f0`
and `3c94ed92f9a14e62a0b35299a4291abcb40370a1e38cbc2370b3d15f27d08824`,
respectively. The local and server copies of the corrected evaluator, its
test, and this ledger matched byte-for-byte; the server test suite passed
five of five. This correction neither inspected training logs/metrics nor
read any audit or per-study prediction result.

## Locked replication continuation

Before observing the seed-42 resource-screen outcome, the two remaining
fold-0 replications were locked in
`FOLD0_A4_SEED_REPLICATION_PREREGISTRATION.json`. The new launcher accepts
only seeds `2026` and `3407`; it will refuse to run unless the A4 seed-42
resource decision explicitly passes, carries no failed gates, names the A4
candidate/scope, and authorizes exactly those two replications. It derives an
artifact-local manifest from the checksum-bound A4 template and verifies that
the only changes are the run name and `training_config.seed`. Existing A4
artifacts, unsafe secrets permissions, inadequate disk, an existing GPU lock,
or any active CUDA process all cause a refusal. This adds no GPU work now.
