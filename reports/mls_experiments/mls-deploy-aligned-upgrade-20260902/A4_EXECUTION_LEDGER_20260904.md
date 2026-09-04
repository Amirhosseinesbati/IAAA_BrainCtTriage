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

The replication contract was committed as `7901f7c`. Its launcher,
preregistration, test, and this ledger were SHA-256 matched after transfer to
the clean server worktree; `bash -n` and the four server-side launcher tests
passed. The launcher remains inert unless the later resource-decision JSON is
the exact passing authorization, so no replication was started during this
setup work.

## Post-launch implementation audit

Without opening an A4 training log, metric, audit artifact, or private
prediction, the pair-ranking path was reread end-to-end. The `patient_id`
field in the MLS-label table is normalized to the study-series identifier
before splitting, sampling, grouping, and returning a sample, so the pair
loader is genuinely same-study despite the legacy column name. The RankNet
label is computed from a perpendicular MLS distance. Its independent pair
augmentations are rotation and translation only, both rigid transforms that
preserve this distance and therefore preserve the local-MLS order. Commit
`d351097` adds a CPU-light regression test that applies distinct rigid
transforms to both members and proves the rank target remains `3.0 > 1.5 mm`;
the A4-related unit suite passed 20/20. No active recipe or outcome was
changed by this audit.

## Locked three-seed audit and triage continuation

`FOLD0_A4_THREE_SEED_AUDIT_PREREGISTRATION.json` and its matching launcher
were created before reading any A4 outcome. They bind exactly fold 0, epoch
15, seeds `42/2026/3407`, the fixed baseline audit hashes, the frozen Champion
branch hash, fixed deploy-aligned comparison, and the existing A4 triage gate.
The launcher cannot overwrite an existing candidate audit or triage result;
it verifies the passed resource decision, terminal statuses and seeds of all
three trainings, every fixed checkpoint path, both baseline hashes, and the
frozen-branch hash before acquiring a GPU through the existing one-workload
audit wrapper. Its CUDA audit uses batch 6 for the concurrent three-model
ensemble. The subsequent 70-row triage comparison is CPU-light, retains its
per-study files only on the server, and ends at the A4 gate—never a promotion
or submission.

This gate was committed as `f0c05fe`; its four transferred files were
SHA-256 matched on the server, passed `bash -n`, and passed four server-side
unit tests. While the resource decision was absent, an explicit dynamic
refusal-path check exited `3` at the required-artifact precondition and proved
that it had created neither the candidate CUDA-audit directory nor the triage
comparison directory. It therefore cannot accidentally start an audit while
seed-42 training is active.
