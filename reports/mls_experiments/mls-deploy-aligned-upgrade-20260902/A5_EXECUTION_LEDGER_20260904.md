# A5 execution ledger — fold 0 / seed 42

- Status at latest update: **training completed; fixed CUDA resource screen running; no A5 terminal metric has been read.**
- Candidate: `mls-vast-da-a5-detached-rank-fold0-seed42`.
- Compute policy: `cuda_only_no_cpu_fallback`; exactly one GPU workload.
- Local source commits: `bc274d1` (initial contract) and `7c735a0`
  (pre-training BatchNorm-normalization correction).

## Immutable source and transfer evidence

Five overwritten remote files were backed up under
`/workspace/iaaa_artifacts/server_source_backups/a5_detached_rank_pre_bc274d1`;
the later model/test/plan draft was backed up under
`/workspace/iaaa_artifacts/server_source_backups/a5_normalization_pre_7c735a0`.
All twelve initial source/manifest/test/report files were SHA-256 matched after
transfer.  The four corrected files matched again with these hashes:

| File | SHA-256 |
|---|---|
| `src/strategies/mls_heatmap/model.py` | `51d6b53572fd0ad720290be802c1cd3a0b4714433c680bf22c8298e501822e0e` |
| `tests/test_mls_detached_rank_selector_cuda.py` | `80bc0e75906e10aac597bc644cbe248ff37d676c7c263fe6ada2ff41390fddda` |
| `A5_DETACHED_RANK_PREREGISTERED_PLAN_20260904.md` | `4159be414d21355353061b1e657bae8e14b7c3d540cd36b9b27b5c52cfab087f` |
| `A5_PREFLIGHT_RECONCILIATION_20260904.md` | `cb604e8b7a666483624bf18969f76e4a183c0942a561b37bea850f22e44ef194` |

## CUDA test and preflight evidence

The server ran twelve A5-related unit tests, including the CUDA assertion that
the rank-only path gives selector-head gradients while leaving backbone and
heatmap-head gradients absent, BatchNorm running mean/variance unchanged, and
BatchNorm tracking restored.  All passed (`12/12`).  Both A5 launchers passed
`bash -n`.

The first synthetic preflight identified an eval-mode normalization mismatch and
was invalidated before training; its reconciliation is recorded separately.
The corrected, distinct v2 preflight then passed on the idle RTX 3090:

```json
{
  "status": "ok",
  "cuda_device": "NVIDIA GeForce RTX 3090",
  "primary_batch_size": 10,
  "pair_size": 2,
  "pair_rank_loss": 0.6688146591,
  "peak_vram_gb": 8.2958612442,
  "rank_backbone_detached": true,
  "compute_policy": "cuda_only_no_cpu_fallback",
  "training_determinism": "strict"
}
```

## Locked execution path

The only next GPU workload is A5 fold-0/seed-42 training from the
checksum-bound manifest.  No training logs or mid-epoch metrics may be read.
At a terminal training state, the only permitted next GPU workload is the
fixed epoch-15 70-study CUDA audit and the five-gate A5 resource screen.  A
rejection ends A5; a pass is reported for manual replication-preregistration
only and never promotes, packages, or submits a model.

## Launch confirmation

The guarded training launcher created the atomic status at
`2026-09-04T07:49:34Z` with state `running`, null exit code, the expected run
name, the checksum-bound training-manifest snapshot
`850966679fb9d7223c297a37d66da726fa55dab4aa94f6d98b77c0e932eaadfd`,
and compute policy `cuda_only_no_cpu_fallback`.  A later terminal-only check
confirmed the corresponding tmux session was live and exactly one CUDA process
was present.  No training log, epoch output, validation metric, or prediction
was opened by that check.

## Training completion and resource-screen launch

The scheduled terminal-only check at `2026-09-04T09:11:05Z` found training
`completed` with exit code `0`, a missing training tmux session, and no CUDA
compute process. The terminal status records completion at
`2026-09-04T08:30:29Z` (40 minutes 55 seconds after launch) and the same
training-manifest SHA-256. No training log or intermediate metric was read.

Before the screen, the decision-publication correction in local commit
`897fcfc` was deployed with SHA-256 checks and eight passing target-server
tests; see `A5_DECISION_CONTRACT_CORRECTION_20260904.md`. This changes decision
metadata/publication only, not training or the five metric gates.

The guarded launcher found no existing resource-screen status/session or GPU
lock, and no CUDA workload. It started the fixed epoch-15 audit in tmux
`mls_da_a5_f0_resource`. At `2026-09-04T09:11:45Z`, launcher status was
`running` with null exit code, `cuda_only_no_cpu_fallback`, and a confirmed
live tmux session. The expected population is 70 fold-0 studies, batch size 16.
The next evidence to inspect is its terminal launcher state followed by the
aggregate resource decision; raw per-study predictions remain server-only.
