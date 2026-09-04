# A5 execution ledger — fold 0 / seed 42

- Status at recording: **ready to launch; no A5 training has started.**
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
