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

The A4 plan prospectively locked only this one-seed resource screen and its
five metrics. It did **not** contain a dedicated A4 three-seed triage
preregistration before the A4 run started. Therefore even a resource-screen
pass must not auto-start seed replications or cross-fold runs: first record a
new prospective A4 continuation protocol without inspecting private
predictions. This protects the leak-free claim more strongly than retrofitting
a screen after an outcome exists.
