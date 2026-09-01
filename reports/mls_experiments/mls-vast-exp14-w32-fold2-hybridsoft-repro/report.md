# MLS Vast experiment: mls-vast-exp14-w32-fold2-hybridsoft-repro

- Status: `failed_before_model_construction`
- Started UTC: `2026-09-01T15:47:01.373019+00:00`
- Finished UTC: `2026-09-01T15:47:05.620405+00:00`
- Exit code: `1`
- Server commit: `13a74ab2ebab8d10baf71db19abd756240afc6d0`
- Vast instance: `49527185`
- MLflow run: `not_created`
- Completed epochs: `0`
- GPU model compute: `none`
- Auto-destroy: `false`

## Failure boundary

The durable tmux launcher started successfully and wrote authoritative state,
but `runtime.prepare_data=true` failed before model construction. The server
had byte-identical DVC raw data but no derived source file:

```text
Data/processed/mls_dataset/mls_labels.csv
```

The multitask strategy attempted to build the explicit-negative dataset
directly, although that builder depends on the positive MLS dataset. It also
depended on an untracked EDA artifact (`reports/eda/deep/deep_series_table.csv`)
that was absent from the clean server clone.

No model was instantiated, no forward/backward pass ran, no checkpoint was
written, and no training metric was emitted. Therefore this attempt carries no
model-quality evidence and must not be compared with Exp10 or used by the Exp15
launch gate.

## Corrective action

Commit `7c3c6eb` makes preprocessing self-contained:

1. `MLSHeatmapStrategy.prepare_data` builds the prerequisite positive dataset
   when its CSV/images are absent.
2. clean-negative study IDs are derived from the authoritative study maximum
   in DVC-tracked `Data/raw/training_df.pkl`.
3. invalid/missing MLS metadata is rejected explicitly.
4. four metadata tests were added; the combined lightweight MLS suite passed
   20 tests without local CPU model execution.

The immutable retry is preregistered as:

`mls-vast-exp14r1-w32-fold2-hybridsoft-repro`

Its validated training configuration is exactly equal to Exp10 and the failed
Exp14 manifest; only preprocessing infrastructure and the run identifier
changed. The original durable server status/log remain outside Git at
`/workspace/iaaa_artifacts/logs/mls-vast-exp14-w32-fold2-hybridsoft-repro`.
