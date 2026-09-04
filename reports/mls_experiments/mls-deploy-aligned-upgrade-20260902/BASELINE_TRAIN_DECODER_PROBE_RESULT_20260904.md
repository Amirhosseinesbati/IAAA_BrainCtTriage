# Baseline decoder probe: a measurable training/deployment gap

## Fixed diagnostic completed

The preregistered Exp16 fold-0/seed-42/epoch-15 checkpoint was evaluated once
on the fixed 128 eligible training slices, representing 89 studies. All 128
forward outputs and decoded coordinate sets were finite and valid. No
held-out images were used. Checkpoint, labels and fold hashes matched the
preregistration. GPU forward ran on the RTX 3090; weights were not updated.

Three tests passed on the target server before execution: deterministic sample
selection with held-out-overlap rejection, physical-distance aggregation on a
known geometry, and rejection of invalid decoder coordinates. All four
transferred source/test/preregistration/report files passed SHA-256 checks.
Source implementation commit: `ae96ba8`; preregistration commit: `6718323`.

## Observed aggregate results

| Training-sample metric | Soft-argmax | DARK |
|---|---:|---:|
| Slice MLS MAE, mm | 0.921113 | 1.461234 |
| Slice MLS signed bias, mm | -0.379489 | -0.329453 |
| F1 at 3 mm | 0.900763 | 0.854962 |
| F1 at 5 mm | 0.945055 | 0.869565 |
| Anterior landmark mean error, mm | 1.710188 | 2.034939 |
| Posterior landmark mean error, mm | 1.735064 | 1.805857 |
| Outer-falx landmark mean error, mm | 6.010192 | 6.387321 |

For identical logits, inter-decoder coordinate distance had mean
`2.069739 mm`, median `1.819628 mm` and p90 `3.765382 mm`. Mean absolute
inter-decoder MLS difference was `1.045275 mm`. DARK's slice MLS MAE exceeded
the supervised soft-argmax MAE by approximately `0.540122 mm` in this sample.
The diagnostic itself took `6.23 seconds`, excluding subsequent MLflow
network logging.

## What this does and does not establish

There is a substantial decoder discrepancy on these baseline training inputs.
It is reasonable to investigate a training objective whose geometry better
matches the peak-local coordinate representation used at deployment.

However, the model's geometry objective explicitly optimizes soft-argmax, so
its advantage on training data is not independent evidence of better
generalization. This sample contains annotated target slices, not the complete
study population or the selector's negative examples. Some slices share a
study. Inputs are processed training PNGs without augmentation; this does not
audit raw-DICOM preprocessing parity. No confidence interval, clinical accuracy
claim, triage improvement, or leaderboard gain follows from these numbers.

The deployed decoder and existing models remain unchanged. This result is not
permission to substitute soft-argmax in an old submission or to search old
checkpoints/pooling profiles. A distinct training intervention must have its
own fixed protocol, matched baseline optimization budget, and subsequent
complete-study and triage gates. The rejected A4/A5 recipes remain rejected.

Before another training run, the next concrete design question is how to align
geometry supervision with local heatmap peaks while retaining stable gradients
and the existing DARK deployment path. Any implementation needs CUDA numerical
tests and a preregistered comparison at baseline batch 5/workers 2 (or an
explicitly controlled alternative), given the optimizer-budget confound found
in `POST_A5_CAUSAL_AUDIT_20260904.md`.

## Tracking and retained evidence

MLflow status: `logged`, dedicated run
`3d21659b52604f90997f636237c83ebb`, in the existing MLS experiment 16:
[training decoder probe](https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/3d21659b52604f90997f636237c83ebb).
Only aggregate metrics and the aggregate JSON were uploaded.

Local evidence is under
`server_aggregate/baseline_train_decoder_probe_20260904/`.
The result binds sample/checkpoint/data/source hashes without publishing sample
identifiers, coordinates or individual predictions.

| File | SHA-256 |
|---|---|
| `baseline_train_decoder_probe_20260904.json` | `030edeabf30710151e3e5cb6a2b0cb47dd8ac7c3a012d09be56b54462f72d2fb` |
| `baseline_train_decoder_probe_20260904.status.json` | `c2c1bf379dec7ceb6c752390d7ae51d027b4b72ea742b88d06f4d4ab7b057feb` |

The two files were copied from the campaign artifact directory on the server;
local and remote SHA-256 values matched. No model or private prediction file
was copied, and no new training run or release was created by this diagnostic.
