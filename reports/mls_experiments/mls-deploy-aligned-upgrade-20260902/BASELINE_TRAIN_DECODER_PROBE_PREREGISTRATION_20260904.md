# Baseline training-only decoder probe — preregistration

Status: frozen before implementation or any probe outcome. This is one
mechanistic diagnostic, not another training run, validation screen or
post-hoc pooling search.

## Inputs and population

- Baseline only: historical Exp16, fold 0, seed 42, epoch 15.
- Checkpoint:
  `/workspace/IAAA_BrainCtTriage_mls_da/models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth`.
- Required checkpoint SHA-256:
  `c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062`.
- Immutable patient-grouped fold manifest SHA-256:
  `d3c4640aec8fbfd8a912286bbf40ee39a7f48756c899cafcf8d976ce664ce2b8`.
- Processed labels SHA-256:
  `01512662b62bcaf484f99cb872c40e28e2cfb300adee60db40957db0d06001ad`.
- Use the training subset returned by the current competition-fold loader;
  exclude all fold-0 studies. Use only target rows with all three valid
  landmark annotations. Augmentation is disabled.
- Select at most 128 rows by sorting the SHA-256 of
  `20260904|normalized_study_id|image_name`; require at least 64 eligible rows.
  Do not resample based on predictions or error. Report sample size and the
  number of represented studies, not their identifiers.

## Execution

Run on the Vast GPU with a guard against another CUDA workload. Load only the
specified checkpoint; missing/hash-mismatched input is a technical failure,
not permission to choose a different model. CUDA forward pass in evaluation
mode, batch 8, no gradient updates. All computation and tests run on the server.
Ordinary lightweight data loading and the existing DARK decoder stay on the
server; there is no CPU model-forward fallback.

For exactly the same logits, compare the existing training soft-argmax and
existing deployed DARK decoder at the checkpoint's image size/temperature.
Use the original annotation coordinates and physical spacing for both.
Do not alter decoder parameters, checkpoint, labels, study aggregation, or
triage rules, and do not inspect any held-out images or predictions.

## Allowed aggregate output

- Input/source hashes, sample/study counts, CUDA device, runtime, finite-output
  and valid-coordinate counts.
- Per-landmark mean and median distance to annotations in millimetres, for
  both decoders.
- Mean/median/90th-percentile inter-decoder coordinate distance in millimetres.
- Slice MLS MAE, signed bias and 3/5-mm F1 for each decoder, plus mean absolute
  inter-decoder MLS difference. These are descriptive training-sample metrics.
- No per-slice/per-study coordinates, identifiers, predictions or images in
  the saved output, local transfer, or MLflow. Log only aggregate evidence.

## Interpretation and stop rules

This probe can quantify whether geometry supervision and deployed decoding
disagree on baseline training inputs. It cannot establish generalization,
clinical accuracy, causal attribution for A4/A5, or a leaderboard improvement.
No alternative checkpoint or diagnostic parameter sweep follows its result.
If the gap is small, do not invent a decoder-alignment intervention merely to
continue training. If meaningful disagreement appears, design one distinct
training intervention with a separately frozen experiment and matched baseline
optimization settings; the current release and decoder remain unchanged.
No seed replication, model promotion or submission is authorized by the probe.
