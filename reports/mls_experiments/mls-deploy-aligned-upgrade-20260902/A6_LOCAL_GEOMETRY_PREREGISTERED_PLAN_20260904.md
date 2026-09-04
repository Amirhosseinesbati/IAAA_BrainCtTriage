# A6: peak-local training geometry, fixed deployment

Status: preregistered before A6 preflight, training, or evaluation.

## Evidence and hypothesis

The frozen baseline training-only probe (128 target slices, 89 studies) found
a mean absolute MLS difference of 1.045275 mm between global soft-argmax and
the deployed DARK decoder. Their training-sample MAEs were 0.921113 and
1.461234 mm. This is mechanistic evidence, not held-out accuracy or proof that
changing a decoder would help. See the probe report for selection and scope.

SciSpace and the primary
[BCIR paper](https://arxiv.org/abs/2301.10431) identify potential bias and
gradient limitations in integral regression on pose benchmarks. Together with
[DARK](https://arxiv.org/abs/1910.06278), this motivates the representation
question. A6 is a new peak-local training hypothesis, not a reproduction of
BCIR or a literature-established solution for this MLS dataset.

## One defined intervention

Replace only the coordinate decoder used by the primary multitask coordinate,
MLS and 1/3/5-mm threshold losses with local soft-argmax:

- Center a square window at each detached discrete heatmap maximum.
- Fix radius to 6 heatmap pixels (13x13 window), twice the existing target
  Gaussian sigma of 3; temperature remains 1.
- Normalize logits within that window and take their coordinate expectation.
- Map coordinates using image-size/heatmap-size, matching DARK's spatial
  convention. This coordinate convention is explicitly part of the proposed
  representation change; the experiment does not isolate locality from scale.
- Retain the whole-map Gaussian cross-entropy, all loss weights and selector
  supervision. It supplies global localization gradients when a peak is wrong.

The window index is discrete; the expectation has gradients only within the
selected window. Wrong peaks, peak switching, truncation at image borders and
poor local gradients are genuine failure risks. CUDA tests must check finite
outputs/gradients, coordinate scaling, a distant secondary peak, and exact
legacy-global behavior. A synthetic full HRNet-W32 multitask optimizer step
must pass on the target GPU before training.

## Matched optimization and resource contract

Use the original baseline configuration: batch 5, workers 2, accumulation 1,
23 epochs, fixed epoch 15, LR 0.0001 with the unchanged epoch-based schedule,
HRNet-W32, 512 px, strict determinism, no AMP, fold 0, seed 42. The current
checksum-bound data imply 541 ordinary steps per epoch and 8,115 by epoch 15.
Rank, bag, signed-offset and independent ordinal auxiliaries are all disabled.
Normalized training-config comparison against the baseline must differ only
in `training_geometry_decoder`; the radius has the same unused legacy default.
Do not silently increase batch size for this experiment.

Run: `mls-vast-da-a6-local-geometry-fold0-seed42`.
No inference, validation-proxy decoding, threshold, pooling or triage rule
changes. Ordinary validation loss follows the selected training geometry;
existing hard-argmax validation proxy metrics remain unchanged and are not
used for this decision. Audit only epoch 15 on all 70 held-out fold-0 studies,
using the unchanged CUDA DARK / gate 0.5 / top3 / p90 resource profile.

| Metric | Fixed requirement |
|---|---:|
| Study MAE | <= 1.4709586392 mm |
| F1 at 3 mm | >= 0.8196721311 |
| F1 at 5 mm | >= 0.7368421053 |
| Boundary F1 | >= 0.7782571182 |
| Selection objective | <= 1.9044444028 |

Any failed gate rejects A6 and stops expansion. Passing is resource-screen
success only; seed replication requires a separately frozen protocol. It does
not authorize pooling/checkpoint searches, promotion or submission. The final
goal still requires improved frozen-Champion triage Macro-F1 and Urgent F1
with every full-coverage hard gate in `TRIAGE_EVALUATION_PREREGISTRATION.json`.

## Execution and preservation

All tests, model operations and training run on the Vast server. CUDA-only
model compute; one workload at a time. Source edits stay in the clean local
worktree, then are transferred with backups and SHA-256 checks. Training
records its exact manifest; no logs or mid-epoch metrics are inspected.
Observe terminal milestones at the established 30-minute cadence. MLflow and
local retention receive only aggregate evidence; raw study predictions remain
server-only. The server is not stopped or destroyed automatically.
