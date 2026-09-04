# Post-A5 audit: optimization confound and next diagnostic

## A correction to causal interpretation

The rejection of the measured A4 and A5 recipes stands. However, their
comparison with the incumbent was not a single-factor ranking ablation:
batch size and worker count changed too. Previous wording implying that the
ranking loss alone explained regression against the incumbent was too strong.
This is a limitation of our experimental design, not evidence of corrupted
data or an invalid numerical resource-screen result.

The actual server manifest snapshots for baseline seeds 2026/3407 specify
batch 5 and two workers. A4/A5 specify batch 10 and four workers. All have
23 epochs, learning rate 0.0001, accumulation 1, and a fixed epoch-15 audit.
The historical seed-42 control is Exp16; its stored report independently
records batch 5, workers 2, and the same schedule.

The metadata-only audit used the actual project DataLoader construction on
the server, without iterating it, loading images/checkpoints, or instantiating
a model. It found 2,706 training rows and 778 validation rows. Sampling draws
the training-row count with replacement and drops the last incomplete batch.

| Quantity | Baseline, batch 5 | A4/A5, batch 10 |
|---|---:|---:|
| Primary batches per epoch | 541 | 270 |
| Optimizer updates through epoch 15 | 8,115 | 4,050 |
| Primary sample draws through epoch 15 | 40,575 | 40,500 |
| Recorded workers | 2 | 4 |

These counts are derived from current checksum-bound data and the inspected
training loop; they are not historical optimizer-step telemetry. The loop
adds rank gradients before the ordinary optimizer step, so the 68 scheduled
pair batches per A4/A5 epoch do not add separate optimizer updates. The cosine
scheduler advances once per epoch. Consequently, epoch-15 comparisons do not
match the number of weight updates despite nearly matching primary sample
draws. BatchNorm behavior, gradient noise, and worker-seeded augmentation can
also differ. The audit does not isolate which factor caused the observed loss.

A5 versus A4 still controls batch, workers and schedule, so its failure to
improve overall performance remains informative about the proposed detached
rank update in that batch-10 setting. Neither experiment justifies claiming
that every possible ranking method is ineffective. No rejected checkpoint is
rescued by this retrospective explanation.

## Training/inference decoder difference

Code inspection establishes another testable mechanism:

- `train_multitask.multitask_loss` supervises geometry and thresholds through
  `train.differentiable_keypoints_from_heatmaps` (soft-argmax).
- `predict_multitask.predict_reader_slices` applies spatial softmax and then
  `utils.decode_heatmap_dark_batch` (DARK) for deployed coordinates.

This difference is not itself a bug. A multimodal or diffuse heatmap can give
different expected and peak-local coordinates, but its magnitude in this
project has not been measured by this audit. It warrants one fixed diagnostic
on baseline **training** examples before implementing a new loss. No decoder,
pooling, threshold, or held-out prediction is changed here.

## Research consulted and limits

SciSpace was queried for CT midline landmark methods and anatomical/context
constraints. Abstract-level evidence from
[CAR-Net](https://arxiv.org/abs/2007.05393) supports exploring structural
connectivity and pose-aware context, but its curve annotations and datasets
are not automatically compatible with this competition's three landmarks.
It is a possible later direction, not justification for launching a new
architecture immediately.

[Hoffer et al.](https://arxiv.org/abs/1705.08741) report that fewer updates can
explain large-batch generalization gaps in their classification experiments.
This motivates auditing update counts here; it does not prove the cause of
our MLS results, and batch sizes 5/10 are not their large-batch setting.

[DARK](https://arxiv.org/abs/1910.06278) studies the importance of coordinate
encoding and decoding in pose estimation. It motivates measuring our existing
decoder gap; it does not establish that replacing DARK or using a new MLS loss
will improve this competition.

## Actionable next step

1. Keep A4 and A5 rejected with their existing gates and original metrics.
2. Run the separately preregistered baseline training-only decoder probe.
3. Use its aggregate evidence to decide whether a distinct localization
   intervention is justified; do not use it as a validation score.
4. The next scientific candidate must match baseline batch 5, workers 2,
   schedule and update budget unless a different optimization protocol is
   explicitly preregistered and controlled. Using a 3090 does not by itself
   make a larger batch scientifically equivalent.
5. Any candidate still needs the full fixed triage gates, including improved
   frozen-context Macro-F1 and Urgent F1, before promotion.

## Reproducibility

Script: `scripts/audit_mls_optimizer_budget.py`.
Remote execution completed successfully; no local Python/test/model execution
was used. Aggregate evidence: `POST_A5_OPTIMIZER_BUDGET_20260904.json`, copied
from `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/post_a5_optimizer_budget_20260904.json`.
SHA-256: `57274b4be3560e93198f138e50fe9b58b283da1cbe4453c0269094f64e6d5cf2`.
The JSON binds the four manifest hashes, labels, fold manifest and source.
