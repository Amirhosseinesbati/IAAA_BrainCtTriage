# A7 causal review: what the completed comparison does and does not establish

Source inspection after final audit; no new training, checkpoint selection, or held-out prediction inspection.

## Verified code facts

- A7 calls the same `multitask_loss` as the original trainer. Geometry, selector target mode and component weights are inherited from the Exp16 template.
- Its hard-coded scheduler matches the original formula for 23 total epochs: two warmup epochs and a 21-epoch cosine denominator. Both stop at the fixed epoch15; the scheduler is not accidentally compressed to 15 epochs.
- Both recipes use base batch5, accumulation1, AdamW, clipping5 and 541 optimizer updates per epoch. This is not the previous batch10/half-step-budget confound.
- A7 changes the model forward batch from five samples to ten correlated views. Both heatmap BatchNorm and dropout exist in the model. Consequently averaging two supervised losses does NOT make this update equivalent to the original batch5 update: batch statistics, stochastic masks, input distribution and gradients differ.
- The extra deterministic 8-pixel shift is applied after the original random augmentation. Translated views are zero padded, and clipped positive labels are excluded only from their second-view loss. Each view's loss is independently reduced before the two losses are averaged. If eligibility differs, effective per-sample weights can differ. Its population frequency has not been measured here.
- The consistency term measures agreement between independently stochastic views as well as translation agreement. Its benefit cannot be attributed exclusively to geometric equivariance.
- A7 control and consistency share a checksum-bound initialization and equal per-epoch exposure. Their contrast is matched. An exact initialization/input-stream match to the historical baseline has NOT been established merely by reusing seed42 and a template. Identical evaluation runtime does not establish identical training provenance.

## Causal limits

The control lost accuracy relative to the baseline, but this does not identify BatchNorm, dropout, extra augmentation, initialization or floating-point training behavior as the cause. The consistency arm recovered part of that loss and still failed the baseline resource gates. Neither causal certainty nor release eligibility follows from this single fold/seed.

Do not describe the existing architecture as broken. Do not launch a weight sweep, select a later checkpoint, or recalibrate on the 70 held-out studies to rescue A7.

## Next bounded diagnostic before another training run

Reuse the previously fixed training-only translation-probe population, not a newly selected set of errors. Evaluate baseline, A7 control and A7 consistency with identical inference settings on that same population, without changing model weights or BN running buffers. Compare translation sensitivity and original-view landmark/MLS errors in aggregate. This distinguishes whether A7 improved the measured mechanism at all versus merely changing held-out thresholds. It does not itself authorize promotion, TTA selection, or a causal claim about BN.

Before implementation, inspect the existing probe's actual sample selection, source pins, and output contract. If the original fixed population cannot be reproduced exactly, report that instead of silently substituting a new population. Register checkpoint hashes and the fixed comparison before running; use one CUDA workload under the campaign lock. Keep individual study/slice outputs server-only. Decide the next training hypothesis only after this mechanistic evidence, while retaining all final triage gates.

Inspection confirmed the dataset and original trainer still match the pinned A7 hashes: `df0852f12eba9329e3a65787d65e9649922d6b8c2704b2d7d47192ec33c9ac1c` and `5135d38b193a59079c763ea8003e66e7d5da1a81be94355b82fead1c4eb81cc3`. No source changes were made during this review.
