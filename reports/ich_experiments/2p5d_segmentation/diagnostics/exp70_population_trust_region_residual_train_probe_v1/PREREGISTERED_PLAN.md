# Exp70 preregistration: population trust-region residual train probe

Status: recipe locked before execution

Scope: ICH only; no MLS, fracture, calibration, outer fold, test data, triage
fusion, or leaderboard inference

## Why this experiment exists

Exp69 failed despite active optimization: independently averaged correction CE
and preservation KL gave a small error set the same component mass as millions
of preservation pixels.  A 2.7M-parameter copied decoder learned an IPH shortcut,
improving conditional accuracy by 0.256 percentage points while reducing macro
recall by 10.638 points and harming 24.235% of correct non-IPH subtype pixels.

Exp70 tests the causal redesign, not another coefficient-only retry.  The full
Exp61 incumbent is frozen and a zero-initialized residual adapter with at most
5,000 parameters adjusts only the five foreground logits.  Correction CE and
soft preservation KL are summed over their pixels and divided by one shared
incumbent-foreground population.  Thus their effective mass follows observed
prevalence.  Hard foreground support and classification logits remain locked.

The low-capacity residual design follows the parameter-sharing principle of
residual adapters, while soft teacher preservation follows segmentation
distillation.  Weak class weights are retained only to reduce the known
long-tail bias; they are intentionally capped at 2 rather than allowed to
dominate the population objective.

## Locked recipe

- Incumbent: Exp61 Unet++ with `tu-efficientnetv2_rw_s`, outer fold 2 and
  calibration fold 1 checkpoint.
- Data: only the official three training folds from schema4.
- Probe: aggregate metrics over the same non-augmented training rows; no row
  identifiers or predictions persisted.
- Frozen: incumbent encoder, decoder, segmentation head, classification head,
  BatchNorm statistics, foreground support, and all classification logits.
- Trainable: one `3x3 Conv -> GroupNorm -> SiLU -> 1x1 Conv` residual head.
- Hidden channels: 16; output channels: five foreground subtypes.
- Final convolution: exact zero initialization.
- Residual: `4 * tanh(raw_residual)`.
- Conditional foreground margin: 1.0.
- Loss denominator: total incumbent-foreground pixels in each batch.
- Correction: CE only on known true-foreground incumbent subtype errors,
  coefficient 4.0.
- Preservation: soft KL to incumbent on every other incumbent-foreground
  pixel, coefficient 1.0.
- Class weights: inverse pixel-frequency power 0.25, clipped to `[1, 2]`.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`.
- One epoch, batch size 16, BF16, seed 42.
- No adapter checkpoint is saved or promoted by this probe.

## Locked gates

All gates must pass simultaneously:

1. Trainable parameters are at most 5,000.
2. Initial hard-mask identity is exact.
3. Final foreground-support mismatch is zero.
4. At least 100 incumbent SAH-to-IPH error pixels exist.
5. At least 10% of those SAH-to-IPH errors are recovered.
6. Harm to correct IPH is at most 0.5%.
7. Harm to correct IVH/SDH/EDH is at most 0.5%.
8. Subtype changes on true-background incumbent foreground are at most 0.5%.
9. Conditional subtype accuracy is non-decreasing.
10. Conditional macro recall is non-decreasing.

Passing authorizes exactly one locked patient-safe calibration screen.  It is
not a promotion, leaderboard claim, or permission to inspect outer data.
Failure rejects this recipe before calibration/outer; gates will not be relaxed
after observing the result.

## Primary research basis

- Rebuffi, Bilen, and Vedaldi, *Learning multiple visual domains with residual
  adapters*, 2017: https://arxiv.org/abs/1705.08045
- Michieli and Zanuttigh, *Knowledge Distillation for Incremental Learning in
  Semantic Segmentation*, 2019: https://arxiv.org/abs/1911.03462
- Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*,
  NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
