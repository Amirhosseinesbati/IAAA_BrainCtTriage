# Exp72 — Decoupled foreground/subtype gradient probe v1

## Decision being tested

The frozen-adapter branch is closed by Exp69–Exp71. Exp72 tests a representation-
level supervision change: preserve the six-logit inference contract, but optimize
hemorrhage support separately from subtype discrimination.

The foreground logit is exactly
`logsumexp(logits[1:]) - logits[0]`; therefore its sigmoid equals the sum of the
five foreground softmax probabilities. Conditional five-way cross-entropy and a
small one-vs-rest focal term see only known true-foreground pixels. The hypothesis
is that this removes the overwhelming background population from direct subtype
competition while retaining explicit foreground/FPR pressure.

Primary motivation is the observed long-tailed subtype supervision and the
failure of frozen feature adapters. Balanced Softmax and Seesaw Loss provide
independent evidence that ordinary softmax gradients can be biased by long-tail
class counts and that majority negatives can suppress tail categories:

- https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.html

## Locked inputs

- Incumbent checkpoint: Exp61 EfficientNetV2-S, SHA-256
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Manifest: schema4 server manifest, SHA-256
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.
- Split: train only. Calibration, outer, test, and leaderboard data are forbidden.
- No optimizer and no parameter updates.
- Seed: checkpoint seed (42); 24 deterministic sampled batches, batch size 8.
- Model runs in evaluation mode; BF16 forward with FP32 loss arithmetic.
- Decoder plus segmentation-head gradients are measured. No row-level prediction
  artifact is written or uploaded.

## Candidate objective

- Foreground Dice weight: 0.40.
- Foreground focal weight: 0.20.
- Conditional five-way subtype CE weight: 0.30.
- Conditional subtype one-vs-rest focal weight: 0.10.
- Existing foreground class weights, background weight, empty-foreground weight,
  and top-fraction are inherited from the checkpoint config.

## Preregistered gates

All gates must pass before a calibration-only warm-start experiment is allowed:

1. Unit tests prove exact foreground probability identity, six-channel output
   compatibility, finite backward gradients, and zero subtype gradient on true
   background pixels.
2. No model parameter changes; checkpoint hash is unchanged.
3. Every foreground subtype has at least 100 supervised pixels in the probe.
4. Mean absolute target-logit attraction ratio (candidate / incumbent objective)
   is at least 1.10 for EDH and 1.25 for SAH.
5. Candidate/incumbent mean absolute foreground-channel gradient ratio on true
   background is at most 1.50.
6. Mean decoder/head gradient cosine between objectives is at least 0.10.
7. All aggregate losses, norms, cosines, and ratios are finite.

Passing this diagnostic only authorizes a separately preregistered calibration-
only warm-start. It cannot promote a checkpoint. Failure closes this exact loss
weighting; it does not by itself reject every hierarchical formulation.
