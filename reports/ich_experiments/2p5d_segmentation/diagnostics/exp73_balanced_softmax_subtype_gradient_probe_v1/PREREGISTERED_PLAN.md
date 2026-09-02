# Exp73 — Conditional Balanced Softmax gradient probe v1

## Rationale fixed before execution

Exp72 showed that hierarchy is structurally safe (true-background gradient
`0.614x`, decoder/head cosine `0.564`) but its standard conditional CE plus OVR
mixture over-amplified IPH (`2.475x`) while EDH fell (`0.714x`) and SAH reached
only `1.097x`. Increasing the global subtype coefficient would worsen that
allocation and is forbidden.

Exp73 replaces standard conditional CE plus independent OVR with conditional
Balanced Softmax. Pixel counts from spatially supervised train masks are added as
log priors to the five subtype logits during loss computation. This makes a
head-class IPH target easier and a tail target harder without changing inference
logits. This follows the primary Balanced Meta-Softmax result rather than a
post-hoc class-specific coefficient fit:

- https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html

## Locked protocol

- Exp61 checkpoint and schema4 manifest with the same SHA-256 values as Exp72.
- Train split only; calibration, outer, test, and leaderboard reads are forbidden.
- Seed 42, 24 batches, batch size 8, BF16 forward, evaluation mode.
- No optimizer, no parameter update, no prediction-row artifact.
- Foreground Dice `0.40`, foreground focal `0.20`, conditional Balanced Softmax
  `0.40`, subtype OVR `0.00`.
- Priors come only from supervised train-mask pixel counts in fixed subtype order.

## Preregistered gates

All gates are conjunctive:

1. All unit/integration tests pass and every aggregate is finite.
2. At least 100 supervised pixels exist for each subtype.
3. EDH target-attraction ratio is at least `1.10`; SAH is at least `1.25`.
4. IPH target-attraction ratio is at most `1.75`.
5. IVH and SDH target-attraction ratios are each at least `0.75`.
6. Candidate/incumbent foreground-channel gradient ratio on true background is at
   most `1.50`.
7. Mean decoder/head gradient cosine is at least `0.10`.

Passing only authorizes a separately preregistered calibration-only warm start.
Failure rejects this exact Balanced Softmax recipe before any held-out inference.
