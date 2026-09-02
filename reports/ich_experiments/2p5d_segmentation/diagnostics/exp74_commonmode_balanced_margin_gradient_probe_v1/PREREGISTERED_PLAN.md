# Exp74 — Common-mode foreground + Balanced Softmax margin probe

## Causal change

Exp73 proved that class priors can strengthen EDH, but IPH remained dominant.
The binary foreground logit has an exact forward definition
`logsumexp(subtypes)-background`; its ordinary derivative distributes foreground
pressure according to current subtype probabilities and therefore changes subtype
margins even though the objective is intended to learn support only.

Exp74 preserves that exact forward value and all inference probabilities, but uses
a straight-through common-mode derivative: every foreground logit receives the
same `1/5` derivative and background receives `-1`. Consequently, foreground loss
cannot alter any pairwise subtype margin. Conditional Balanced Softmax is the only
source of subtype-margin gradients.

## Locked protocol

- Same Exp61 checkpoint, schema4 manifest, train-only 24 batches, batch size 8,
  seed 42, BF16 forward, evaluation mode, and no optimizer as Exp72/73.
- No calibration, outer, test, leaderboard, or row-level prediction access.
- Weights: foreground Dice `0.40`, foreground focal `0.20`, conditional Balanced
  Softmax `0.40`, subtype OVR `0.00`.
- Train-mask pixel counts are the only subtype priors.
- The primary subtype metric is signed target-margin attraction:
  `mean(other_foreground_gradient) - target_gradient`. Positive values mean a
  gradient-descent step raises the true subtype relative to its competitors.

## Preregistered gates

1. Tests prove exact forward probability identity and exactly equal `0.2`
   foreground derivatives under common-mode backward.
2. All five subtypes have at least 100 supervised pixels and all aggregates are
   finite.
3. Candidate/incumbent margin-attraction ratio: EDH at least `1.10`, SAH at least
   `1.25`, IVH and SDH each at least `0.75`, and IPH between `0.75` and `1.75`.
4. True-background foreground-channel absolute-gradient ratio is at most `1.50`.
5. Decoder/head gradient cosine is at least `0.10`.

Passing authorizes only a separately preregistered calibration-only warm start.
Failure rejects this exact common-mode/Balanced-Softmax recipe before held-out use.
