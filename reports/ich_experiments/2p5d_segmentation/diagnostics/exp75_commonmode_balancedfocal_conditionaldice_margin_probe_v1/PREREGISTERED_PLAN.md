# Exp75 — Common-mode + Balanced Focal + conditional Dice margin probe

## Hypothesis fixed before execution

Exp74 successfully removed foreground-to-subtype coupling, but conditional
Balanced Softmax alone supplied only `0.114x` of incumbent SDH margin and `1.022x`
of SAH margin. The legacy multiclass objective contains class-wise Dice pressure
that is valuable for diffuse morphology; removing it discarded more than
background competition.

Exp75 retains exact-forward/common-mode-backward foreground decoupling and adds a
five-class Dice computed only inside known true-foreground pixels. It therefore
learns subtype overlap without allowing background pixels into subtype
competition. Balanced Softmax receives focal gamma `2` so easy correct foreground
pixels contribute less than hard subtype confusions.

## Locked protocol and objective

- Same Exp61 checkpoint, schema4 train split, seed 42, 24 batches, batch size 8,
  BF16 forward, eval mode, no optimizer and no held-out access.
- Foreground Dice `0.30`.
- Foreground focal `0.15`.
- Conditional subtype Dice `0.35`.
- Conditional Balanced Softmax focal `0.20`, gamma `2.0`.
- Subtype OVR `0.00`.
- Common-mode foreground gradient and train-mask pixel priors.
- No row-level prediction artifact.

## Preregistered gates

The same signed subtype-margin gates as Exp74 are retained without relaxation:

- EDH at least `1.10x`; SAH at least `1.25x`.
- IVH and SDH each at least `0.75x`.
- IPH between `0.75x` and `1.75x`.
- True-background foreground gradient at most `1.50x`.
- Decoder/head gradient cosine at least `0.10`.
- At least 100 pixels per subtype; all aggregate values finite; all tests pass.

Passing authorizes a separately preregistered calibration-only warm start. Failure
rejects this exact morphology-aware hierarchical recipe before held-out use.
