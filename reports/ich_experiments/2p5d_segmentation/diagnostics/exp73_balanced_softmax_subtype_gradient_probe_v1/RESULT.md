# Exp73 result — Balanced Softmax recipe rejected

The preregistered train-only probe completed on commit `6840f33`, with no
optimizer, calibration, outer, test, or row-level prediction use. All 85 related
tests passed. MLflow run: `71c281f9852940d083109911c1a7804f`.

Balanced Softmax improved EDH target-logit attraction from Exp72's `0.714x` to
`1.384x`, passing its gate. It did not improve the complete allocation: IPH
remained over-amplified at `2.541x`; IVH=`0.648x`, SDH=`0.429x`, and SAH=`0.939x`
failed their gates. Structural safety remained good: true-background gradient
ratio=`0.614x` and decoder/head cosine=`0.543`.

Decision: `reject_exact_loss_weighting_before_calibration_or_outer`. The result
shows that class-prior correction alone cannot isolate subtype learning because
the binary foreground objective still backpropagates through `logsumexp` in
proportion to current subtype probabilities. The next design must make the
foreground gradient common-mode across subtype logits and evaluate actual
subtype-margin gradients, not absolute target-channel gradients.

Aggregate artifact SHA-256:
`6c54bed7c9edc7df89d4868038b4d4d7ef67f05a829003b953cab3ddb946f77a`.
