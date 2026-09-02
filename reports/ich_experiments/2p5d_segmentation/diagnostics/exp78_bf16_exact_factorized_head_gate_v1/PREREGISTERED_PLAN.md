# Exp78 — BF16-exact factorized residual composition gate

## Hypothesis fixed before execution

Exp77 Phase A showed that the original factorized algebra is accurate to about
`1e-7` in probability, yet BF16 log-sum-exp cancellation can flip a few
near-tied hard-mask pixels and violate the locked epoch-zero metric tolerance.
The gate is not relaxed. Exp78 changes only the numerically stable composition:

- keep the legacy background and foreground logits as the forward reference;
- add the subtype residual after centering it by the difference between legacy
  and adjusted foreground `logsumexp` values, preserving total foreground mass;
- add the scalar foreground residual equally to all five centered foreground
  logits;
- perform this residual composition in FP32 inside the BF16 autocast pipeline.

With zero residuals, the returned logits—not only probabilities—must equal the
BF16 legacy logits exactly. For nonzero residuals the foreground branch changes
only total foreground odds and the subtype branch changes only the conditional
five-class distribution.

## Locked technical protocol and gates

- Same Exp61 checkpoint and code/data provenance as Exp76/77.
- Synthetic seed `42`, batch `2`, `9×384×384`, but the real forward path is now
  explicitly under CUDA BF16 autocast.
- No optimizer, patient images, calibration, outer or test access.
- All related tests must pass, including a CPU-BF16 exact-logit regression test.
- Maximum logit and classification-logit difference exactly zero.
- Maximum probability difference at most `2e-6`; argmax mismatch exactly zero.
- Both zero-initialized residual outputs exactly zero.
- Both cross-gradient maxima at most `1e-6`.
- Encoder/classifier frozen, spatial trainable scope and module modes correct,
  every aggregate finite, MLflow and analytical Persian Telegram recorded.

If every gate passes, the exact same Exp77 four-step recipe is repeated under a
new run name. Its performance values still may not alter the already locked
three-epoch recipe; full calibration is authorized only if epoch-zero metric
identity now passes the original `1e-6` tolerance.
