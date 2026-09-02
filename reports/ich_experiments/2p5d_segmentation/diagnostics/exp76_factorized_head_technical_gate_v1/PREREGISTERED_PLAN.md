# Exp76 — Factorized foreground/subtype architecture technical gate

## Hypothesis fixed before execution

Exp72–Exp75 showed that a hierarchical loss can reduce background pressure but
cannot consistently improve rare/diffuse subtype margins through the incumbent
six-class output representation. Exp76 changes the output architecture to the
exact factorization `P(foreground) × P(subtype | foreground)` and gives the two
factors separate zero-initialized spatial residual heads. The shared decoder and
legacy segmentation head remain trainable in the later screen so representation
can change; the encoder and auxiliary classification head remain frozen.

For legacy mask logits `z`, initialization is locked to:

- `foreground_logit = logsumexp(z[1:]) - z[0]`;
- `conditional_subtype_logits = z[1:]`;
- composed logits `[0, foreground_logit + log_softmax(subtype_logits)]`.

This must reproduce the legacy six-class softmax probabilities and hard argmax
while making the foreground objective invariant to subtype residuals and the
conditional subtype objective invariant to the foreground residual.

## Locked technical protocol

- Warm-start checkpoint: Exp61 fold 2, schema4 calibration-only incumbent.
- No optimizer, no patient images, no train/calibration/outer/test inference.
- Synthetic tensor seed `42`, batch size `2`, shape `9×384×384`, FP32.
- Aggregate JSON only; no row-level or medical prediction artifact.
- All existing ICH segmentation tests plus new identity, gradient-isolation,
  trainable-scope and legacy-checkpoint expansion tests must pass on the server.
- MLflow diagnostic logging and one analytical Persian Telegram completion event.

## Preregistered gates

- Maximum six-class probability difference at most `2e-6`.
- Hard-mask argmax mismatch fraction exactly `0`.
- Auxiliary classification-logit maximum difference exactly `0`.
- Foreground and subtype residual outputs exactly zero at initialization.
- Foreground→subtype and subtype→foreground cross-gradient maxima each at most
  `1e-6`.
- Encoder and classification trainable parameter counts exactly zero.
- Decoder/legacy segmentation head/new spatial heads expose a nonzero trainable
  parameter count and their train/eval modes match the locked scope.
- All aggregate values finite and all related tests pass.

Passing authorizes only a separately preregistered bounded pipeline smoke and
calibration-only screen. Failure rejects the implementation before any data or
optimizer access; gates may not be relaxed after observing the result.
