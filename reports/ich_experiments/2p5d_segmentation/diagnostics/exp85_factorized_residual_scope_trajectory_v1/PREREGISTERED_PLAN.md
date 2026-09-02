# Exp85 — factorized residual early trajectory and scope attribution

## Hypotheses fixed before execution

Exp84 produced a monotonic SAH-versus-SDH/volume trade-off over three epochs.
Exp85 asks two narrower causal questions before changing the architecture:

1. Does either the joint model or an isolated residual head briefly beat Exp61
   before the full first-epoch collapse?
2. Is the growing volume underestimation caused by the foreground residual head,
   while the SDH-to-SAH reallocation is caused by the subtype residual head?

## Locked execution

- Exact Exp61 checkpoint and Schema4 manifest used by Exp83/84; outer `2` is
  never inferred and calibration `1` is the only held-out evaluation fold.
- Seed 42, BF16, batch 16, AdamW `5e-5`, weight decay `1e-4`, no scheduler,
  gradient clipping `5.0`, exactly one 303-update train-loader traversal.
- Three fresh zero-residual scopes see the same deterministic sampler sequence:
  both heads (870 parameters), foreground only (145), and subtype only (725).
- Full Exp80 objective remains fixed. Algebraic factorization ensures losses
  irrelevant to a frozen head cannot update that head.
- Full calibration is evaluated after updates `4, 16, 32, 64, 128, 192, 303`.
- The joint scope at step 4 must reproduce Exp83 and at step 303 must reproduce
  Exp84 epoch 1 within maximum absolute aggregate difference `1e-6`; recorded
  batch-identity hashes must match across scopes. Otherwise interpretation stops.
- Diagnostic-only aggregate JSON; no model/checkpoint, row-level predictions,
  external logging, Telegram, OOF, promotion or outer evaluation.

## Locked decision

- Any milestone passing the unchanged Exp84 conjunctive gate authorizes a
  separately preregistered replication, not outer inference.
- A checkpoint-score-only improvement without the full gate is recorded as a
  hypothesis signal but cannot be promoted.
- If no scope/milestone beats baseline, the current two-head factorization is
  closed and the next design must impose class-selective safety rather than tune
  epochs or learning rate blindly.
