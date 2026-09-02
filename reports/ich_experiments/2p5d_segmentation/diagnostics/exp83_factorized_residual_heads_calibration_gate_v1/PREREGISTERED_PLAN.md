# Exp83 — residual-head-only factorized calibration gate

## Hypothesis fixed before execution

Exp82 showed that four foreground-only updates still damaged SDH/EDH and volume
when the shared decoder and legacy segmentation head were trainable. Exp83 tests
whether freezing the complete legacy representation and training only the two
algebraically separated zero-residual heads removes that early held-out collapse.

## Locked execution

- Exp61 checkpoint and Schema4 manifest hashes are unchanged.
- Outer fold `2` remains untouched; only calibration fold `1` is evaluated.
- Exact same first four seed-42 sampler batches, batch 16, BF16, AdamW `5e-5`,
  weight decay `1e-4`, and full Exp80 objective.
- Encoder, classifier, decoder and legacy six-class segmentation head are
  frozen/eval. Only the one-channel foreground residual and five-channel
  centered subtype residual convolutions are trainable.
- Local-only aggregate JSON; no DagsHub/MLflow, Telegram, row-level artifact,
  checkpoint/model write, outer inference, OOF, or promotion.

## Locked safety gate

Compared with exact Exp61 initialization after four updates:

- mean foreground Dice drop at most `0.005`;
- SDH, EDH and SAH Dice drop individually at most `0.005`;
- normal FPR and presence F1 noninferior;
- volume MAE increase at most `0.5 mL`;
- absolute total-volume bias increase at most `0.5 mL`;
- all required aggregates finite and exact provenance preserved.

All gates are conjunctive. Passing authorizes implementation and preregistration
of a three-epoch residual-head-only calibration screen; it does not authorize
outer evaluation or checkpoint promotion.
