# Exp84 — three-epoch factorized residual-head-only calibration screen

## Hypothesis fixed before execution

Exp82 attributed the fast held-out collapse to shared decoder/legacy-head updates,
and Exp83 showed that restricting the exact same objective to the 870 residual-head
parameters preserved calibration after four updates. Exp84 tests whether those
heads can produce a material SAH/SDH improvement over three full epochs without
damaging the incumbent representation.

## Locked execution

- Exp61 checkpoint SHA-256
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Schema4 manifest SHA-256
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.
- Patient-safe split: train folds exclude outer `2` and calibration `1`;
  calibration fold `1` is evaluated after each epoch; outer fold `2` is never
  inferred.
- Seed `42`, batch `16`, workers `4`, BF16, AdamW `5e-5`, weight decay `1e-4`,
  cosine schedule for exactly three epochs, gradient clipping `5.0`.
- Full Exp80 hierarchical foreground/subtype objective.
- Encoder, classifier, decoder and legacy six-class head remain frozen and in
  eval mode. Only the zero-initialized one-channel foreground residual and the
  centered five-channel subtype residual convolutions are trained: 870
  parameters total.
- Local-only aggregate JSON and optional experimental checkpoint. No row-level
  predictions, MLflow/DagsHub, Telegram, outer inference, OOF or promotion.
- A candidate checkpoint is written only when an epoch beats the exact Exp61
  initialization on the locked FPR/volume-penalized checkpoint score.

## Locked gate for replication

The best calibration epoch must satisfy every condition:

- checkpoint score and selection score gains at least `0.003`;
- mean foreground Dice gain at least `0.005`;
- SAH Dice gain at least `0.010` and SDH Dice gain at least `0.005`;
- IVH, IPH and EDH Dice may each fall by at most `0.005`;
- normal FPR, presence F1, total-volume MAE and absolute total-volume bias are
  noninferior;
- Any-ICH and macro-subtype study AUC remain exactly unchanged, as required by
  the frozen classifier;
- every required aggregate is finite.

Passing authorizes a deterministic/multiseed replication on calibration before
any outer evaluation. Failure rejects outer evaluation and checkpoint promotion;
a small score-only gain may be retained locally for diagnosis but is not an
accepted model.
