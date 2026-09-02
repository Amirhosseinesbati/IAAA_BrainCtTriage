# Exp77 — Factorized hierarchy decoder fine-tune, warm Exp61, schema4 fold 2

## Hypothesis fixed before optimizer or calibration execution

Exp76 proved that the factorized output wrapper preserves the real Exp61
probabilities and hard mask at initialization while algebraically isolating the
foreground and conditional-subtype residual branches. Exp77 tests whether
allowing the shared decoder, legacy segmentation head and two spatial residual
heads (`2,837,996` parameters) to learn can improve rare/diffuse subtype
representation without moving the frozen encoder or auxiliary classification
head.

This is the first optimizer run of this architecture. It is not a sweep. The
single recipe below is fixed from Exp61 and the train-only Exp72–Exp75 gradient
evidence before calibration results are observed.

## Locked data and initialization

- Manifest: `Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv`.
- Patient-safe split: outer fold `2`, calibration fold `1`, seed `42`.
- Warm start: Exp61 `best.pth`, SHA-256
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Architecture/encoder: `unetplusplus/tu-efficientnetv2_rw_s`, 9-channel 2.5D,
  resolution `384`, batch `16`, workers `4`, BF16.
- Encoder and auxiliary classification head frozen; decoder, legacy mask head,
  foreground residual head and subtype residual head trainable. Decoder BatchNorm
  running statistics remain frozen while affine parameters may train.
- No MLS, fracture, triage, external data or row-level MLflow artifacts.
- Outer fold remains completely unevaluated in smoke and full calibration screen.

## Locked objective and optimizer

- Hierarchical foreground/subtype objective through the factorized logits.
- Foreground Dice `0.325`; foreground focal `0.175`.
- Conditional subtype Dice `0.325`.
- Conditional class-weighted focal CE `0.175`, gamma `2.0`, pixel-prior weights
  with power `1.0` and cap `8.0`; Balanced Softmax and OVR disabled.
- Probability-weighted foreground derivative; factorization must make its
  subtype-residual cross-gradient zero by construction.
- Classification loss weight `0` because that branch is deliberately frozen.
- Existing hard-empty foreground penalty `0.05` on top fraction `0.001` retained
  for FPR control; checkpoint selection remains `fpr_volume_penalized`.
- AdamW learning rate `5e-5`, weight decay `1e-4`, cosine schedule, three epochs,
  patience three.

## Phase A — bounded pipeline smoke

- Separate output directory, one partial epoch capped at four optimizer steps.
- Required: all losses/metrics finite, no OOM, no outer evaluation, checkpoint and
  aggregate MLflow artifacts written, peak VRAM below 20 GiB.
- Epoch-zero calibration identity must match the Exp61 reference within:
  selection/Dice/AUC/MAE/bias absolute difference `1e-6`, FPR/F1 exactly equal.
- Smoke performance after four steps is not a promotion signal and may not be
  used to change the locked full-run recipe.

Only technical completion and epoch-zero identity authorize Phase B.

## Phase B — three-epoch calibration-only screen

The best checkpoint is selected only by the existing
`fpr_volume_penalized` score. Promotion beyond calibration requires every gate:

- checkpoint score at least `0.589668` (Exp61 `0.586668` + `0.003`);
- selection score at least `0.669162` (Exp61 `0.666162` + `0.003`);
- mean foreground Dice at least `0.464106` (Exp61 `0.459106` + `0.005`);
- normal FPR at 0.1 ml no greater than `0.194444`;
- presence F1 no lower than `0.882353`;
- total-volume MAE no greater than `10.762716` and absolute bias no greater than
  `6.236357` ml;
- Any-ICH AUC no lower than `0.923386` and macro subtype AUC no lower than
  `0.910919` (these should be invariant because encoder/classifier are frozen);
- SAH Dice at least `0.063024` and SDH Dice at least `0.386665`;
- IVH, IPH and EDH Dice each no more than `0.005` below their Exp61 values;
- all aggregate values finite, no outer evaluation and manifest/checkpoint/code
  provenance preserved in MLflow and checkpoint metadata.

Failure rejects this exact recipe before outer/OOF. Passing authorizes a new,
separately preregistered same-family fold replication; gates are not relaxed
after results are observed.
