# Exp80 — BF16-exact factorized hierarchy, three-epoch calibration screen

## Authorization and hypothesis fixed before execution

Exp78 proved exact BF16 logit/probability/argmax identity and Exp79 passed every
pipeline smoke gate. Exp80 executes the three-epoch recipe that was locked before
Exp77. It tests whether a trainable decoder and independent foreground/subtype
output factors can improve diffuse subtype morphology without losing Exp61's
FPR, volume or frozen auxiliary-classification behavior.

## Locked recipe

- Exp61 warm start SHA-256
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Schema4 manifest SHA-256
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.
- Patient-safe outer fold `2`, calibration fold `1`, seed `42`; outer inference
  disabled.
- `unetplusplus/tu-efficientnetv2_rw_s`, 9-channel 2.5D, 384 resolution, batch
  `16`, workers `4`, CUDA BF16.
- Factorized output; encoder/classifier frozen; decoder, legacy mask head and two
  residual heads trainable (`2,837,996` parameters), decoder BN statistics fixed.
- Foreground Dice `0.325`, foreground focal `0.175`, conditional subtype Dice
  `0.325`, conditional class-weighted focal CE `0.175` with gamma `2`; OVR and
  auxiliary classification loss zero.
- Pixel-prior class weighting power `1`, cap `8`; hard-empty penalty `0.05` on
  top `0.001`; no hard-negative resampling.
- AdamW learning rate `5e-5`, weight decay `1e-4`, cosine schedule, three epochs,
  patience three, `fpr_volume_penalized` checkpoint selection.
- MLflow logs only safe aggregate/model artifacts; analytical Persian Telegram
  on start, completion and final gate. No MLS, fracture or triage work.

## Locked promotion gates versus Exp61

- checkpoint score at least `0.5896680239` (`+0.003`);
- selection score at least `0.6691624032` (`+0.003`);
- mean foreground Dice at least `0.4641058994` (`+0.005`);
- normal FPR at 0.1 ml no greater than `0.1944444444`;
- presence F1 no lower than `0.8823529412`;
- total-volume MAE no greater than `10.7627157621` ml and absolute bias no
  greater than `6.2363560641` ml;
- Any-ICH AUC at least `0.9233860968` and macro subtype AUC at least
  `0.9109191965`;
- SAH Dice at least `0.0630242235` and SDH Dice at least `0.3866645469`;
- IVH, IPH and EDH Dice no more than `0.005` below their Exp61 values;
- required aggregates finite, exact locked config/provenance, checkpoint exists,
  and outer evaluation remains absent.

All gates are conjunctive. Failure rejects this recipe before outer/OOF and the
existing accepted checkpoint remains unchanged. Passing authorizes a separately
preregistered same-family replication rather than immediate outer consumption.
