# Exp84 result — residual-head-only full calibration screen rejected

## Decision

`reject_residual_head_only_three_epoch_screen`

No epoch exceeded the exact Exp61 initialization on the preregistered
FPR/volume-penalized checkpoint score. Consequently no candidate checkpoint was
written, no model was promoted and outer fold 2 was not evaluated.

## Locked provenance and execution

- Initial checkpoint SHA-256:
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Schema4 manifest SHA-256:
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.
- Seed 42, BF16, batch 16, 303 updates per epoch, AdamW `5e-5`, cosine
  schedule, three epochs.
- Only 870 residual-head parameters were trainable. Encoder, decoder,
  classifier and legacy segmentation head remained frozen/eval.
- 4,848 train slices; 1,346 calibration slices. The 1,466 outer slices were
  enumerated by split construction but never inferred.
- Runtime 146.58 seconds; peak allocated VRAM 0.94 GiB.
- External reporting disabled; only aggregate metrics were persisted.

## Calibration trajectory

| Epoch | checkpoint score | selection | mean Dice | IVH | IPH | SDH | EDH | SAH | volume MAE (mL) | bias (mL) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.586668 | 0.666162 | 0.459106 | 0.648020 | 0.674845 | 0.381665 | 0.537976 | 0.053024 | 10.7627 | -6.2364 |
| 1 | 0.581003 | 0.663438 | 0.454153 | 0.650407 | 0.678299 | 0.349073 | 0.532163 | 0.060823 | 11.1012 | -7.4848 |
| 2 | 0.578276 | 0.661989 | 0.451518 | 0.650905 | 0.678553 | 0.336335 | 0.529418 | 0.062380 | 11.2982 | -7.7779 |
| 3 | 0.577582 | 0.661642 | 0.450887 | 0.650851 | 0.678466 | 0.333411 | 0.529257 | 0.062452 | 11.3609 | -7.8111 |

FPR (`0.194444`), presence F1 (`0.882353`), Any-ICH AUC (`0.923387`) and
macro-subtype AUC (`0.910920`) stayed unchanged, as expected from the frozen
classifier and stable study-level presence decisions.

At epoch 3 versus baseline, SAH Dice improved by `+0.009428`, IPH by
`+0.003621` and IVH by `+0.002831`, but SDH fell by `-0.048254`, EDH by
`-0.008719`, mean Dice by `-0.008219`, checkpoint score by `-0.009086`, and
volume MAE worsened by `+0.5982 mL`. The score and SDH degradation were monotonic
across all three epochs.

## Interpretation

Exp83 correctly established *short-horizon safety*: freezing the shared
representation prevents the four-update collapse. Exp84 establishes that this
is not sufficient for useful full-epoch learning. The conditional subtype head
systematically reallocates probability toward scarce SAH at the expense of SDH
and EDH, while the foreground residual increasingly underestimates total
volume. This is a stable trade-off, not transient optimizer noise.

The next experiment must locate the onset and separate the two residual heads
before any new architecture or outer evaluation. A preregistered early-update
trajectory followed by foreground-only versus subtype-only parameter-scope
attribution is justified; another unrestricted three-epoch run is not.

## Artifact integrity

- `history.json` SHA-256:
  `528b0a56057f044863031a63e4f537db72ed23c3511b9b874c96b7bfe9aae17e`.
- `run_summary.json` SHA-256:
  `e387960bb4dd2b3503937513cefdded8ee6e721e2394ee686b9c5349dd134493`.
