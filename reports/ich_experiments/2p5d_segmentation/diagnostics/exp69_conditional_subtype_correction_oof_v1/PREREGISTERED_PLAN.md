# Exp69 preregistration: correction-only conditional subtype OOF probe

Status: recipe locked before Exp69 outer inference

Scope: ICH only; no MLS, fracture, triage fusion, leaderboard, or test data

Evaluation role: five-fold patient-disjoint **development OOF**, not a final test

## Motivation

Exp68 proved that a copied decoder/head can relabel hemorrhage subtype while
preserving the incumbent foreground/background support and classification
logits exactly.  It recovered 74.14% of train-set SAH→IPH errors, but its
class-weighted CE was applied to every supported true-foreground pixel.  That
objective traded correct IPH pixels for IVH/SAH and reduced conditional
accuracy by 0.204 percentage points.  The train baseline was also saturated:
its 99.527% macro recall made the preregistered +1 percentage-point gate
mathematically unreachable.

Exp69 therefore tests a different, correction-preserving hypothesis on
patient-disjoint outer folds: ground-truth CE is applied only where the frozen
incumbent already predicts foreground but chooses the wrong foreground
subtype.  All other incumbent-foreground pixels are anchored to the incumbent
soft distribution with KL divergence.  KL is exactly zero at initialization,
so batches without correction targets cannot sharpen or drift the incumbent.

## Provenance and interpretation boundary

The five source checkpoints were selected without evaluating their own outer
fold (`evaluate_outer=false`) and use the shared patient-level fold manifest.
The project has previously inspected these outer folds through other models,
so Exp69 must not be described as a never-seen final test.  Its recipe is fixed
before this suite's inference and the competition leaderboard remains the only
external final validation.

| Outer | Calibration | Source experiment | Encoder |
|---:|---:|---|---|
| 0 | 1 | exp30 | efficientnet-b2 |
| 1 | 0 | exp32 | efficientnet-b2 |
| 2 | 1 | exp26 | efficientnet-b2 |
| 3 | 1 | exp34 | efficientnet-b2 |
| 4 | 1 | exp36 | efficientnet-b2 |

The official split constructor must report zero patient intersection among
train, calibration, and outer for every fold.  Aggregate-only metrics are
persisted; no patient IDs or row-level predictions are copied off the server.

## Locked recipe

- Architecture: Unet++ with EfficientNet-B2, initialized independently from
  the matching fold checkpoint.
- Incumbent encoder, decoder, segmentation head, classification head, hard
  foreground support, and classification logits: frozen.
- Trainable scope: copied decoder and copied segmentation head only.
- BatchNorm running statistics in the copy: frozen; affine parameters remain
  trainable.
- Training data: only the three train folds belonging to that checkpoint.
- Evaluation data: only its patient-disjoint outer fold; calibration is not
  used to tune Exp69.
- Epochs: 1.
- Optimizer: AdamW, learning rate `5e-5`, weight decay `1e-4`.
- Correction loss: CE only on spatially known true-foreground pixels that are
  inside incumbent foreground and have the wrong incumbent subtype.
- Correction class weights: square-root inverse pixel frequency, clipped to
  `[1, 4]` and computed independently from that fold's training rows.
- Preservation: soft KL to incumbent on every other incumbent-foreground
  pixel, weight `1.0`.
- Seed: 42; batch size 16; BF16; conditional support margin 1.0.
- Diagnostic refiner weights may be retained on the server/MLflow for
  reproducibility but are marked `not_promoted=true` and are not copied into
  `checkpoint/ich` unless a later full-metric gate passes.

## Locked gates

All gates must pass simultaneously:

1. Five unique outer folds 0..4 are present.
2. Initial hard-mask identity is exact on every outer fold.
3. Final foreground support mismatch is zero on every outer fold.
4. At least 100 aggregate incumbent SAH→IPH error pixels exist.
5. At least 10% of aggregate SAH→IPH errors are recovered.
6. Aggregate harm to correct IPH is at most 1%.
7. Aggregate harm to correct IVH/SDH/EDH is at most 1%.
8. Aggregate subtype changes on true-background incumbent foreground are at
   most 2%.
9. Aggregate conditional accuracy improves by at least 0.1 percentage points.
10. Aggregate conditional macro recall improves by at least 0.2 percentage
    points.
11. Conditional accuracy is non-negative on at least three of five folds.
12. Conditional macro recall is non-negative on at least three of five folds.
13. Worst-fold conditional accuracy delta is no lower than -0.5 percentage
    points.
14. Worst-fold conditional macro-recall delta is no lower than -1 percentage
    point.

Passing authorizes one locked full-metric OOF screen measuring subtype Dice,
study AUC, presence, physical-volume MAE/bias, and FPR with the existing fold
rules.  It does not authorize promotion or leaderboard claims.  Failure rejects
this recipe before full-metric OOF; thresholds must not be relaxed after seeing
Exp69 results.
