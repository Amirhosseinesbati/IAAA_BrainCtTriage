# Exp80 result: BF16-exact factorized ICH calibration screen

## Decision

`reject_before_outer_or_oof`

Exp80 was a preregistered three-epoch calibration-only screen of the factorized
foreground/subtype output head warm-started from Exp61. The best checkpoint
remained epoch 0, which is an exact functional copy of Exp61. No outer or OOF
evaluation was performed because the locked calibration promotion gates failed.
The checkpoint was not promoted to `checkpoint/ich`.

## Reproducibility

- MLflow run: `a36b6ba841384b938ee4f84ec3c335a9`
- Training commit: `ce99c7715fb38ef358b2d00cfe6cbb1f883f9aba`
- Warm-start checkpoint SHA-256: `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`
- Data manifest SHA-256: `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`
- Selected identity checkpoint SHA-256: `92283eaafc605e537b6e7cb087a3f3e17a787ae1f8862967df83ecdc86b08fa3`
- Final promotion gate SHA-256: `36bed6dc2173c4816d60b96bfe34b4e6fbb1418e5c44654b2ef4a585ee470cb2`
- Duration: `195.85 s`; peak GPU memory: `1.875 GiB`
- Trainable spatial parameters: `2,837,996`
- Scope: ICH only; no MLS, fracture, triage, or row-level medical predictions

## Calibration trajectory

| Epoch | Selection | Mean Dice | IVH | IPH | EDH | SAH | SDH | FPR | F1 | Volume MAE (mL) | Bias (mL) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.666162 | 0.459106 | 0.648020 | 0.674845 | 0.537976 | 0.053024 | 0.381665 | 0.194444 | 0.882353 | 10.7627 | -6.2364 |
| 1 | 0.641222 | 0.413759 | 0.667698 | 0.685564 | 0.452833 | 0.052008 | 0.210694 | 0.166667 | 0.895522 | 14.2565 | -11.3183 |
| 2 | 0.652811 | 0.434830 | 0.673214 | 0.694771 | 0.472173 | 0.040552 | 0.293443 | 0.194444 | 0.882353 | 12.9387 | -9.7636 |
| 3 | 0.657935 | 0.444147 | 0.670254 | 0.691983 | 0.520254 | 0.044327 | 0.293915 | 0.194444 | 0.882353 | 12.6082 | -9.4937 |

Epoch 0 exactly reproduced Exp61 for every locked aggregate. The full-precision
gate recomputation correctly marks all equality/non-inferiority checks as passed.
The experiment is rejected only because it produced no required positive gain:
checkpoint score, selection score, mean Dice, SAH Dice, and SDH Dice all failed
their preregistered improvement thresholds.

## Interpretation

The architecture and warm-start path are technically sound: the BF16 identity
problem discovered in Exp77 was fixed, and zero residual heads preserve Exp61
exactly. The failure is therefore attributed to the first factorized training
objective, not to initialization, mixed precision, checkpoint loading, or the
evaluation gate.

The conditional class-weighted focal plus conditional Dice update favored the
more frequent/compact IVH and IPH classes while sharply suppressing diffuse SDH
and leaving SAH near zero. It also increased negative total-volume bias. The
partial recovery by epoch 3 did not return SDH, SAH, volume MAE, or the composite
scores to baseline, so extending the same recipe is not justified.

## Next evidence-gathering step

Before another full calibration run, measure per-loss-component gradients and
short-update class drift on train-only batches using the same Exp61 checkpoint.
The diagnostic should isolate whether conditional Dice or class-weighted focal
causes the SDH/SAH suppression. The next promoted recipe should then either
remove/rebalance the harmful component, anchor non-target logits to Exp61, or
use a separate subtype decoder with an explicit trust-region constraint.
