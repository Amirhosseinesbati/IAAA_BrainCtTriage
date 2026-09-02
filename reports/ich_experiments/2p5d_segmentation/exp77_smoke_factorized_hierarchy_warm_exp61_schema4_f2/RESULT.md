# Exp77 Phase A result — pipeline healthy, epoch-zero identity gate failed

The four-step smoke completed on commit `612f44c` in `30.76s`, used only
`1.805GiB` peak VRAM, wrote its checkpoint/aggregate artifacts to MLflow run
`4701f46735504b6da6ce76045b707456`, and did not evaluate the outer fold. All
training and calibration values were finite.

The epoch-zero factorized model retained exact Any/subtype AUC, FPR and F1, but
did not satisfy the preregistered `1e-6` identity tolerance for all continuous
spatial/volume metrics. Compared with Exp61, mean Dice differed by approximately
`4.52e-5`, selection by `2.48e-5`, total-volume MAE by `2.49e-3 ml`, and bias by
`1.05e-2 ml`. The cause is BF16 evaluation of log-sum-exp factorization at a few
near-tied pixels: Exp76 proved FP32 probability equivalence, but the real training
pipeline performs the model forward under BF16 autocast.

After four optimizer steps SAH Dice moved from `0.0530` to `0.0745`, while SDH
fell from `0.3819` to `0.2695` and mean Dice to `0.4362`. The best checkpoint
therefore correctly remained epoch zero. Per preregistration, this partial-step
performance is not used to retune the locked Phase B recipe.

Decision: `reject_before_full_calibration_or_outer`. The `1e-6` gate is not
relaxed. A new technical revision must compose factorized residuals in FP32 while
returning the original BF16 legacy logits exactly when residuals are zero, then
repeat the same bounded smoke before any full run.
