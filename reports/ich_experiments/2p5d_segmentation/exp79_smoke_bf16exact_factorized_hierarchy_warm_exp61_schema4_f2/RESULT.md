# Exp79 result — exact pipeline identity gate passed

Exp79 repeated the locked Exp77 four-step smoke after the Exp78 numerical fix.
It completed on commit `791900e` in `32.78s`, used `1.872GiB` peak VRAM and
registered MLflow run `c8ccac3a9f7a479c89ab63af64e73d3a`. All 91 related
tests passed and the outer fold was not evaluated.

The dedicated epoch-zero JSON matched Exp61 exactly: absolute differences for
selection, mean Dice, Any AUC, macro subtype AUC, total-volume MAE, bias, FPR and
F1 were all `0`. Every required loss/metric was finite, the partial epoch and
checkpoint completed, and all eight smoke gates passed.

The four updates again increased SAH Dice (`0.0530→0.0746`) but reduced SDH
(`0.3817→0.2693`) and mean Dice (`0.4591→0.4362`); epoch zero correctly remained
the best checkpoint. As preregistered, these partial-step values do not change
the full-run recipe.

Decision: `authorize_locked_three_epoch_calibration_screen`.

Smoke-gate artifact SHA-256:
`32b015bbdb145b8daba932e93b560c44d8f33c52fcefd10af03e1a7de1919b3c`.

Run-summary SHA-256:
`10f7d600fb44f96dab2157f2ffad818329cf660f9435fa60396dd72a4a9af775`.
