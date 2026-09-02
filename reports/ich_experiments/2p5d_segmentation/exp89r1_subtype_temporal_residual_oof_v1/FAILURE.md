# Exp89r1 partial result and cache-provenance stop

The evaluator-only retry completed outer folds 0 and 1 and selected the fold2
temporal checkpoint, then stopped before fold2 outer inference because an older
`outer_f2c1_c63e609c652c` feature cache had metadata that did not match the current
strict cache contract. The validator refused reuse as designed. No model,
hyperparameter or promotion gate failed, and the old cache was not deleted or
silently replaced.

Fold0 selected epoch 9 after calibration macro subtype AUC delta `+0.044283`, but
its outer macro delta was `-0.003805`. Fold0 outer SDH improved `+0.040073`, while
SAH declined `-0.046875`, IPH `-0.004371`, IVH `-0.004049`, and EDH AUC was
undefined because the fold contained one class. Fold1 selected the exact identity
checkpoint at epoch 0 because no trained epoch improved calibration; consequently
all fold1 outer deltas were exactly zero. Patient overlap was zero in both folds.

Fold2 reached a best calibration macro delta of `+0.015359` at epoch 15, but no
fold2 outer result was produced. The next retry must use a fresh cache namespace,
preserve every locked scientific choice, and retain both failed directories.
Aggregate partial results are stored in `partial_fold_summaries.json`; no row-level
predictions were copied locally.
