# Exp10 fold2 full-study checkpoint audit

- Status: complete; training finished 23/23, MLflow run is `FINISHED`, and all six CUDA-only full-study audits plus the common 6048-profile grid completed successfully.
- Compute policy: model loading and every forward pass are CUDA-only; no CPU inference fallback is permitted.
- Fold: 2 (67 studies, verified from the immutable competition fold manifest).
- Batch size: 6.
- Checkpoints were pre-registered before full-study evaluation: epochs 13, 15, 17, 19, 21 and 23.
- Each checkpoint writes an atomic resumable local `study_slice_predictions.csv` plus aggregate `metrics.json`.
- Raw per-study/slice prediction CSV files remain local and must not be uploaded to MLflow.
- `search_mls_checkpoint_pooling.py` evaluated the identical common aggregation grid used for folds 0 and 1. The fold2 in-fold winner was epoch17, but the final decision was deferred to strict cross-fold selection as pre-registered.

Selection safeguards:

1. Internal validation metrics are diagnostic only and do not eliminate any pre-registered checkpoint.
2. In-fold best pooling profiles are diagnostic and cannot be treated as production-locked.
3. Final architecture/profile decisions use strict cross-fold/leave-one-fold-out transfer across folds 0, 1 and 2.
4. Snapshot ensembles are considered only if their gain transfers across folds and justifies extra inference cost.
5. The audit explains the fold2 behavior: epoch15 is the most transferable aligned checkpoint, while unrestricted snapshot selection is unstable. Additional blind epochs are not authorized; subsequent experiments must target selection/calibration variance or demonstrate a low-cost ensemble benefit.
