# Exp09 fold1 full-study checkpoint audit

- Status: inference complete; 402/402 study-checkpoint evaluations succeeded with zero failures. Common-grid pooling analysis is next.
- Compute policy: model loading and every forward pass are CUDA-only; no CPU inference fallback is permitted.
- Fold: 1 (67 studies, verified from the competition fold manifest at runtime; the earlier 70-study expectation belonged to fold0).
- Batch size: 6.
- Checkpoints were pre-registered before seeing fold1 full-study results: epochs 13, 15, 17, 19, 21 and 23.
- Each checkpoint writes an atomic resumable local `study_slice_predictions.csv` plus aggregate `metrics.json`.
- Raw per-study/slice predictions remain local and must not be uploaded to MLflow.
- After all six candidates finish, `search_mls_checkpoint_pooling.py` will evaluate the same common aggregation grid and produce an aggregate summary suitable for MLflow.

Selection safeguards:

1. Internal validation metrics are diagnostic only; they do not eliminate a pre-registered checkpoint.
2. In-fold best pooling profiles are diagnostic and cannot be treated as production-locked.
3. Frozen transfer profiles and comparison with the fold1 peak-aware baseline determine whether hybrid transfer is real.
4. A fold2 training decision is deferred until this audit and the fold1 blend comparison are complete.

Pre-registered targeted blend shortlist (declared before measuring blend metrics):

- Exp06 best-objective 75% + Exp09 epoch15 25%: direct transfer of the conservative fold0 blend policy.
- Exp06 best-objective 50% + Exp09 epoch15 50%, and 25% + 75%: bounded weight sensitivity controls.
- Exp06 best-objective 75% + Exp09 epochs 17, 19 or 21 at 25%: checkpoint-role controls for internal-objective, common-grid-MAE and internal-MAE winners.
- Exp09 epoch15 + epoch17 mean: two balanced neighboring snapshots.
- Exp09 epochs 15 + 17 + 19 mean and median: a small robustness control that includes the common-grid MAE winner.

No epoch13/23 blends are added after seeing their weaker primary audit results. The shortlist is intentionally small and uses no new model inference.
