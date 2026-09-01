# Exp09 fold1 CUDA end-to-end checkpoint audit

## Audit contract

- Six checkpoints were declared before full-study evaluation: epochs 13, 15, 17, 19, 21 and 23.
- All model inference ran on CUDA. CPU use was limited to file I/O and lightweight aggregation of saved predictions.
- Fold 1 contains 67 studies. All `402/402` study-checkpoint evaluations completed with zero failures, no OOM and no CPU model fallback.
- Total recorded inference time was about 549 seconds, or about 91.5 seconds per checkpoint.
- Per-study/slice prediction CSV files remain local. Only aggregate reports and summaries may be uploaded to MLflow.

## Internal validation did not predict the E2E winner

Internal validation selected epoch 17 for the balanced objective (`1.375`), epoch 21 for study MAE (`0.941 mm`) and epoch 13 for Boundary-F1 (`0.825`). Full-study common-grid evaluation instead selected epoch 15 as the best balanced single checkpoint. This repeats the proxy/E2E mismatch observed on folds 0 and 2 and confirms that final checkpoint selection cannot rely on the training validation proxy.

## Common-grid single-checkpoint results

Every checkpoint was evaluated on the same 6048 pooling profiles. The rows below are in-fold diagnostics and are not production-locked profiles.

| Checkpoint | Best balanced MAE | Boundary-F1 | Objective |
|---|---:|---:|---:|
| epoch 13 | 1.3694 | 0.8047 | 1.7601 |
| epoch 15 | 1.2722 | 0.8635 | **1.5453** |
| epoch 17 | 1.2569 | 0.8158 | 1.6253 |
| epoch 19 | **1.1877** | 0.7983 | 1.5912 |
| epoch 21 | 1.3574 | 0.7896 | 1.7783 |
| epoch 23 | 1.3137 | 0.7982 | 1.7172 |

Epoch 19 has the lowest diagnostic MAE, but epoch 15 is materially safer at the 3 mm and 5 mm decision boundaries. Epoch 15 is therefore the primary single-checkpoint candidate; epoch 19 is a complementary low-MAE member, not a standalone winner.

## Fair comparison with the previous fold1 peak-aware baseline

On the identical common grid, the previous Exp06 best-objective checkpoint achieved `MAE=1.3301`, `Boundary-F1=0.8035` and `objective=1.7231`. Exp09 epoch15 achieved `MAE=1.2722`, `Boundary-F1=0.8635` and `objective=1.5453`: approximately 4.35% lower MAE, +0.060 Boundary-F1 and 10.3% lower objective.

The stronger transfer test uses a profile selected on fold0 and frozen before this fold was evaluated:

| Candidate | Frozen MAE | Frozen Boundary-F1 | Frozen objective |
|---|---:|---:|---:|
| Exp06 best-objective | 1.6206 | 0.7340 | 2.1526 |
| Exp09 epoch15 | 1.4812 | 0.7902 | 1.9009 |
| 75% Exp06 + 25% Exp09 epoch21 | **1.4096** | 0.7859 | **1.8379** |

The hybrid single checkpoint improves frozen MAE by 8.60% and Boundary-F1 by 0.0562 over Exp06. The conservative blend improves frozen MAE further while retaining a large boundary gain. This is evidence of transfer, not merely an in-fold pooling search benefit.

## Targeted blend audit

Nine blend candidates were declared before their metrics were measured. No model inference was repeated.

- The fold0 policy, 75% baseline + 25% hybrid epoch15, reached `MAE=1.2239`, `Boundary-F1=0.8360`, `objective=1.5519` on its best common-grid profile. It improved MAE but was marginally worse in balanced objective than the single epoch15 result.
- A 25% baseline + 75% epoch15 blend reached `MAE=1.1916`, `Boundary-F1=0.8265`, `objective=1.5386`.
- The median of epochs 15, 17 and 19 produced the best diagnostic objective: `MAE=1.1195`, `Boundary-F1=0.8175`, `objective=1.4846`.
- A boundary-favoring profile of the same median ensemble reached `MAE=1.2083` and `Boundary-F1=0.8512`.

The three-snapshot median is promising but remains an in-fold diagnostic and triples inference cost. It cannot replace the simpler epoch15 model until fold2 and strict cross-fold transfer confirm the gain.

## Decision

1. Do not spend more GPU time extending fold1 beyond epoch 23. Late training did not produce a consistently better E2E checkpoint.
2. Carry the unchanged hybrid-soft architecture, target, seed policy and 23-epoch snapshot schedule to fold2.
3. Treat epoch15 as the primary fold1 single model and epochs 17/19/21 as complementary ensemble candidates.
4. Recompute strict cross-fold/leave-one-fold-out profile and blend selection after fold2. No in-fold optimum is production-locked.
5. Upload only this report and aggregate JSON summaries to MLflow; retain raw prediction CSV files locally.
