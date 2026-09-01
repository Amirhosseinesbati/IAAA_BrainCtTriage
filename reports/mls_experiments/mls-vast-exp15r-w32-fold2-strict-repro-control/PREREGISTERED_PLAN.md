# Exp15r strict MLS reproducibility control

## Question

Can the Exp14r2 configuration produce a stable, auditable RTX-3060 training anchor when all known CUDA, epoch, sampler and DataLoader-worker randomness is explicitly controlled?

## Single changed factor

The normalized training configuration must equal Exp14r2 in every field except `training_determinism`, which changes from the historical default `benchmark` to `strict`. No warm start, checkpoint transfer, data rebuild, loss change, pooling retune, model averaging or ICH input is permitted.

Strict mode requires:

- cuDNN benchmark off and cuDNN deterministic on;
- deterministic PyTorch algorithms with errors, not warnings, for unsupported operations;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before model arithmetic;
- epoch-addressable Python, NumPy, PyTorch and CUDA RNG seeds;
- explicit Python/NumPy seeding in each DataLoader worker;
- CUDA-only model compute with no CPU fallback.

## Preconditions

1. The server commit contains the manifest and reproducibility implementation.
2. The final processed-data contract remains 3484 rows, 338 studies, 1781 positive rows and 1703 negative rows, with all paths resolved.
3. PyTorch sees the RTX 3060 only after the real host driver library (`/usr/lib/x86_64-linux-gnu`) is ahead of the stale toolkit compatibility library.
4. A strict-mode HRNet-W32 512-pixel forward/backward smoke test completes with finite loss.
5. MLflow secrets remain root-only and the remote run can be created.
6. No MLS tmux session or durable run lock with this run name already exists.

## Evaluation protocol

- Train all 23 epochs unless the existing non-metric safety gate stops the run.
- Preserve odd snapshots 13, 15, 17, 19, 21 and 23, plus best-objective and best-selector-AUC checkpoints.
- Primary post-training evaluation: GPU-only inference on all 67 fold-2 studies for each preregistered candidate, followed by the historical locked production pooling profile.
- Secondary diagnostic: same-fold pooling grid, labelled explicitly as optimistic and never used alone for promotion.
- Compare epoch histories against Exp14r2 and historical Exp10, while treating exact numerical equality to either historical benchmark-mode run as neither expected nor required.

## Decision rules

- Infrastructure success requires terminal completion, a nonempty MLflow run ID, strict deterministic flags in MLflow, no CPU fallback, 67/67 successful studies for every audited checkpoint, and no NaN/Inf.
- Production promotion requires a preregistered checkpoint under the locked profile to achieve MAE no worse than 1.75 mm and boundary F1 at least 0.88, and to remain scientifically plausible under the secondary diagnostics. The current trusted Exp10 epoch-15 candidate remains preferred otherwise.
- If the run is stable but not better, it becomes the controlled baseline for exactly one subsequent architecture/loss intervention; it is not promoted merely for being deterministic.
- Sigma annealing Exp15 remains blocked because the Exp14r2 reproduction gate failed.

The Vast instance must remain running after completion; no automatic stop or destroy action is authorized.
