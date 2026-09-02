# Exp20 post-failure named-best checkpoint diagnostic plan

## Status boundary

The epoch21 third-fold primary test failed and remains failed. This diagnostic
cannot replace or rescue it.

## Fixed secondary hypothesis

All dual-selector trainers already save `mls_multitask_best.pth` when the online
study selection objective reaches a new minimum. This pre-existing selection
rule, applied before full-study audit, maps to:

- Exp18: epoch12;
- Exp19: epoch21;
- Exp20: epoch11.

The Exp20 named checkpoint exists with SHA-256
`115809f572d69661c95bebda36e3f382a3a6d00c04b0ae4a18174d0f58b48184`.

Exactly one additional CUDA audit is allowed:

1. Candidate: Exp20 `mls_multitask_best.pth` only.
2. Fold: 2; expected studies: 67; zero failures required.
3. Baseline: Exp15r epoch17 saved CUDA predictions.
4. Transfer recipe: `regression_only`, alpha 0.10, unchanged production profile.
5. Diagnostic comparison uses the same hard gates: MAE <=`1.5483543317709396`,
   Boundary-F1 >=`0.8925925925925926`, objective <=`1.7531691465857544`.

If this candidate fails, checkpoint-timing diagnostics stop; no snapshot or
named metric checkpoint may substitute. If it passes, the common named-best
selection rule becomes a package candidate, but because this hypothesis was
formalized after the fold2 primary failure it still requires unused-fold or
leaderboard validation before any release claim.

The audit is CUDA-only. Transfer evaluation uses saved predictions and light
aggregate post-processing. Raw prediction CSVs remain excluded from MLflow and
local synchronization.
