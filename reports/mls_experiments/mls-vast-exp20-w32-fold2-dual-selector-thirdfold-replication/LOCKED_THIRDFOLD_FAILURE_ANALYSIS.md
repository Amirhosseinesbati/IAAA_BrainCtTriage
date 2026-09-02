# Exp20 locked third-fold failure analysis

## Immutable primary result

Exp20 completed 23/23 CUDA-only epochs with exit code zero. The frozen primary
audit evaluated only epoch21 on all 67 fold2 studies and completed with zero
failures on the RTX3060. The fixed deployment recipe was 90% Exp15r epoch17 and
10% Exp20 epoch21 `mls_mm`, with all selector, peak/ranking and heatmap outputs
retained from Exp15r.

| metric | Exp15r baseline | fixed hybrid | delta | gate |
|---|---:|---:|---:|---|
| MAE (mm) | 1.548354332 | **1.533389346** | **-0.014964986** | pass |
| RMSE (mm) | 2.545112403 | 2.551152994 | +0.006040591 | diagnostic |
| bias (mm) | -0.334721173 | -0.397352078 | -0.062630905 | diagnostic |
| F1@3mm | 0.851851852 | 0.836363636 | -0.015488215 | diagnostic |
| F1@5mm | 0.933333333 | 0.933333333 | 0 | diagnostic |
| Boundary-F1 | 0.892592593 | **0.884848485** | **-0.007744108** | fail |
| objective | 1.763169147 | **1.763692376** | **+0.000523229** | fail; required <=1.753169147 |

The primary decision is therefore a valid scientific failure, not an
operational failure. It cannot be rescued by changing alpha, checkpoint or
pooling after observing fold2. Exp20 epoch21 is not copied into the local model
release directory.

## Mechanism of failure

The challenger provides a consistent MAE benefit but shifts predictions more
negative. F1@5mm is unchanged, while F1@3mm drops by one discrete confusion
step. This boundary discontinuity dominates the objective despite the improved
MAE. Exp20 standalone is not a replacement either: its locked candidate profile
has MAE `2.159952872mm`.

The preregistered post-failure alpha screen used saved CUDA predictions only and
tested 0.01, 0.025, 0.05, 0.075 and 0.10. No candidate passed all gates:

| alpha | MAE | Boundary-F1 | objective | all gates |
|---:|---:|---:|---:|---|
| 0.010 | 1.547982458 | 0.892592593 | 1.762797273 | no |
| 0.025 | 1.547158882 | 0.892592593 | 1.761973697 | no |
| 0.050 | 1.545385770 | 0.884848485 | 1.775688801 | no |
| 0.075 | 1.535993255 | 0.884848485 | 1.766296286 | no |
| 0.100 | 1.533389346 | 0.884848485 | 1.763692376 | no |

Weights up to 0.025 preserve the boundary but improve objective by at most
`0.001195449`, far below the required 0.01. At alpha 0.05 the 3mm boundary
changes discontinuously. Reducing alpha alone is therefore not a worthwhile
three-fold deployment solution.

## What remains supported

- Fold1 Exp09/Exp18 regression complement: passed its frozen numerical gates.
- Fold0 Exp16/Exp19 regression complement: passed its independent frozen gate.
- Fold2 Exp15r/Exp20 fixed epoch21 complement: failed.

The general fixed-epoch21 90/10 recipe is rejected as a three-fold release.
A conservative future package may retain baseline Exp15r unchanged on fold2 and
use only the independently passing fold0/fold1 components, but it still needs
packaged OOF/runtime and leaderboard validation.

## One bounded secondary diagnostic

The trainer has always maintained a named checkpoint selected by minimum online
study objective. That fixed selection rule yields Exp18 epoch12, Exp19 epoch21
and Exp20 epoch11. Exp18 named-best and Exp19 named-best already showed positive
component results. A single full-study audit of Exp20 `mls_multitask_best.pth`
is therefore justified to distinguish a checkpoint-timing problem from a model
family problem. This is secondary post-failure evidence and cannot alter the
primary conclusion. No other Exp20 checkpoint will be audited if it fails.

All aggregate reports were uploaded to MLflow run
`aa4d88acea4246a8a7e5c27a0a33a6c6`, independently verified as `FINISHED`.
Study-level prediction CSVs remained on Vast and were not copied locally or
uploaded to MLflow.
