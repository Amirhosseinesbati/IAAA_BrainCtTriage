# Exp20 post-failure alpha-sensitivity diagnostic plan

## Immutable primary conclusion

The preregistered fold2 primary test failed. With the fixed regression-only
alpha 0.10 recipe, MAE improved by `0.014964986mm`, but Boundary-F1 decreased by
`0.007744108` and objective worsened by `0.000523229`. No result from this
diagnostic can rescue, relabel or replace that primary conclusion.

## Diagnostic question

The observed failure is concentrated at the 3mm boundary: F1@3mm decreased
from `0.851851852` to `0.836363636`, while F1@5mm remained `0.933333333` and
MAE improved. The challenger also made bias more negative by `0.062630905mm`.
This screen asks whether the boundary loss is monotonic in challenger weight and
whether a smaller regression contribution lies on a better MAE/Boundary Pareto
front.

## Frozen diagnostic grid

- Baseline: Exp15r epoch17 saved full-study CUDA predictions.
- Challenger: Exp20 epoch21 saved full-study CUDA predictions.
- Alphas: `0.01`, `0.025`, `0.05`, `0.075`, `0.10`.
- Component modes: the existing four fixed screen modes. Interpretation gives
  priority to `regression_only`; differences involving selector/peak/heatmap are
  mechanistic diagnostics, not deployment choices.
- Locked production profile and numerical gates remain unchanged.
- The guarded profile emitted by the existing screen remains diagnostic only.
- Model/image inference is forbidden; the screen consumes saved CUDA
  predictions and performs only lightweight aggregate post-processing.

The screen output may guide a future independently validated recipe. It cannot
authorize release, local checkpoint promotion, package integration or a claim
of three-fold replication. Any new alpha selected after this fold2 failure must
be validated on genuinely independent evidence such as an unused fold or a
carefully limited leaderboard submission.
