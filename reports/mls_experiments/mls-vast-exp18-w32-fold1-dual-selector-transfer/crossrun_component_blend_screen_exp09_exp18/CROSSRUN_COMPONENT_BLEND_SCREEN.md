# Exp09 + Exp18 cross-run component-blend screen

This is a same-fold diagnostic screen, not an unbiased promotion estimate. No model or image inference was run.

- Baseline: `baseline_exp09: MAE=1.259036 mm, Boundary-F1=0.823729, objective=1.611578`
- Best nonbaseline: `exp18_epoch21__regression_only__a0p1: MAE=1.248085 mm, Boundary-F1=0.831034, objective=1.586016`
- Selected diagnostic: `exp18_epoch21__regression_only__a0p1: MAE=1.248085 mm, Boundary-F1=0.831034, objective=1.586016`
- Eligible nonbaseline candidates: `7`
- Improves same-runtime baseline objective: `True`
- Decision: Complementarity signal found under all frozen numerical gates; cross-fold/leaderboard validation is still required.

## Selected delta versus Exp09

- MAE: `-0.010951 mm`
- Boundary-F1: `+0.007306`
- objective: `-0.025562`

The locked production profile was not tuned in this screen. The guarded
profile is diagnostic only. Promotion requires independent cross-fold and
ultimately leaderboard evidence.
