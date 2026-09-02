# Exp09 + Exp18 cross-run component-blend screen

This is a same-fold diagnostic screen, not an unbiased promotion estimate. No model or image inference was run.

- Baseline: `baseline_exp09: MAE=1.548354 mm, Boundary-F1=0.892593, objective=1.763169`
- Best nonbaseline: `exp20_epoch21__regression_only__a0p025: MAE=1.547159 mm, Boundary-F1=0.892593, objective=1.761974`
- Selected diagnostic: `exp20_epoch21__regression_only__a0p025: MAE=1.547159 mm, Boundary-F1=0.892593, objective=1.761974`
- Eligible nonbaseline candidates: `0`
- Improves same-runtime baseline objective: `True`
- Decision: A same-runtime complementarity signal improved the baseline objective, but no blend passed all frozen release gates.

## Selected delta versus Exp09

- MAE: `-0.001195 mm`
- Boundary-F1: `+0.000000`
- objective: `-0.001195`

The locked production profile was not tuned in this screen. The guarded
profile is diagnostic only. Promotion requires independent cross-fold and
ultimately leaderboard evidence.
