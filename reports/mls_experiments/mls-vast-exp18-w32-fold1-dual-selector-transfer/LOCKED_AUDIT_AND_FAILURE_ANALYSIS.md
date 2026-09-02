# Exp18 locked audit and failure analysis

## Decision

Exp18 is not promoted. The incumbent fold1 model remains Exp09/epoch15.

This is based on a completed full-study CUDA audit, not the online metric seen
during training. Twelve preregistered checkpoint labels were evaluated over all
67 fold1 studies: 804/804 study-checkpoint evaluations completed without a
failure or CPU model fallback. A 6048-profile grid per checkpoint produced
72576 rows, after which the already-frozen production profile was applied.

## Frozen gate

The production profile was fixed before inspecting Exp18 full-study results:

- family: `severity_window`
- radius: `3`
- selector gate: `0.5`
- minimum active slices: `3`
- quantile: `0.75`
- probability weighted: `true`
- heatmap guard: `0.0`

Promotion required all three conditions:

- MAE `<= 1.2586648668 mm`
- Boundary-F1 `>= 0.82`
- selection objective `<= 1.6112072397`

No Exp18 checkpoint passed all three.

## Locked-profile checkpoint ranking

| candidate | source epoch | MAE (mm) | Boundary-F1 | objective |
|---|---:|---:|---:|---:|
| epoch021 | 21 | 1.392992 | 0.771236 | 1.850521 |
| epoch015 | 15 | 1.483992 | 0.750739 | 1.982514 |
| epoch019 | 19 | 1.515612 | 0.736111 | 2.043390 |
| epoch023 | 23 | 1.573683 | 0.747731 | 2.078220 |
| best_mae | 14 | 1.635402 | 0.735714 | 2.163974 |
| best_selector_auc | 14 | 1.635402 | 0.735714 | 2.163974 |
| best_objective | 12 | 1.939928 | 0.755952 | 2.428023 |
| best_study_boundary | 12 | 1.939928 | 0.755952 | 2.428023 |
| best_study | 12 | 1.939928 | 0.755952 | 2.428023 |
| epoch017 | 17 | 2.030694 | 0.677858 | 2.674977 |
| epoch013 | 13 | 2.276171 | 0.650000 | 2.976171 |
| best_peak_auc | 10 | 3.105654 | 0.659750 | 3.786154 |

The best locked result, epoch21, had RMSE `2.2025165 mm`, bias
`-0.045889 mm`, F1@3mm `0.785714`, and F1@5mm `0.756757`.

## Why the online result was misleading

The best online checkpoint was epoch12 with study MAE `0.899815 mm`,
Boundary-F1 `0.847662`, and objective `1.254940`. The full ordered-series audit
of that model under the locked deployment path yielded MAE `1.939928 mm`,
Boundary-F1 `0.755952`, and objective `2.428023`.

Moreover, epoch21—not epoch12—was the best locked checkpoint. This rank reversal
means online sampled validation is useful for training health but cannot select
a release checkpoint. Full-study evaluation is mandatory for all future MLS
runs.

## What did improve, and what did not

The unrestricted same-fold grid found a diagnostic profile for
`best_objective`:

- `severity_window`, radius `2`
- selector gate `0.7`, minimum active slices `3`
- quantile `0.65`, probability weighted
- MAE `1.127073515 mm`
- RMSE `1.795146566 mm`
- bias `-0.125642 mm`
- F1@3mm `0.842105`, F1@5mm `0.777778`
- Boundary-F1 `0.809942`, objective `1.507190`

This is evidence that Exp18 contains useful regression signal. It is not a
promotion result: the profile was selected on the same fold and Boundary-F1 is
still about `0.0101` below the frozen floor.

Relative to Exp17's best locked checkpoint, Exp18/epoch21 improved MAE by only
about `0.0069 mm`, while Boundary-F1 fell by about `0.0222` and the objective
worsened by about `0.0375`. Separating presence and peak heads therefore helped
the online proxy but did not solve transferred boundary calibration.

## Failure classification

The evidence argues against a corrupt-data or broken-audit explanation:

- all 804 CUDA evaluations completed;
- all 67 studies were present for every candidate;
- no evaluation failure was recorded;
- Exp09 backward parity was reproduced exactly before launch;
- training completed 23/23 finite epochs without NaN, OOM or CPU fallback.

The most likely failure is objective/deployment mismatch:

1. sampled online study estimates do not reproduce the full ordered series;
2. presence/peak probabilities are not calibrated for the frozen gate/count and
   ranking path across checkpoints;
3. the regression objective does not directly protect the 3mm and 5mm triage
   boundaries;
4. single-slice input lacks adjacent anatomical context needed to distinguish a
   true maximal-shift slice from local ambiguity.

## Next experiment gate

Before training again, run a component-level blend screen using saved CUDA
predictions only:

- Exp09 supplies the trusted selector gate;
- then Exp09 supplies both selector and peak ranking;
- Exp18 contributes progressively only heatmap/regression components;
- the production profile stays frozen;
- the four modes are full blend, Exp09 selector only, Exp09 selector+peak, and
  Exp09 selector+peak+heatmap (regression-only blend);
- challenger contribution alpha is preregistered as `0.10`, `0.25`, `0.50`,
  `0.75`, and `1.00`;
- the two fixed Exp18 candidates are epoch21 (best locked) and
  `best_objective`/epoch12 (best online).

The Exp09 CUDA regeneration completed before any blend result was inspected.
Its MAE differed from the historical artifact by only `+0.0003709553 mm`, while
Boundary-F1 was exactly unchanged. The identical 67-study/1346-slice coverage
and exact legacy peak=selector fallback support a cross-runtime numerical drift
interpretation. A separate baseline-parity tolerance of `0.001` was fixed at
this point. It does not relax or replace any release gate.

The completed screen contained 41 locked-profile candidates plus the parallel
guarded diagnostic rows. Seven candidates passed all frozen numerical gates;
every eligible blend used alpha `0.10`. For Exp18/epoch21, full,
baseline-selector, baseline-selector+peak and regression-only modes all produced
exactly the same MAE `1.248084723`, Boundary-F1 `0.831034483`, and objective
`1.586015757`. The only necessary change is therefore a 90/10 blend of the
Exp09/Exp18 `mls_mm` outputs. A minimum-component tie-break selects
`regression_only`; this changes no metric and avoids attributing the gain to
selector, peak or heatmap components that made no numerical difference.

This screen cannot promote a model because it is same-fold, but it can answer
whether Exp18 offers complementary regression signal. If every component blend
fails, Exp18 should be treated as neither a replacement nor a useful ensemble
member.

The next training architecture should then be a controlled change rather than
more epochs or a seed sweep. Priority candidates are 2.5D adjacent-slice
context, auxiliary boundary-aware supervision at 3/5mm, and cross-fold
calibration selected on other OOF folds. Each needs an ablation and the same
full-study CUDA gate.

## Artifact locations on the Vast server

- audit root: `/workspace/IAAA_BrainCtTriage/reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/end_to_end_checkpoint_audit`
- audit status: `end_to_end_checkpoint_audit/audit_status.json`
- pooling grid: `end_to_end_checkpoint_audit/checkpoint_pooling_expanded/checkpoint_pooling_grid.csv`
- pooling summary: `end_to_end_checkpoint_audit/checkpoint_pooling_expanded/checkpoint_pooling_summary.json`
- promotion decision: `end_to_end_checkpoint_audit/promotion_gate.json`
- MLflow run: `18474f1d10234ca5900caefe3f62c2eb`

The server must not be stopped or destroyed without user coordination.
