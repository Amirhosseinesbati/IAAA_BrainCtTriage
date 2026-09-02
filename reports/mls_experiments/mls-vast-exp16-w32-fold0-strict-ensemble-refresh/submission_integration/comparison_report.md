# MLS submission integration audit

- Finished UTC: `2026-09-02T04:04:47.983437+00:00`
- GPU: `NVIDIA GeForce RTX 3060`
- Studies: `70`
- Compute policy: model forward passes are CUDA-only; no CPU resize was allowed.
- Interpretation: single-model fold0 rows are OOF evidence; ensemble rows are diagnostic only.

## Metrics

| Candidate | MAE (mm) | Boundary F1 | Objective | Bias (mm) |
|---|---:|---:|---:|---:|
| fold0_baseline | 1.665417 | 0.822263 | 2.020892 | -0.378399 |
| fold0_challenger | 1.604478 | 0.827333 | 1.949813 | 0.037588 |
| baseline_ensemble_diagnostic | 0.960407 | 0.871154 | 1.218099 | -0.002062 |
| challenger_ensemble_diagnostic | 1.028320 | 0.878418 | 1.271483 | 0.020684 |

## Paired diagnostic ensemble delta

- MAE delta: `+0.067913` mm
- Boundary-F1 delta: `+0.007264`
- Improved/worse/tied studies by absolute error: `13/10/47`

## Packaged-runtime parity

- Reference available: `True`
- Studies checked: `70`
- Index mismatches: `0`
- Max |selector delta|: `0`
- Max |MLS delta|: `1.1920929e-07` mm
- Max |heatmap-peak delta|: `0`
- Gate passed: `True`
