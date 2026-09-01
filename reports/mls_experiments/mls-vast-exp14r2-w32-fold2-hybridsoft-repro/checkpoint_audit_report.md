# Exp14r2 MLS checkpoint audit and decision record

## Provenance and compute contract

- Run: `mls-vast-exp14r2-w32-fold2-hybridsoft-repro`
- MLflow run ID: `efdfb96e8e2740918836991ac0fff2bf`
- Source commit: `6e762037d15d499c8e9229a4f171826af02ba5ca`
- GPU: RTX 3060 12 GB; model training and inference were CUDA-only, with no CPU fallback.
- Training completed all 23 epochs with exit code 0. Peak allocated VRAM was about 4.55 GiB.
- Start/end: 2026-09-01 17:29:26 UTC / 18:44:23 UTC.
- Evaluation cohort: competition fold 2, 67 studies. Every one of the eight audited checkpoints completed 67/67 studies (402/402 total inference jobs) without failure.

## Pre-registered reproducibility gate

The epoch-15 comparison against the historical Exp10 reference failed the pre-registered gate, despite an essentially matching selector AUC:

| Check | Exp14r2 | Exp10 reference | Tolerance | Result |
| --- | ---: | ---: | ---: | --- |
| Selector AUC | 0.908015 | 0.911161 | absolute delta <= 0.03 | pass |
| Study MLS MAE (mm) | 1.888036 | 1.475568 | absolute delta <= 0.35 | fail |
| Study boundary F1 | 0.840166 | 0.921769 | absolute delta <= 0.06 | fail |

Because this gate failed, the planned sigma-annealing follow-up was deliberately not launched. The failure means this trajectory cannot be treated as an exact reproduction or automatic replacement for Exp10.

## Full end-to-end audit

The pre-registered odd snapshots (epochs 13, 15, 17, 19, 21 and 23) were evaluated first. The global balanced result among those six candidates was epoch 15:

- Same-fold diagnostic MAE: 1.675602 mm
- Same-fold diagnostic boundary F1: 0.896853
- Selection objective: 1.881895
- Best raw MAE found in that six-snapshot search: 1.658658 mm

Two additional checkpoints selected by training-time metrics were then audited:

| Checkpoint | Selection reason | Best same-fold MAE (mm) | Boundary F1 | Objective |
| --- | --- | ---: | ---: | ---: |
| epoch 12 | best internal objective/study behavior | 1.682793 | 0.917647 | 1.847499 |
| epoch 16 | best selector AUC | **1.603443** | 0.908730 | **1.785983** |

Epoch 16 is therefore the strongest diagnostic checkpoint in this run, using `topk`, size 7, selector gate 0.3, at least 3 active slices, quantile 0.75, unweighted pooling and heatmap guard 0.5. This profile was selected and measured on the same fold, so these numbers are exploratory and are not an unbiased production estimate.

## Transfer and calibration analysis

The historical locked production profile (`severity_window`, size 3, selector gate 0.5, at least 3 active slices, quantile 0.75, probability weighted, no heatmap guard) exposes the important failure mode:

| Model/checkpoint | Locked-profile MAE (mm) | Locked-profile boundary F1 |
| --- | ---: | ---: |
| Exp10 epoch 15 | **1.714359** | **0.894697** |
| Exp14r2 epoch 12 | 1.944410 | 0.892593 |
| Exp14r2 epoch 16 | 1.968059 | 0.818605 |
| Exp14r2 epoch 15 | 2.088763 | 0.820000 |

The representation did not simply collapse. Selector rank separation improved from approximately 0.834/0.836 (max/top-3 mean) in Exp10 to 0.859/0.860 in Exp14r2 epoch 16. The absolute probability scale shifted, however: the positive/negative median selector maxima changed from about 0.871/0.625 to 0.830/0.457. A post-processing rule tuned to the old calibration consequently transfers poorly.

Prediction-level cross-seed blending confirmed that this is not fixed by a simple average. Same-fold tuning made a 50/50 blend look attractive (MAE 1.630431, boundary F1 0.917508), but under the locked production rule performance degraded monotonically as Exp14r2 weight increased:

| Exp10 / Exp14r2 weight | Locked-profile MAE (mm) | Locked-profile boundary F1 |
| --- | ---: | ---: |
| 100 / 0 | **1.714359** | **0.894697** |
| 75 / 25 | 1.757390 | 0.880770 |
| 50 / 50 | 1.867160 | 0.854900 |
| 25 / 75 | 1.923370 | 0.841680 |
| 0 / 100 | 1.968060 | 0.818600 |

Historical within-trajectory SWA (Exp10 epochs 13/15/17) was also weak at roughly 1.95 mm under the locked rule. Model or prediction averaging is therefore not promoted on current evidence.

## Root-cause hypothesis

The historical trainer seeded Python, NumPy and PyTorch, but enabled cuDNN benchmarking, did not request deterministic algorithms, did not seed Python/NumPy explicitly inside DataLoader workers, and relied on an implicit global sampler generator. GTX 1660 and RTX 3060 can consequently choose different CUDA/cuDNN kernels and consume augmentation/sampling randomness differently. The curves already diverged at epoch 1 despite similar training loss. This is consistent with a strong but differently calibrated trajectory, not data corruption.

The data-transfer integrity checks were independent of model results and passed exactly. The apparent `5420` versus `3484` discrepancy was a dataset-version/preprocessing-contract difference; the final training contract contains 3484 rows (1781 positive, 1703 negative), 338 studies, and all referenced files resolve. It is not evidence of damaged transfer.

## Decision

1. Do not promote Exp14r2 over the current trusted Exp10 epoch-15 production candidate.
2. Retain Exp14r2 epoch 16 as a complementary diagnostic checkpoint, not as a production checkpoint.
3. Do not launch the previously planned sigma-annealing experiment from this failed reproduction gate.
4. First run a one-factor deterministic control with the same architecture, split, data, loss and hyperparameters. The only intended change is the explicit reproducibility policy (deterministic CUDA policy, epoch-addressable RNG seeds and deterministic worker seeding).
5. Evaluate that control with both the locked production profile and a separately labelled same-fold diagnostic search. Only after calibration stability is understood should architecture/loss experiments resume.

This order avoids spending GPU time on an intervention whose apparent gain could again be a trajectory/calibration artifact.
