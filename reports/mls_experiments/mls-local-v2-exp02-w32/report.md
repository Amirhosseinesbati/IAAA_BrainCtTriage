# MLS experiment: mls-local-v2-exp02-w32

- Status: `completed`
- Updated UTC: `2026-08-27T05:38:49.648405+00:00`
- MLflow run id: `695ef10c0a1f4cd19f12135eb3e2974e`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 3, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 12.0,
  "train_loss": 1.850313395316852,
  "lr": 7.500000000000001e-05,
  "peak_vram_gb": 4.545560836791992,
  "val_loss": 1.6918101872150333,
  "selector_auc": 0.9130455032369242,
  "selector_f1": 0.852112676056338,
  "selector_accuracy": 0.8380462724935732,
  "selector_positive_mean": 0.7896234799428642,
  "selector_negative_mean": 0.2312818803858398,
  "keypoint_mae_px": 9.007754444981604,
  "mls_mae_mm": 1.8392154110911925,
  "mls_rmse_mm": 2.703087800834641,
  "mls_f1_3mm": 0.7937915742793792,
  "mls_f1_5mm": 0.8041237113402062,
  "selection_objective": 2.221944555741592,
  "train_spatial_loss": 5.103698895290466,
  "train_coordinate_loss": 0.026003586915973122,
  "train_mls_loss": 0.7614495808282539,
  "train_threshold_loss": 0.2592735188875472,
  "train_selector_loss": 0.34509713335613446
}
```

## Epoch history

| epoch | train loss | val MLS MAE | kp MAE | selector AUC | selector F1 | F1@3 | F1@5 | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7710 | 32.2329 | 124.19 | 0.4151 | 0.0000 | 0.6755 | 0.5138 | 4.76 |
| 2 | 4.4255 | 4.9325 | 14.07 | 0.7696 | 0.0000 | 0.6568 | 0.5804 | 4.55 |
| 3 | 3.6938 | 4.8906 | 25.08 | 0.7921 | 0.0554 | 0.6226 | 0.4635 | 4.55 |
| 4 | 2.9760 | 2.9375 | 11.33 | 0.8386 | 0.6865 | 0.7277 | 0.6775 | 4.55 |
| 5 | 2.6675 | 2.2498 | 9.94 | 0.8306 | 0.8004 | 0.7692 | 0.7383 | 4.55 |
| 6 | 2.4072 | 2.3083 | 9.94 | 0.8393 | 0.7918 | 0.7788 | 0.7476 | 4.55 |
| 7 | 2.3215 | 2.0620 | 9.26 | 0.8863 | 0.8092 | 0.7560 | 0.7516 | 4.55 |
| 8 | 2.1068 | 2.0977 | 10.29 | 0.8979 | 0.8015 | 0.7417 | 0.7063 | 4.55 |
| 9 | 2.1223 | 1.9007 | 9.50 | 0.9062 | 0.7974 | 0.7822 | 0.7651 | 4.55 |
| 10 | 1.9284 | 2.3192 | 9.66 | 0.8997 | 0.8141 | 0.7752 | 0.7338 | 4.55 |
| 11 | 1.9165 | 2.4551 | 9.47 | 0.9020 | 0.8068 | 0.7637 | 0.7508 | 4.55 |
| 12 | 1.8503 | 1.8392 | 9.01 | 0.9130 | 0.8521 | 0.7938 | 0.8041 | 4.55 |
| 13 | 1.8708 | 2.0486 | 9.09 | 0.9072 | 0.8237 | 0.7904 | 0.7613 | 4.55 |
| 14 | 1.7444 | 2.2177 | 9.38 | 0.9073 | 0.8284 | 0.7580 | 0.7443 | 4.55 |
| 15 | 1.7680 | 2.0381 | 8.68 | 0.9085 | 0.8229 | 0.7699 | 0.7451 | 4.55 |
| 16 | 1.6471 | 2.0042 | 9.42 | 0.8761 | 0.7389 | 0.7807 | 0.7541 | 4.55 |
| 17 | 1.5630 | 2.1368 | 9.65 | 0.8823 | 0.6825 | 0.7220 | 0.7467 | 4.55 |
| 18 | 1.5897 | 2.1285 | 9.39 | 0.8771 | 0.7480 | 0.7646 | 0.7483 | 4.55 |
| 19 | 1.4883 | 2.1735 | 9.65 | 0.8848 | 0.7507 | 0.7293 | 0.7400 | 4.55 |
| 20 | 1.4445 | 2.2662 | 10.16 | 0.8806 | 0.7662 | 0.7352 | 0.7467 | 4.55 |

## Interpretation and end-to-end validation

- Training stopped at epoch 20 after eight consecutive epochs without improving
  the selection objective.  The best checkpoint is epoch 12, not the final
  epoch; the widening train/validation gap after epoch 12 is evidence of
  measurement-head overfitting.
- Best target-slice validation at epoch 12: MAE `1.8392 mm`, RMSE `2.7031 mm`,
  keypoint MAE `9.01 px`, selector AUC `0.9130`, selector F1 `0.8521`, and
  selection objective `2.2219`.
- Versus W18 experiment 01, target-slice MAE improved from `2.0371` to `1.8392`
  (about `9.7%`) and the selection objective improved from `2.5203` to `2.2219`
  (about `11.8%`).
- Strict CUDA end-to-end evaluation completed all `70/70` fold-0 studies and
  `1723` slices with zero failures in `109.85 s`.

### Study-level profiles

| profile | MAE mm | RMSE mm | bias mm | F1@3 | F1@5 | combined macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 / top-3 / p90 | 2.0001 | 2.8045 | +0.5164 | 0.7647 | 0.8000 | 0.5891 |
| Legacy locked relative-component r=.3 / q=.75 | 1.9540 | 2.6073 | -0.0394 | 0.7742 | 0.8108 | 0.6396 |
| Fold-0 locked relative-component r=.5 / q=.75 | 1.9669 | 2.6879 | +0.0412 | 0.7869 | 0.8718 | 0.6396 |
| Fold-0 locked anchor-window ±2 / q=.65 | 1.6380 | 2.4423 | -0.1912 | 0.7937 | 0.8108 | 0.6025 |
| Fold-0 locked top-7 / q=.75 | 1.6746 | 2.5953 | +0.0138 | 0.7812 | 0.8718 | 0.6269 |

The three fold-0-derived profiles are hypotheses, not independent estimates.
They were frozen before experiment 03 and must be tested unchanged on fold 1.

### Error decomposition

- Slice selector AUC against annotated target slices: `0.9355`.
- Target retrieval recall: top-1 `0.9744`, top-3/top-5/top-10 `1.0000`.
- Oracle annotated-target MAE: median `1.7107 mm`, p90 `1.0746 mm`, max
  `1.5619 mm`.
- Predicted selector top-3 MAE: median `1.8838 mm`, p90 `2.0001 mm`; predicted
  top-5 median `1.8801 mm`.
- Approximate added selector/pooling cost above oracle-target median:
  `0.1731 mm`.

The selector is no longer the dominant bottleneck.  The major remaining gains
come from stable study pooling and measurement calibration.  W32 materially
improves the oracle p90 floor and overall combined score, while the previously
locked W18 pooling ratio does not minimize W32 MAE.

### Decision

W32 replaces W18 as the leading MLS checkpoint because it improves the
target-slice objective, fixed-profile end-to-end MAE/RMSE, bias, and combined
score.  Experiment 03 changes only the validation fold to provide an independent
OOF transfer test of the three frozen pooling hypotheses.  No fold-1 result may
be used to retroactively alter those hypotheses.

Artifacts: `end_to_end/metrics.json`, `end_to_end/decomposition.json`,
`end_to_end/decomposition_report.md`, and
`end_to_end/postprocessing_search.json`.  Per-study predictions remain local and
are intentionally excluded from remote artifact upload.
