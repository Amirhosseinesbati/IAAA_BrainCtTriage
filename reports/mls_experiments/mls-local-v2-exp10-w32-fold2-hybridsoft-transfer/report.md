# MLS experiment: mls-local-v2-exp10-w32-fold2-hybridsoft-transfer

- Status: `completed`
- Updated UTC: `2026-08-28T07:34:17.963568+00:00`
- MLflow run id: `a4c44492fcc141058e5aae71266c6c33`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 14.0,
  "train_loss": 1.825163653434699,
  "lr": 3.887395330218429e-05,
  "peak_vram_gb": 4.54531717300415,
  "val_loss": 1.6450761646421543,
  "selector_auc": 0.9142624767624767,
  "selector_f1": 0.8238993710691824,
  "selector_accuracy": 0.8325859491778774,
  "selector_peak_auc": 0.782772699032573,
  "selector_positive_mean": 0.6921963510425152,
  "selector_negative_mean": 0.17211097352403504,
  "keypoint_mae_px": 9.28208927898197,
  "mls_mae_mm": 2.377559211049173,
  "mls_rmse_mm": 3.838904844260529,
  "mls_f1_3mm": 0.8811188811188811,
  "mls_f1_5mm": 0.8323353293413174,
  "study_mls_mae_mm": 1.518155766423069,
  "study_mls_f1_3mm": 0.9803921568627451,
  "study_mls_f1_5mm": 0.9333333333333333,
  "study_boundary_f1": 0.9568627450980391,
  "selection_objective": 1.6472990378457522,
  "train_spatial_loss": 5.127231578420365,
  "train_coordinate_loss": 0.022734285442267757,
  "train_mls_loss": 0.5519199024652338,
  "train_threshold_loss": 0.22010530736064898,
  "train_selector_loss": 0.3719981042122333
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6318 | 28.4976 | 4.8816 | 0.0000 | 46.75 | 0.3755 | 0.3256 | 4.76 |
| 2 | 3.8007 | 6.5047 | 4.8816 | 0.0000 | 40.04 | 0.2765 | 0.2930 | 4.55 |
| 3 | 3.3950 | 3.2751 | 4.8816 | 0.0000 | 14.11 | 0.5103 | 0.4049 | 4.55 |
| 4 | 2.6826 | 2.8983 | 4.8816 | 0.0000 | 13.86 | 0.7730 | 0.6247 | 4.55 |
| 5 | 2.4878 | 2.4067 | 4.8816 | 0.0000 | 11.08 | 0.8120 | 0.6770 | 4.55 |
| 6 | 2.3243 | 2.6190 | 4.8816 | 0.0000 | 10.66 | 0.8356 | 0.7224 | 4.55 |
| 7 | 2.2634 | 2.3637 | 4.8816 | 0.0000 | 11.45 | 0.8399 | 0.7073 | 4.55 |
| 8 | 2.1388 | 2.6074 | 3.2660 | 0.5733 | 11.01 | 0.8645 | 0.7280 | 4.55 |
| 9 | 2.1081 | 2.3560 | 2.1103 | 0.8718 | 9.38 | 0.8803 | 0.7421 | 4.55 |
| 10 | 1.9748 | 2.4845 | 2.7824 | 0.6992 | 9.88 | 0.8832 | 0.7538 | 4.55 |
| 11 | 1.9272 | 2.6410 | 1.6461 | 0.9345 | 9.03 | 0.9076 | 0.8018 | 4.55 |
| 12 | 1.8635 | 2.4045 | 1.7066 | 0.8693 | 9.87 | 0.9084 | 0.7937 | 4.55 |
| 13 | 1.8639 | 2.2300 | 1.9472 | 0.8598 | 8.95 | 0.8893 | 0.7688 | 4.55 |
| 14 | 1.8252 | 2.3776 | 1.5182 | 0.9569 | 9.28 | 0.9143 | 0.7828 | 4.55 |
| 15 | 1.7446 | 2.3374 | 1.4756 | 0.9218 | 8.93 | 0.9112 | 0.7978 | 4.55 |
| 16 | 1.6985 | 2.2390 | 1.5421 | 0.9583 | 8.77 | 0.9119 | 0.7697 | 4.55 |
| 17 | 1.7138 | 2.2874 | 1.5459 | 0.9124 | 8.95 | 0.9124 | 0.7753 | 4.55 |
| 18 | 1.6640 | 2.3933 | 1.7907 | 0.8754 | 8.39 | 0.9054 | 0.7588 | 4.55 |
| 19 | 1.5934 | 2.5729 | 1.7694 | 0.9075 | 8.74 | 0.9059 | 0.7808 | 4.55 |
| 20 | 1.6128 | 2.4493 | 1.7298 | 0.8828 | 8.69 | 0.9013 | 0.7674 | 4.55 |
| 21 | 1.5798 | 2.5085 | 1.8087 | 0.8661 | 8.58 | 0.8880 | 0.7577 | 4.55 |
| 22 | 1.5537 | 2.4962 | 1.7335 | 0.8776 | 8.70 | 0.8957 | 0.7626 | 4.55 |
| 23 | 1.5498 | 2.4205 | 1.7162 | 0.8754 | 8.51 | 0.8959 | 0.7610 | 4.55 |
