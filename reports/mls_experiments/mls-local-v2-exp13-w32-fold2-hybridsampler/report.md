# MLS experiment: mls-local-v2-exp13-w32-fold2-hybridsampler

- Status: `failed`
- Updated UTC: `2026-08-28T14:49:44.782857+00:00`
- MLflow run id: `0a2cf48a6fce417ba2f89c50a7ad185f`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "hybrid_study_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`
- Error: `MlflowException: API request to https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/api/2.0/mlflow/runs/log-batch failed with exception HTTPSConnectionPool(host='dagshub.com', port=443): Max retries exceeded with url: /amiresbati62/BrainCtTriage.mlflow/api/2.0/mlflow/runs/log-batch (Caused by NameResolutionError("HTTPSConnection(host='dagshub.com', port=443): Failed to resolve 'dagshub.com' ([Errno 11001] getaddrinfo failed)"))`

## Best validation

```json
{
  "epoch": 14.0,
  "train_loss": 1.8618139936381928,
  "lr": 3.887395330218429e-05,
  "peak_vram_gb": 4.54531717300415,
  "val_loss": 1.646551482282134,
  "selector_auc": 0.906138281138281,
  "selector_f1": 0.8006535947712419,
  "selector_accuracy": 0.8176382660687593,
  "selector_peak_auc": 0.7861231916215495,
  "selector_positive_mean": 0.6384733753088672,
  "selector_negative_mean": 0.16402213907262886,
  "keypoint_mae_px": 8.972645995972512,
  "mls_mae_mm": 2.239272742859415,
  "mls_rmse_mm": 3.271054081886109,
  "mls_f1_3mm": 0.8658823529411764,
  "mls_f1_5mm": 0.823170731707317,
  "study_mls_mae_mm": 1.502948647314933,
  "study_mls_f1_3mm": 0.9411764705882353,
  "study_mls_f1_5mm": 0.9090909090909091,
  "study_boundary_f1": 0.9251336898395721,
  "selection_objective": 1.6996121270666482,
  "train_spatial_loss": 5.092233275012055,
  "train_coordinate_loss": 0.021550672386957376,
  "train_mls_loss": 0.5653344993644971,
  "train_threshold_loss": 0.25425151491197506,
  "train_selector_loss": 0.4112215607987839
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6381 | 78.5565 | 4.8816 | 0.0000 | 148.81 | 0.4486 | 0.3712 | 4.76 |
| 2 | 4.5779 | 9.9302 | 4.8816 | 0.0000 | 117.75 | 0.5402 | 0.4915 | 4.55 |
| 3 | 4.2002 | 6.2647 | 4.8816 | 0.0000 | 33.99 | 0.3443 | 0.3793 | 4.55 |
| 4 | 3.4315 | 3.5601 | 4.8816 | 0.0000 | 14.11 | 0.6211 | 0.5479 | 4.55 |
| 5 | 2.7920 | 2.6071 | 4.8816 | 0.0000 | 10.59 | 0.8202 | 0.7133 | 4.55 |
| 6 | 2.5306 | 2.4684 | 4.8816 | 0.0000 | 10.49 | 0.8287 | 0.7875 | 4.55 |
| 7 | 2.3905 | 2.6474 | 4.8816 | 0.0000 | 10.21 | 0.8443 | 0.7569 | 4.55 |
| 8 | 2.2468 | 2.5640 | 3.3083 | 0.5384 | 9.72 | 0.8654 | 0.7212 | 4.55 |
| 9 | 2.1394 | 2.7412 | 1.5486 | 0.9361 | 9.99 | 0.8880 | 0.7788 | 4.55 |
| 10 | 2.0281 | 2.6394 | 1.7137 | 0.8541 | 9.39 | 0.8831 | 0.7513 | 4.55 |
| 11 | 2.0046 | 2.4149 | 1.7110 | 0.8674 | 9.57 | 0.9075 | 0.7610 | 4.55 |
| 12 | 1.9130 | 2.3250 | 1.5486 | 0.9239 | 9.10 | 0.9092 | 0.7676 | 4.55 |
| 13 | 1.9486 | 2.5339 | 1.5233 | 0.9239 | 9.23 | 0.9038 | 0.7521 | 4.55 |
| 14 | 1.8618 | 2.2393 | 1.5029 | 0.9251 | 8.97 | 0.9061 | 0.7861 | 4.55 |
| 15 | 1.8089 | 2.4603 | 1.5257 | 0.9239 | 9.26 | 0.9033 | 0.7782 | 4.55 |
| 16 | 1.7875 | 2.5039 | 1.8348 | 0.8887 | 8.86 | 0.9124 | 0.7777 | 4.55 |
| 17 | 1.7412 | 2.4836 | 1.7281 | 0.9267 | 8.72 | 0.9185 | 0.7907 | 4.55 |
| 18 | 1.7217 | 2.5170 | 1.9863 | 0.9002 | 8.91 | 0.9073 | 0.7796 | 4.55 |
| 19 | 1.6940 | 2.4038 | 1.5774 | 0.9345 | 8.91 | 0.9207 | 0.7902 | 4.55 |
