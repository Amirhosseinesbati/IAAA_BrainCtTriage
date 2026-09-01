# MLS experiment: mls-local-v2-exp03-w32-fold1

- Status: `completed`
- Updated UTC: `2026-08-27T08:57:33.697001+00:00`
- MLflow run id: `df4717b978054f4b9874b0121fef579e`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 1, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 7, "aggregation": "quantile", "selector_relative_ratio": 0.5, "aggregation_quantile": 0.75, "anchor_window_radius": 2, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 14.0,
  "train_loss": 1.8240947719411493,
  "lr": 6.434016163555452e-05,
  "peak_vram_gb": 4.545072555541992,
  "val_loss": 1.6131521574876926,
  "selector_auc": 0.9052666180991132,
  "selector_f1": 0.8212703101920237,
  "selector_accuracy": 0.819672131147541,
  "selector_positive_mean": 0.7681354232828854,
  "selector_negative_mean": 0.20392499426428598,
  "keypoint_mae_px": 8.262450915415846,
  "mls_mae_mm": 1.8772323965550972,
  "mls_rmse_mm": 2.5451876935418642,
  "mls_f1_3mm": 0.7023809523809523,
  "mls_f1_5mm": 0.7323943661971831,
  "selection_objective": 2.3294251580719365,
  "train_spatial_loss": 5.176937512231467,
  "train_coordinate_loss": 0.02853789110502288,
  "train_mls_loss": 0.8011240597202701,
  "train_threshold_loss": 0.29066168847155,
  "train_selector_loss": 0.2862442612740811
}
```

## Epoch history

| epoch | train loss | val MLS MAE | kp MAE | selector AUC | selector F1 | F1@3 | F1@5 | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.8398 | 105.1781 | 335.58 | 0.3347 | 0.0000 | 0.6273 | 0.4633 | 4.76 |
| 2 | 4.4746 | 25.4801 | 63.49 | 0.6508 | 0.0000 | 0.6198 | 0.4408 | 4.54 |
| 3 | 3.8460 | 3.6668 | 13.19 | 0.7945 | 0.0000 | 0.6356 | 0.5571 | 4.55 |
| 4 | 3.3910 | 2.9145 | 12.33 | 0.7126 | 0.0000 | 0.7018 | 0.6512 | 4.55 |
| 5 | 2.9986 | 2.0266 | 10.43 | 0.7829 | 0.0000 | 0.7421 | 0.7487 | 4.55 |
| 6 | 2.7106 | 2.2100 | 9.62 | 0.8301 | 0.7786 | 0.7000 | 0.7255 | 4.54 |
| 7 | 2.4744 | 2.2263 | 10.24 | 0.8355 | 0.7873 | 0.6892 | 0.6927 | 4.54 |
| 8 | 2.2431 | 1.9895 | 8.96 | 0.8874 | 0.8194 | 0.7117 | 0.7343 | 4.54 |
| 9 | 2.1661 | 1.9582 | 8.48 | 0.9018 | 0.8172 | 0.7439 | 0.7122 | 4.54 |
| 10 | 2.0469 | 1.9964 | 8.58 | 0.9149 | 0.8224 | 0.7446 | 0.7081 | 4.55 |
| 11 | 1.9361 | 1.9808 | 8.43 | 0.9228 | 0.8403 | 0.7081 | 0.7184 | 4.55 |
| 12 | 1.9721 | 2.2325 | 8.75 | 0.8825 | 0.7301 | 0.6967 | 0.6545 | 4.54 |
| 13 | 1.8401 | 1.9100 | 8.30 | 0.8929 | 0.8013 | 0.7419 | 0.7400 | 4.54 |
| 14 | 1.8241 | 1.8772 | 8.26 | 0.9053 | 0.8213 | 0.7024 | 0.7324 | 4.55 |
| 15 | 1.7494 | 1.9152 | 8.53 | 0.8891 | 0.8122 | 0.7197 | 0.6931 | 4.54 |
| 16 | 1.7003 | 1.9990 | 8.68 | 0.8867 | 0.7906 | 0.6957 | 0.7184 | 4.55 |
| 17 | 1.6025 | 2.0382 | 8.31 | 0.8789 | 0.7480 | 0.6951 | 0.7404 | 4.54 |
| 18 | 1.5947 | 2.0169 | 7.96 | 0.8778 | 0.7545 | 0.7040 | 0.7389 | 4.54 |
| 19 | 1.5460 | 2.1602 | 9.13 | 0.8844 | 0.6919 | 0.6687 | 0.6825 | 4.54 |
| 20 | 1.5110 | 2.1333 | 8.35 | 0.8876 | 0.7645 | 0.6667 | 0.6792 | 4.54 |
| 21 | 1.4418 | 2.1623 | 7.84 | 0.8798 | 0.7707 | 0.6529 | 0.6957 | 4.54 |
| 22 | 1.4142 | 2.0600 | 8.09 | 0.8758 | 0.7590 | 0.7257 | 0.6912 | 4.54 |
