# MLS experiment: mls-local-v2-exp07-w32-fold2-peakaware-crossfold

- Status: `completed`
- Updated UTC: `2026-08-27T21:15:30.325756+00:00`
- MLflow run id: `3b07a5d204b6452696ad89c3a03ec1d9`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.5, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.7, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 1, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 19.0,
  "train_loss": 1.7282237670720153,
  "lr": 3.5659838364445505e-05,
  "peak_vram_gb": 4.54580545425415,
  "val_loss": 1.657409878496303,
  "selector_auc": 0.9213588588588589,
  "selector_f1": 0.8343558282208589,
  "selector_accuracy": 0.8385650224215246,
  "selector_peak_auc": 0.7743409958285257,
  "selector_positive_mean": 0.672974327408538,
  "selector_negative_mean": 0.17672229338339612,
  "keypoint_mae_px": 9.183139569408542,
  "mls_mae_mm": 2.4219331635950923,
  "mls_rmse_mm": 4.007640357485053,
  "mls_f1_3mm": 0.8544600938967136,
  "mls_f1_5mm": 0.8036253776435045,
  "study_mls_mae_mm": 1.4089642731556253,
  "study_mls_f1_3mm": 0.9615384615384616,
  "study_mls_f1_5mm": 0.8372093023255814,
  "study_boundary_f1": 0.8993738819320215,
  "selection_objective": 1.649537079862153,
  "train_spatial_loss": 5.026595984640172,
  "train_coordinate_loss": 0.018681653594034914,
  "train_mls_loss": 0.32312841389868086,
  "train_threshold_loss": 0.1534433802813808,
  "train_selector_loss": 0.3661075050464532
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6553 | 44.2743 | 4.8816 | 0.0000 | 106.52 | 0.4075 | 0.3346 | 4.76 |
| 2 | 4.0390 | 4.6102 | 4.8816 | 0.0000 | 14.25 | 0.1893 | 0.2277 | 4.55 |
| 3 | 3.5025 | 4.1068 | 4.8816 | 0.0000 | 18.07 | 0.2119 | 0.2650 | 4.55 |
| 4 | 3.2517 | 4.3814 | 4.8816 | 0.0000 | 18.33 | 0.2852 | 0.2978 | 4.55 |
| 5 | 2.8689 | 2.9449 | 4.8816 | 0.0000 | 13.22 | 0.3356 | 0.4539 | 4.55 |
| 6 | 2.5747 | 2.6165 | 4.8816 | 0.0000 | 11.55 | 0.8042 | 0.7405 | 4.55 |
| 7 | 2.3763 | 2.3825 | 4.8816 | 0.0000 | 10.20 | 0.8278 | 0.7119 | 4.55 |
| 8 | 2.2228 | 2.4952 | 4.8816 | 0.0000 | 10.34 | 0.8717 | 0.7551 | 4.55 |
| 9 | 2.2208 | 2.4374 | 4.8816 | 0.0000 | 10.84 | 0.8894 | 0.7662 | 4.55 |
| 10 | 2.0913 | 2.6662 | 4.8816 | 0.0000 | 10.21 | 0.8879 | 0.7295 | 4.55 |
| 11 | 2.0457 | 2.7681 | 4.8816 | 0.0000 | 10.36 | 0.9147 | 0.7952 | 4.55 |
| 12 | 1.9863 | 2.4423 | 4.8816 | 0.0000 | 9.98 | 0.9085 | 0.7914 | 4.55 |
| 13 | 2.0010 | 2.2492 | 4.7183 | 0.0787 | 9.03 | 0.9016 | 0.7680 | 4.55 |
| 14 | 1.9474 | 2.4389 | 2.5793 | 0.7757 | 9.89 | 0.9196 | 0.7896 | 4.55 |
| 15 | 1.8866 | 2.3484 | 2.4729 | 0.7656 | 9.75 | 0.9122 | 0.7567 | 4.55 |
| 16 | 1.8406 | 2.3539 | 2.1157 | 0.8130 | 9.64 | 0.9209 | 0.7784 | 4.55 |
| 17 | 1.8737 | 2.3454 | 2.5679 | 0.7453 | 9.74 | 0.9123 | 0.7525 | 4.55 |
| 18 | 1.8309 | 2.5233 | 2.7406 | 0.6643 | 9.54 | 0.9085 | 0.7408 | 4.55 |
| 19 | 1.7282 | 2.4219 | 1.4090 | 0.8994 | 9.18 | 0.9214 | 0.7743 | 4.55 |
| 20 | 1.7454 | 2.2387 | 1.9520 | 0.8661 | 8.99 | 0.9053 | 0.7365 | 4.55 |
| 21 | 1.7172 | 2.1072 | 1.7478 | 0.9002 | 8.57 | 0.9062 | 0.7617 | 4.55 |
| 22 | 1.6884 | 2.2963 | 2.1470 | 0.7644 | 9.79 | 0.8896 | 0.7281 | 4.55 |
| 23 | 1.6478 | 2.2364 | 1.7599 | 0.8661 | 8.53 | 0.9104 | 0.7655 | 4.55 |
| 24 | 1.6361 | 2.3774 | 1.5863 | 0.8776 | 8.53 | 0.9101 | 0.7731 | 4.55 |
| 25 | 1.6194 | 2.1948 | 1.7229 | 0.8400 | 8.57 | 0.9129 | 0.7720 | 4.55 |
| 26 | 1.6555 | 2.1861 | 1.5830 | 0.8521 | 8.51 | 0.9063 | 0.7671 | 4.55 |
| 27 | 1.6382 | 2.3287 | 1.5535 | 0.8886 | 9.03 | 0.9122 | 0.7709 | 4.55 |
