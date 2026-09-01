# MLS experiment: mls-local-v2-exp09-w32-fold1-hybridsoft-transfer

- Status: `completed`
- Updated UTC: `2026-08-28T04:32:48.781982+00:00`
- MLflow run id: `5e1449cbe73b4152b66589df7f20898c`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 1, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 17.0,
  "train_loss": 1.722546440824283,
  "lr": 1.8825509907063327e-05,
  "peak_vram_gb": 4.54458475112915,
  "val_loss": 1.674409465243419,
  "selector_auc": 0.90348975639226,
  "selector_f1": 0.7671691792294807,
  "selector_accuracy": 0.7928464977645305,
  "selector_peak_auc": 0.8033428606934261,
  "selector_positive_mean": 0.5985943927368818,
  "selector_negative_mean": 0.11757936279173009,
  "keypoint_mae_px": 7.384131895682224,
  "mls_mae_mm": 1.8133565705844397,
  "mls_rmse_mm": 2.48701116011082,
  "mls_f1_3mm": 0.7530120481927711,
  "mls_f1_5mm": 0.7619047619047619,
  "study_mls_mae_mm": 0.9501973552712754,
  "study_mls_f1_3mm": 0.8235294117647058,
  "study_mls_f1_5mm": 0.8,
  "study_boundary_f1": 0.8117647058823529,
  "selection_objective": 1.3749230653104394,
  "train_spatial_loss": 5.079395944961874,
  "train_coordinate_loss": 0.020575949225783534,
  "train_mls_loss": 0.41493718022385023,
  "train_threshold_loss": 0.17215379711509565,
  "train_selector_loss": 0.32145980746216835
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.8186 | 38.4340 | 3.5840 | 0.0000 | 148.11 | 0.3870 | 0.4008 | 4.76 |
| 2 | 4.1985 | 10.7654 | 3.5840 | 0.0000 | 37.40 | 0.6068 | 0.6267 | 4.54 |
| 3 | 3.7952 | 7.7038 | 3.5840 | 0.0000 | 44.43 | 0.7983 | 0.7324 | 4.55 |
| 4 | 3.4720 | 2.8029 | 3.5840 | 0.0000 | 22.40 | 0.7766 | 0.7792 | 4.55 |
| 5 | 3.0506 | 2.1036 | 3.5840 | 0.0000 | 10.18 | 0.8288 | 0.7768 | 4.55 |
| 6 | 2.6578 | 2.1412 | 3.5840 | 0.0000 | 9.58 | 0.8370 | 0.7329 | 4.54 |
| 7 | 2.4462 | 1.7441 | 3.5840 | 0.0000 | 8.72 | 0.8449 | 0.7788 | 4.54 |
| 8 | 2.3184 | 1.8931 | 1.7015 | 0.6635 | 9.23 | 0.8866 | 0.8098 | 4.54 |
| 9 | 2.1767 | 1.7255 | 2.1246 | 0.4676 | 8.53 | 0.8730 | 0.7830 | 4.54 |
| 10 | 2.0735 | 1.8799 | 1.1581 | 0.8199 | 7.71 | 0.9052 | 0.8135 | 4.55 |
| 11 | 2.0036 | 1.9090 | 1.5760 | 0.7037 | 9.41 | 0.8982 | 0.7792 | 4.55 |
| 12 | 1.9757 | 1.7453 | 1.5531 | 0.6889 | 7.86 | 0.8916 | 0.7810 | 4.54 |
| 13 | 1.8998 | 1.8058 | 1.0502 | 0.8247 | 8.06 | 0.8981 | 0.8215 | 4.54 |
| 14 | 1.8839 | 1.7425 | 0.9837 | 0.8000 | 7.79 | 0.9010 | 0.7971 | 4.55 |
| 15 | 1.8283 | 1.8265 | 1.0160 | 0.8106 | 8.38 | 0.8898 | 0.7977 | 4.54 |
| 16 | 1.8267 | 1.7741 | 1.0896 | 0.7407 | 7.35 | 0.9050 | 0.8162 | 4.55 |
| 17 | 1.7225 | 1.8134 | 0.9502 | 0.8118 | 7.38 | 0.9035 | 0.8033 | 4.54 |
| 18 | 1.7032 | 1.7873 | 1.0451 | 0.8082 | 7.42 | 0.8893 | 0.7904 | 4.54 |
| 19 | 1.6874 | 1.6756 | 1.1533 | 0.7267 | 7.34 | 0.8931 | 0.7897 | 4.54 |
| 20 | 1.6787 | 1.7165 | 1.2438 | 0.7591 | 7.35 | 0.8905 | 0.7923 | 4.54 |
| 21 | 1.6141 | 1.7356 | 0.9413 | 0.8000 | 7.24 | 0.8951 | 0.7978 | 4.54 |
| 22 | 1.6068 | 1.7298 | 1.2170 | 0.7514 | 7.23 | 0.8922 | 0.7850 | 4.54 |
| 23 | 1.6370 | 1.7436 | 1.1009 | 0.7718 | 7.29 | 0.8947 | 0.7910 | 4.55 |
