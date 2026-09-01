# MLS experiment: mls-local-v2-exp06-w32-fold1-peakaware-transfer

- Status: `completed`
- Updated UTC: `2026-08-27T18:06:06.989115+00:00`
- MLflow run id: `4646e57c62e240ae8e415027ef8006e7`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 1, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.5, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.7, "aggregation_quantile": 0.75, "aggregation_probability_weighted": false, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.5, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 14.0,
  "train_loss": 1.9536093712065143,
  "lr": 6.434016163555452e-05,
  "peak_vram_gb": 4.54507303237915,
  "val_loss": 1.6371309215409888,
  "selector_auc": 0.8945077204641163,
  "selector_f1": 0.7717569786535303,
  "selector_accuracy": 0.7928464977645305,
  "selector_peak_auc": 0.801354542065366,
  "selector_positive_mean": 0.5808966906398579,
  "selector_negative_mean": 0.15670833645878543,
  "keypoint_mae_px": 7.628981314474232,
  "mls_mae_mm": 1.740796070634346,
  "mls_rmse_mm": 2.8144852457095935,
  "mls_f1_3mm": 0.7672955974842768,
  "mls_f1_5mm": 0.8137254901960784,
  "study_mls_mae_mm": 1.1420482109064487,
  "study_mls_f1_3mm": 0.75,
  "study_mls_f1_5mm": 0.7878787878787878,
  "study_boundary_f1": 0.7689393939393939,
  "selection_objective": 1.6569155627956027,
  "train_spatial_loss": 5.161060099924162,
  "train_coordinate_loss": 0.02716039635967695,
  "train_mls_loss": 0.7691891322254645,
  "train_threshold_loss": 0.2629367378118143,
  "train_selector_loss": 0.43117319490243533
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.8457 | 88.1442 | 3.5840 | 0.0000 | 246.68 | 0.2447 | 0.2431 | 4.76 |
| 2 | 4.5042 | 8.9073 | 3.5840 | 0.0000 | 43.25 | 0.2186 | 0.2217 | 4.54 |
| 3 | 3.8054 | 6.1126 | 3.5840 | 0.0000 | 14.29 | 0.3320 | 0.2975 | 4.55 |
| 4 | 3.4321 | 2.9045 | 3.5840 | 0.0000 | 13.52 | 0.1743 | 0.2359 | 4.55 |
| 5 | 2.8391 | 2.3844 | 3.5840 | 0.0000 | 9.75 | 0.7166 | 0.6401 | 4.55 |
| 6 | 2.6870 | 2.4610 | 3.5840 | 0.0000 | 11.07 | 0.7952 | 0.7337 | 4.54 |
| 7 | 2.4260 | 2.1423 | 3.5840 | 0.0000 | 9.26 | 0.8317 | 0.7638 | 4.54 |
| 8 | 2.2761 | 1.6426 | 3.5840 | 0.0000 | 8.52 | 0.8607 | 0.7685 | 4.54 |
| 9 | 2.2179 | 2.0198 | 3.5840 | 0.0000 | 9.14 | 0.8673 | 0.8214 | 4.54 |
| 10 | 2.1101 | 1.7895 | 1.9655 | 0.5961 | 8.62 | 0.8853 | 0.8168 | 4.55 |
| 11 | 2.0528 | 1.8265 | 1.5530 | 0.6359 | 8.33 | 0.8923 | 0.8024 | 4.55 |
| 12 | 2.0257 | 1.7692 | 1.7816 | 0.5616 | 7.97 | 0.8884 | 0.8064 | 4.54 |
| 13 | 1.9391 | 1.7168 | 1.6011 | 0.6030 | 7.69 | 0.8852 | 0.7908 | 4.54 |
| 14 | 1.9536 | 1.7408 | 1.1420 | 0.7689 | 7.63 | 0.8945 | 0.8014 | 4.55 |
| 15 | 1.8839 | 1.8025 | 1.3932 | 0.7185 | 7.65 | 0.8799 | 0.7871 | 4.54 |
| 16 | 1.8812 | 1.8585 | 1.7564 | 0.6030 | 7.66 | 0.8891 | 0.7970 | 4.55 |
| 17 | 1.7897 | 1.7709 | 1.2352 | 0.7228 | 7.57 | 0.8900 | 0.8049 | 4.54 |
| 18 | 1.7762 | 2.0367 | 1.9561 | 0.4927 | 7.52 | 0.8637 | 0.7887 | 4.54 |
| 19 | 1.7529 | 1.8501 | 1.3977 | 0.7528 | 7.71 | 0.8923 | 0.7940 | 4.54 |
| 20 | 1.7452 | 1.7811 | 1.4399 | 0.7185 | 7.59 | 0.8812 | 0.7941 | 4.54 |
| 21 | 1.6703 | 1.9259 | 1.2425 | 0.7326 | 7.43 | 0.8749 | 0.7839 | 4.54 |
| 22 | 1.6614 | 1.8633 | 1.3664 | 0.7556 | 7.53 | 0.8577 | 0.7850 | 4.54 |
