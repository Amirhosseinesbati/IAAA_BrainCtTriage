# MLS experiment: mls-vast-exp19-w32-fold0-dual-selector-replication

- Status: `completed`
- Updated UTC: `2026-09-02T12:03:40.561099+00:00`
- MLflow run id: `5383a78d31bf4a79a5bf6aff3c086e8c`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "heatmap_sigma_anneal_end": null, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_head_mode": "dual", "selector_peak_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.0, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "slice_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "resume_checkpoint": null, "lr_scheduler_patience": 5, "use_amp": false, "training_determinism": "strict", "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 21.0,
  "train_loss": 1.5821160807406836,
  "train_heatmap_sigma": 3.0,
  "lr": 2.221359710692961e-06,
  "peak_vram_gb": 4.647613048553467,
  "val_loss": 1.8594033274566755,
  "selector_auc": 0.8981483329673914,
  "selector_f1": 0.8175,
  "selector_accuracy": 0.8123393316195373,
  "selector_peak_auc": 0.7805546806973341,
  "selector_positive_mean": 0.7260235197678421,
  "selector_negative_mean": 0.1705881807257236,
  "peak_selector_positive_mean": 0.4154080335546347,
  "peak_selector_negative_mean": 0.0752654456105005,
  "keypoint_mae_px": 8.85161790749617,
  "mls_mae_mm": 2.141858175774094,
  "mls_rmse_mm": 3.4524036883049085,
  "mls_f1_3mm": 0.7685393258426966,
  "mls_f1_5mm": 0.7718120805369127,
  "study_mls_mae_mm": 1.1181203338716716,
  "study_mls_f1_3mm": 0.8214285714285714,
  "study_mls_f1_5mm": 0.8333333333333334,
  "study_boundary_f1": 0.8273809523809523,
  "selection_objective": 1.5142842626260713,
  "train_spatial_loss": 5.059861554235716,
  "train_coordinate_loss": 0.016486553817889747,
  "train_mls_loss": 0.20392041749365386,
  "train_threshold_loss": 0.13169158002851025,
  "train_selector_loss": 0.24475815048004035,
  "train_selector_presence_loss": 0.1562372390938279,
  "train_selector_peak_loss": 0.3332790626202626
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7445 | 70.9624 | 4.0333 | 0.0000 | 145.22 | 0.7683 | 0.2496 | 4.65 |
| 2 | 4.2940 | 26.5153 | 4.0333 | 0.0000 | 88.40 | 0.7868 | 0.2117 | 4.65 |
| 3 | 3.6612 | 4.6224 | 4.0333 | 0.0000 | 17.33 | 0.7644 | 0.3421 | 4.65 |
| 4 | 3.3283 | 3.5798 | 4.0333 | 0.0000 | 15.03 | 0.8096 | 0.3730 | 4.65 |
| 5 | 2.7452 | 2.7372 | 4.0333 | 0.0000 | 11.02 | 0.7748 | 0.6701 | 4.65 |
| 6 | 2.4937 | 2.5106 | 4.0333 | 0.0000 | 9.64 | 0.8169 | 0.7393 | 4.65 |
| 7 | 2.2567 | 2.0742 | 4.0333 | 0.0000 | 9.08 | 0.8857 | 0.7740 | 4.65 |
| 8 | 2.1961 | 2.3656 | 1.9277 | 0.7265 | 9.45 | 0.8857 | 0.7904 | 4.65 |
| 9 | 2.0607 | 2.2057 | 1.2908 | 0.8138 | 10.20 | 0.8778 | 0.7719 | 4.65 |
| 10 | 2.0314 | 2.3131 | 1.6209 | 0.8108 | 9.50 | 0.8941 | 0.7842 | 4.65 |
| 11 | 1.9419 | 2.1349 | 1.4877 | 0.7596 | 9.58 | 0.8952 | 0.7741 | 4.65 |
| 12 | 1.9065 | 2.3303 | 1.4990 | 0.7756 | 9.23 | 0.8963 | 0.7794 | 4.65 |
| 13 | 1.8694 | 2.1999 | 1.4722 | 0.7525 | 8.93 | 0.9118 | 0.7696 | 4.65 |
| 14 | 1.8576 | 2.1568 | 1.2308 | 0.8158 | 8.79 | 0.8962 | 0.7598 | 4.65 |
| 15 | 1.7641 | 2.2452 | 1.3668 | 0.8000 | 8.89 | 0.8943 | 0.7748 | 4.65 |
| 16 | 1.7437 | 2.4055 | 1.2668 | 0.8132 | 8.95 | 0.8932 | 0.7805 | 4.65 |
| 17 | 1.6646 | 2.0144 | 1.1525 | 0.8054 | 8.69 | 0.8971 | 0.7649 | 4.65 |
| 18 | 1.6760 | 2.1264 | 1.4020 | 0.8027 | 9.01 | 0.8698 | 0.7567 | 4.65 |
| 19 | 1.6389 | 2.2375 | 1.1433 | 0.8421 | 8.72 | 0.8859 | 0.7663 | 4.65 |
| 20 | 1.6173 | 2.2127 | 1.1610 | 0.8496 | 8.96 | 0.8949 | 0.7755 | 4.65 |
| 21 | 1.5821 | 2.1419 | 1.1181 | 0.8274 | 8.85 | 0.8981 | 0.7806 | 4.65 |
| 22 | 1.5793 | 2.1141 | 1.1911 | 0.8393 | 8.83 | 0.8913 | 0.7649 | 4.65 |
| 23 | 1.5917 | 2.1788 | 1.2854 | 0.7889 | 8.91 | 0.8949 | 0.7714 | 4.65 |
