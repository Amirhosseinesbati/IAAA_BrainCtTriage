# MLS experiment: mls-vast-exp20-w32-fold2-dual-selector-thirdfold-replication

- Status: `completed`
- Updated UTC: `2026-09-02T14:03:25.559298+00:00`
- MLflow run id: `aa4d88acea4246a8a7e5c27a0a33a6c6`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "heatmap_sigma_anneal_end": null, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_head_mode": "dual", "selector_peak_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.0, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "slice_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "resume_checkpoint": null, "lr_scheduler_patience": 5, "use_amp": false, "training_determinism": "strict", "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 11.0,
  "train_loss": 1.8792508441961469,
  "train_heatmap_sigma": 3.0,
  "lr": 6.112604669781572e-05,
  "peak_vram_gb": 4.647613048553467,
  "val_loss": 1.767128704345104,
  "selector_auc": 0.8857428857428857,
  "selector_f1": 0.7585034013605442,
  "selector_accuracy": 0.7877428998505231,
  "selector_peak_auc": 0.7569894381823024,
  "selector_positive_mean": 0.6102303826717196,
  "selector_negative_mean": 0.16029806987249426,
  "peak_selector_positive_mean": 0.35354259688205814,
  "peak_selector_negative_mean": 0.07106701291888298,
  "keypoint_mae_px": 9.30901925341861,
  "mls_mae_mm": 2.3255419860574396,
  "mls_rmse_mm": 3.4971238384487067,
  "mls_f1_3mm": 0.8787185354691075,
  "mls_f1_5mm": 0.7940298507462686,
  "study_mls_mae_mm": 1.3310733661945189,
  "study_mls_f1_3mm": 0.9411764705882353,
  "study_mls_f1_5mm": 0.9777777777777777,
  "study_boundary_f1": 0.9594771241830065,
  "selection_objective": 1.469247674957063,
  "train_spatial_loss": 5.021309800088723,
  "train_coordinate_loss": 0.022228763828018507,
  "train_mls_loss": 0.6347355911134731,
  "train_threshold_loss": 0.3059291367043716,
  "train_selector_loss": 0.42353220299865596,
  "train_selector_presence_loss": 0.40700281883852946,
  "train_selector_peak_loss": 0.4400615873969893
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7054 | 19.0467 | 4.8816 | 0.0000 | 38.77 | 0.6297 | 0.3209 | 4.65 |
| 2 | 3.7992 | 12.2762 | 4.8816 | 0.0000 | 51.79 | 0.7116 | 0.2943 | 4.65 |
| 3 | 3.3367 | 3.8589 | 4.8816 | 0.0000 | 15.42 | 0.8030 | 0.3284 | 4.65 |
| 4 | 2.9384 | 3.1064 | 4.8816 | 0.0000 | 12.40 | 0.8020 | 0.4248 | 4.65 |
| 5 | 2.5138 | 2.6434 | 4.8816 | 0.0000 | 11.13 | 0.8338 | 0.7169 | 4.65 |
| 6 | 2.3560 | 2.7774 | 4.8816 | 0.0000 | 9.81 | 0.8546 | 0.7406 | 4.65 |
| 7 | 2.2007 | 2.4777 | 2.3035 | 0.7737 | 9.18 | 0.8760 | 0.7248 | 4.65 |
| 8 | 2.1324 | 2.1879 | 1.6950 | 0.8786 | 9.20 | 0.8674 | 0.7542 | 4.65 |
| 9 | 2.0104 | 2.3940 | 2.0373 | 0.8794 | 8.92 | 0.8885 | 0.7591 | 4.65 |
| 10 | 1.9421 | 2.3007 | 1.3750 | 0.9357 | 9.62 | 0.8971 | 0.7386 | 4.65 |
| 11 | 1.8793 | 2.3255 | 1.3311 | 0.9595 | 9.31 | 0.8857 | 0.7570 | 4.65 |
| 12 | 1.8720 | 2.3294 | 1.3810 | 0.9489 | 9.06 | 0.9069 | 0.7755 | 4.65 |
| 13 | 1.8302 | 2.4765 | 1.5964 | 0.8741 | 8.66 | 0.9125 | 0.7810 | 4.65 |
| 14 | 1.7847 | 2.7664 | 2.0887 | 0.8402 | 9.70 | 0.8922 | 0.7393 | 4.65 |
| 15 | 1.7299 | 2.5856 | 1.6528 | 0.8918 | 8.27 | 0.9081 | 0.7716 | 4.65 |
| 16 | 1.7357 | 2.3603 | 1.9201 | 0.8303 | 8.87 | 0.9049 | 0.7371 | 4.65 |
| 17 | 1.6740 | 2.5612 | 1.9872 | 0.8277 | 8.49 | 0.8911 | 0.7577 | 4.65 |
| 18 | 1.6193 | 2.4700 | 1.7109 | 0.8676 | 8.73 | 0.9020 | 0.7569 | 4.65 |
| 19 | 1.5981 | 2.5896 | 1.5878 | 0.8808 | 8.63 | 0.8954 | 0.7533 | 4.65 |
| 20 | 1.5605 | 2.4978 | 1.3828 | 0.9373 | 8.37 | 0.9018 | 0.7556 | 4.65 |
| 21 | 1.5678 | 2.4969 | 1.6011 | 0.9028 | 8.41 | 0.8999 | 0.7603 | 4.65 |
| 22 | 1.5804 | 2.5344 | 1.5340 | 0.9129 | 8.50 | 0.8934 | 0.7543 | 4.65 |
| 23 | 1.6069 | 2.6147 | 1.4968 | 0.9129 | 8.71 | 0.8974 | 0.7628 | 4.65 |
