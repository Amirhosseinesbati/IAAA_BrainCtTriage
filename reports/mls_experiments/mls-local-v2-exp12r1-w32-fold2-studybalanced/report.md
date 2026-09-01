# MLS experiment: mls-local-v2-exp12r1-w32-fold2-studybalanced

- Status: `completed`
- Updated UTC: `2026-08-28T11:30:19.313586+00:00`
- MLflow run id: `4af89dc814d8439590e69f886c30909b`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "study_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 17.0,
  "train_loss": 1.7272682306927432,
  "lr": 1.8825509907063327e-05,
  "peak_vram_gb": 4.54531717300415,
  "val_loss": 1.706146438091771,
  "selector_auc": 0.912474974974975,
  "selector_f1": 0.8242612752721618,
  "selector_accuracy": 0.8310911808669657,
  "selector_peak_auc": 0.7518860388745895,
  "selector_positive_mean": 0.7134629932369243,
  "selector_negative_mean": 0.17009583215743382,
  "keypoint_mae_px": 9.105179388243872,
  "mls_mae_mm": 2.3615311753933317,
  "mls_rmse_mm": 3.8670286655887445,
  "mls_f1_3mm": 0.8666666666666667,
  "mls_f1_5mm": 0.8383233532934131,
  "study_mls_mae_mm": 1.3169805339690466,
  "study_mls_f1_3mm": 0.9387755102040817,
  "study_mls_f1_5mm": 0.9545454545454546,
  "study_boundary_f1": 0.9466604823747682,
  "selection_objective": 1.4674220817320227,
  "train_spatial_loss": 5.0770544971920035,
  "train_coordinate_loss": 0.02078953290964662,
  "train_mls_loss": 0.47517490109339816,
  "train_threshold_loss": 0.1779842217361306,
  "train_selector_loss": 0.3110176947225898
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6888 | 128.9886 | 4.8816 | 0.0000 | 254.88 | 0.4830 | 0.3982 | 4.76 |
| 2 | 4.2860 | 4.2110 | 4.8816 | 0.0000 | 14.93 | 0.2375 | 0.2851 | 4.55 |
| 3 | 3.4260 | 3.3409 | 4.8816 | 0.0000 | 12.78 | 0.4518 | 0.4171 | 4.55 |
| 4 | 2.8703 | 2.8839 | 4.8816 | 0.0000 | 12.94 | 0.7837 | 0.6829 | 4.55 |
| 5 | 2.5707 | 2.5706 | 4.8816 | 0.0000 | 10.73 | 0.7940 | 0.6594 | 4.55 |
| 6 | 2.4482 | 2.4252 | 4.8816 | 0.0000 | 10.48 | 0.8104 | 0.7039 | 4.55 |
| 7 | 2.3685 | 2.2916 | 4.8816 | 0.0000 | 9.38 | 0.8273 | 0.6799 | 4.55 |
| 8 | 2.2247 | 2.1178 | 4.8816 | 0.0000 | 9.46 | 0.8553 | 0.7244 | 4.55 |
| 9 | 2.1345 | 2.6680 | 2.2994 | 0.7638 | 9.73 | 0.8773 | 0.7356 | 4.55 |
| 10 | 2.0355 | 2.3452 | 1.6889 | 0.8980 | 9.64 | 0.8933 | 0.7501 | 4.55 |
| 11 | 2.0076 | 2.1639 | 2.3256 | 0.7800 | 9.00 | 0.8944 | 0.7134 | 4.55 |
| 12 | 1.9262 | 2.2872 | 1.5551 | 0.9267 | 8.95 | 0.9167 | 0.7875 | 4.55 |
| 13 | 1.8859 | 2.3336 | 1.3549 | 0.9573 | 9.47 | 0.9187 | 0.7968 | 4.55 |
| 14 | 1.8445 | 2.3032 | 2.3595 | 0.7901 | 9.91 | 0.8943 | 0.7320 | 4.55 |
| 15 | 1.8146 | 2.3206 | 1.9442 | 0.8325 | 9.57 | 0.8798 | 0.7114 | 4.55 |
| 16 | 1.7685 | 2.3834 | 1.7181 | 0.8858 | 8.91 | 0.9199 | 0.7486 | 4.55 |
| 17 | 1.7273 | 2.3615 | 1.3170 | 0.9467 | 9.11 | 0.9125 | 0.7519 | 4.55 |
| 18 | 1.6980 | 2.3126 | 1.7355 | 0.8858 | 8.84 | 0.9029 | 0.7372 | 4.55 |
| 19 | 1.6496 | 2.3243 | 1.6649 | 0.8992 | 9.11 | 0.8871 | 0.7301 | 4.55 |
| 20 | 1.6408 | 2.3167 | 1.4932 | 0.9357 | 8.85 | 0.9014 | 0.7433 | 4.55 |
| 21 | 1.6391 | 2.3547 | 1.5312 | 0.9230 | 8.94 | 0.8908 | 0.7231 | 4.55 |
| 22 | 1.5983 | 2.3837 | 1.5248 | 0.9324 | 8.90 | 0.8956 | 0.7281 | 4.55 |
| 23 | 1.6186 | 2.4145 | 1.5106 | 0.9324 | 8.85 | 0.8937 | 0.7275 | 4.55 |
