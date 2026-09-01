# MLS experiment: mls-local-v2-exp05-w32-fold0-peakaware

- Status: `completed`
- Updated UTC: `2026-08-27T15:15:05.965402+00:00`
- MLflow run id: `532d62f07a84421681c8f199ccba462d`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.5, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.5, "negative_value_mm": 0.1, "aggregation": "quantile", "selector_relative_ratio": 0.5, "aggregation_quantile": 0.65, "aggregation_probability_weighted": false, "anchor_window_radius": 3, "min_active_slices": 2, "heatmap_guard_ratio": 0.5, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 13.0,
  "train_loss": 1.9540347304890644,
  "lr": 6.980398830195785e-05,
  "peak_vram_gb": 4.54556131362915,
  "val_loss": 1.812776207852249,
  "selector_auc": 0.9016347631689254,
  "selector_f1": 0.8062418725617685,
  "selector_accuracy": 0.8084832904884319,
  "selector_peak_auc": 0.8047131185504938,
  "selector_positive_mean": 0.5971395235327283,
  "selector_negative_mean": 0.15985606075068856,
  "keypoint_mae_px": 9.02144073231864,
  "mls_mae_mm": 2.19994576796646,
  "mls_rmse_mm": 5.565842141728602,
  "mls_f1_3mm": 0.7799564270152506,
  "mls_f1_5mm": 0.7884615384615384,
  "study_mls_mae_mm": 1.0536743410570284,
  "study_mls_f1_3mm": 0.8421052631578947,
  "study_mls_f1_5mm": 0.8205128205128205,
  "study_boundary_f1": 0.8313090418353576,
  "selection_objective": 1.4402388758018505,
  "train_spatial_loss": 5.148793172924444,
  "train_coordinate_loss": 0.025536144684437257,
  "train_mls_loss": 0.7605796478032778,
  "train_threshold_loss": 0.2263484813778819,
  "train_selector_loss": 0.44128860874779785
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7570 | 52.3530 | 4.0333 | 0.0000 | 172.90 | 0.4163 | 0.3778 | 4.76 |
| 2 | 4.4462 | 4.6610 | 4.0333 | 0.0000 | 14.90 | 0.3050 | 0.3185 | 4.55 |
| 3 | 3.6304 | 5.1080 | 4.0333 | 0.0000 | 15.59 | 0.1840 | 0.2949 | 4.55 |
| 4 | 2.9917 | 2.6430 | 4.0333 | 0.0000 | 13.56 | 0.1838 | 0.3339 | 4.55 |
| 5 | 2.6707 | 2.6659 | 4.0333 | 0.0000 | 11.99 | 0.6665 | 0.7102 | 4.55 |
| 6 | 2.4082 | 2.3877 | 4.0333 | 0.0000 | 10.40 | 0.7525 | 0.7460 | 4.55 |
| 7 | 2.3156 | 2.2193 | 4.0333 | 0.0000 | 10.00 | 0.8283 | 0.7536 | 4.55 |
| 8 | 2.1868 | 2.3012 | 1.2045 | 0.8440 | 10.02 | 0.8556 | 0.7720 | 4.55 |
| 9 | 2.2385 | 2.0509 | 1.3041 | 0.8177 | 9.09 | 0.8635 | 0.7607 | 4.55 |
| 10 | 2.0426 | 2.0757 | 1.3583 | 0.8068 | 8.60 | 0.9085 | 0.7830 | 4.55 |
| 11 | 1.9978 | 2.3851 | 1.5143 | 0.7684 | 9.62 | 0.9134 | 0.7781 | 4.55 |
| 12 | 1.9386 | 1.9785 | 1.1729 | 0.7994 | 9.12 | 0.9115 | 0.7749 | 4.55 |
| 13 | 1.9540 | 2.1999 | 1.0537 | 0.8313 | 9.02 | 0.9016 | 0.8047 | 4.55 |
| 14 | 1.8614 | 2.1429 | 1.2822 | 0.7602 | 8.83 | 0.9013 | 0.7802 | 4.55 |
| 15 | 1.9282 | 2.0588 | 1.1494 | 0.7800 | 8.74 | 0.9199 | 0.8020 | 4.55 |
| 16 | 1.8173 | 2.1206 | 1.0425 | 0.8021 | 8.73 | 0.9107 | 0.7928 | 4.55 |
| 17 | 1.7399 | 2.4393 | 2.5613 | 0.4739 | 8.85 | 0.8768 | 0.7453 | 4.55 |
| 18 | 1.7612 | 2.1164 | 1.2069 | 0.7590 | 8.52 | 0.9044 | 0.8010 | 4.55 |
| 19 | 1.7125 | 2.4293 | 1.5354 | 0.7369 | 8.97 | 0.9024 | 0.7617 | 4.55 |
| 20 | 1.6815 | 2.3261 | 1.3426 | 0.7602 | 9.26 | 0.9029 | 0.7784 | 4.55 |
| 21 | 1.6642 | 2.2978 | 1.1879 | 0.7650 | 8.74 | 0.9082 | 0.8049 | 4.55 |
