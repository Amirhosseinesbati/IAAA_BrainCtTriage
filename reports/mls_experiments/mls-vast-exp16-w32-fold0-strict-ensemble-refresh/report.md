# MLS experiment: mls-vast-exp16-w32-fold0-strict-ensemble-refresh

- Status: `completed`
- Updated UTC: `2026-09-02T02:54:53.765773+00:00`
- MLflow run id: `a2478b8410d74de2b2806ef08d79051d`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "heatmap_sigma_anneal_end": null, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "slice_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "resume_checkpoint": null, "lr_scheduler_patience": 5, "use_amp": false, "training_determinism": "strict", "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 11.0,
  "train_loss": 1.990374464204263,
  "train_heatmap_sigma": 3.0,
  "lr": 6.112604669781572e-05,
  "peak_vram_gb": 4.64663553237915,
  "val_loss": 1.8472591405018017,
  "selector_auc": 0.9075297577463288,
  "selector_f1": 0.8281829419035847,
  "selector_accuracy": 0.8213367609254498,
  "selector_peak_auc": 0.8063063897586009,
  "selector_positive_mean": 0.6659203017409228,
  "selector_negative_mean": 0.21907762104661024,
  "keypoint_mae_px": 9.028265312656183,
  "mls_mae_mm": 2.158423070701817,
  "mls_rmse_mm": 3.1497344705705737,
  "mls_f1_3mm": 0.7702407002188184,
  "mls_f1_5mm": 0.7656765676567657,
  "study_mls_mae_mm": 1.1336633000629293,
  "study_mls_f1_3mm": 0.8852459016393442,
  "study_mls_f1_5mm": 0.8717948717948718,
  "study_boundary_f1": 0.8785203867171081,
  "selection_objective": 1.4228576477555488,
  "train_spatial_loss": 5.148230800346615,
  "train_coordinate_loss": 0.02684869278457305,
  "train_mls_loss": 0.7423737055149763,
  "train_threshold_loss": 0.2680607658595676,
  "train_selector_loss": 0.47749291016081563
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7739 | 42.2075 | 4.0333 | 0.0000 | 143.69 | 0.4641 | 0.3539 | 4.65 |
| 2 | 4.5712 | 13.8282 | 4.0333 | 0.0000 | 40.41 | 0.4778 | 0.4334 | 4.65 |
| 3 | 3.8502 | 5.3690 | 4.0333 | 0.0000 | 21.89 | 0.5840 | 0.5558 | 4.65 |
| 4 | 3.4651 | 2.7548 | 4.0333 | 0.0000 | 14.55 | 0.7807 | 0.7290 | 4.65 |
| 5 | 2.7485 | 2.8441 | 4.0333 | 0.0000 | 11.88 | 0.7864 | 0.6937 | 4.65 |
| 6 | 2.5313 | 2.1746 | 4.0333 | 0.0000 | 10.01 | 0.8282 | 0.7304 | 4.65 |
| 7 | 2.3950 | 2.5563 | 4.0333 | 0.0000 | 9.74 | 0.8497 | 0.7577 | 4.65 |
| 8 | 2.3268 | 2.3987 | 3.5455 | 0.1515 | 9.58 | 0.8469 | 0.7739 | 4.65 |
| 9 | 2.1407 | 2.6102 | 1.4724 | 0.8388 | 9.30 | 0.9022 | 0.7894 | 4.65 |
| 10 | 2.0912 | 2.1129 | 1.2423 | 0.8417 | 8.87 | 0.8923 | 0.8003 | 4.65 |
| 11 | 1.9904 | 2.1584 | 1.1337 | 0.8785 | 9.03 | 0.9075 | 0.8063 | 4.65 |
| 12 | 1.9445 | 2.2182 | 2.2881 | 0.6014 | 9.01 | 0.9050 | 0.7725 | 4.65 |
| 13 | 1.9085 | 2.3091 | 1.3654 | 0.8055 | 8.66 | 0.9045 | 0.7606 | 4.65 |
| 14 | 1.8764 | 1.9873 | 1.1562 | 0.8318 | 9.12 | 0.9111 | 0.7661 | 4.65 |
| 15 | 1.7826 | 2.1371 | 1.1581 | 0.8237 | 8.92 | 0.9080 | 0.8028 | 4.65 |
| 16 | 1.7693 | 2.1318 | 1.5342 | 0.7860 | 8.61 | 0.9161 | 0.7736 | 4.65 |
| 17 | 1.6787 | 2.0951 | 1.4261 | 0.7500 | 8.82 | 0.9037 | 0.7646 | 4.65 |
| 18 | 1.6872 | 2.0433 | 1.7325 | 0.7132 | 8.46 | 0.8998 | 0.7602 | 4.65 |
| 19 | 1.6537 | 2.1195 | 1.4860 | 0.7643 | 8.75 | 0.8900 | 0.7500 | 4.65 |
| 20 | 1.6352 | 2.0581 | 1.3801 | 0.7872 | 8.76 | 0.8951 | 0.7558 | 4.65 |
| 21 | 1.5919 | 2.1456 | 1.5174 | 0.7982 | 8.76 | 0.8945 | 0.7623 | 4.65 |
| 22 | 1.5881 | 2.1512 | 1.5653 | 0.8031 | 8.69 | 0.8995 | 0.7561 | 4.65 |
| 23 | 1.5995 | 2.0462 | 1.3747 | 0.8031 | 8.66 | 0.9007 | 0.7579 | 4.65 |
