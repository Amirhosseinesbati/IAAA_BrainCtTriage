# MLS experiment: mls-vast-exp14r2-w32-fold2-hybridsoft-repro

- Status: `completed`
- Updated UTC: `2026-09-01T18:44:21.108670+00:00`
- MLflow run id: `efdfb96e8e2740918836991ac0fff2bf`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "heatmap_sigma_anneal_end": null, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 23, "batch_size": 5, "val_split": 0.2, "sampling_mode": "slice_class_balanced", "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 23, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "resume_checkpoint": null, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 12.0,
  "train_loss": 1.9152617098806595,
  "train_heatmap_sigma": 3.0,
  "lr": 5.373650467932122e-05,
  "peak_vram_gb": 4.54604959487915,
  "val_loss": 1.732770896544541,
  "selector_auc": 0.9071929071929072,
  "selector_f1": 0.799373040752351,
  "selector_accuracy": 0.8086696562032885,
  "selector_peak_auc": 0.7838821336646844,
  "selector_positive_mean": 0.6313204672690984,
  "selector_negative_mean": 0.19422999715191377,
  "keypoint_mae_px": 10.693657179733178,
  "mls_mae_mm": 2.55192457109287,
  "mls_rmse_mm": 4.258971725870656,
  "mls_f1_3mm": 0.859122401847575,
  "mls_f1_5mm": 0.8259587020648967,
  "study_mls_mae_mm": 1.5350487984828094,
  "study_mls_f1_3mm": 0.9387755102040817,
  "study_mls_f1_5mm": 0.9333333333333333,
  "study_boundary_f1": 0.9360544217687075,
  "selection_objective": 1.7093435013489409,
  "train_spatial_loss": 5.12060519094789,
  "train_coordinate_loss": 0.02430097279068559,
  "train_mls_loss": 0.6229198789900964,
  "train_threshold_loss": 0.25308263296233197,
  "train_selector_loss": 0.4419216874384753
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6438 | 8.4519 | 4.8816 | 0.0000 | 20.93 | 0.2018 | 0.2306 | 4.57 |
| 2 | 3.9397 | 6.3459 | 4.8816 | 0.0000 | 27.44 | 0.1910 | 0.2744 | 4.55 |
| 3 | 3.4010 | 3.7361 | 4.8816 | 0.0000 | 17.00 | 0.6853 | 0.6778 | 4.55 |
| 4 | 2.8531 | 3.0177 | 4.8816 | 0.0000 | 12.46 | 0.8144 | 0.7380 | 4.55 |
| 5 | 2.6023 | 2.4900 | 4.8816 | 0.0000 | 11.22 | 0.8012 | 0.7060 | 4.55 |
| 6 | 2.4090 | 2.7068 | 4.8816 | 0.0000 | 11.03 | 0.8437 | 0.7324 | 4.55 |
| 7 | 2.3251 | 2.7399 | 4.8816 | 0.0000 | 10.08 | 0.8370 | 0.7277 | 4.55 |
| 8 | 2.1866 | 2.3582 | 4.5610 | 0.1834 | 10.67 | 0.8527 | 0.7341 | 4.55 |
| 9 | 2.1435 | 2.4708 | 1.5677 | 0.8880 | 10.22 | 0.8680 | 0.7296 | 4.55 |
| 10 | 2.0283 | 2.2503 | 1.7780 | 0.8880 | 10.50 | 0.8899 | 0.7376 | 4.55 |
| 11 | 1.9755 | 2.6126 | 2.1195 | 0.8858 | 10.10 | 0.8968 | 0.7700 | 4.55 |
| 12 | 1.9153 | 2.5519 | 1.5350 | 0.9361 | 10.69 | 0.9072 | 0.7839 | 4.55 |
| 13 | 1.9296 | 2.3254 | 1.8192 | 0.8402 | 9.73 | 0.8722 | 0.7741 | 4.55 |
| 14 | 1.8840 | 2.5056 | 1.8647 | 0.8754 | 10.01 | 0.9017 | 0.7629 | 4.55 |
| 15 | 1.8082 | 2.4962 | 1.8880 | 0.8402 | 9.95 | 0.9080 | 0.7580 | 4.55 |
| 16 | 1.7743 | 2.2417 | 1.6925 | 0.8908 | 9.33 | 0.9154 | 0.7562 | 4.55 |
| 17 | 1.7840 | 2.3503 | 1.7627 | 0.8521 | 9.61 | 0.8947 | 0.7458 | 4.55 |
| 18 | 1.7396 | 2.2625 | 1.9832 | 0.8255 | 9.29 | 0.8892 | 0.7364 | 4.55 |
| 19 | 1.6635 | 2.3993 | 1.7563 | 0.8583 | 8.90 | 0.9020 | 0.7511 | 4.55 |
| 20 | 1.6686 | 2.2221 | 1.7370 | 0.8132 | 8.85 | 0.8987 | 0.7434 | 4.55 |
| 21 | 1.6445 | 2.2033 | 1.7199 | 0.8490 | 8.73 | 0.8941 | 0.7390 | 4.55 |
| 22 | 1.6166 | 2.2242 | 1.7337 | 0.8636 | 8.81 | 0.8983 | 0.7468 | 4.55 |
| 23 | 1.6040 | 2.2592 | 1.7287 | 0.8521 | 8.87 | 0.8910 | 0.7421 | 4.55 |
