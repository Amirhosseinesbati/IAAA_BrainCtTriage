# MLS experiment: mls-local-v2-exp01

- Status: `completed`
- Updated UTC: `2026-08-27T02:42:34.726047+00:00`
- MLflow run id: `8fee771402924977bcfdc6e028c6625e`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w18", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 6, "val_split": 0.2, "top_k_slices": 3, "aggregation": "p90", "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 25.0,
  "train_loss": 1.4385807919925173,
  "lr": 8.225609429353187e-06,
  "peak_vram_gb": 3.811601161956787,
  "val_loss": 2.16796354608467,
  "selector_auc": 0.8832644696833603,
  "selector_f1": 0.8167539267015707,
  "selector_accuracy": 0.8200514138817481,
  "selector_positive_mean": 0.7283023769687695,
  "selector_negative_mean": 0.10578333557566598,
  "keypoint_mae_px": 8.926271675412654,
  "mls_mae_mm": 2.037052567367997,
  "mls_rmse_mm": 3.0933062872149177,
  "mls_f1_3mm": 0.7886710239651417,
  "mls_f1_5mm": 0.7737704918032787,
  "selection_objective": 2.5202802442814956,
  "train_spatial_loss": 5.169523205302507,
  "train_coordinate_loss": 0.016932983177651066,
  "train_mls_loss": 0.3055944590362958,
  "train_threshold_loss": 0.13091250400981105,
  "train_selector_loss": 0.04824363667840043
}
```

## Epoch history

| epoch | train loss | val MLS MAE | kp MAE | selector AUC | selector F1 | F1@3 | F1@5 | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.8910 | 57.8674 | 153.35 | 0.6213 | 0.0000 | 0.5796 | 0.4455 | 4.54 |
| 2 | 4.5259 | 112.1095 | 183.00 | 0.8101 | 0.0000 | 0.6910 | 0.5136 | 3.81 |
| 3 | 3.9434 | 54.7006 | 57.16 | 0.7222 | 0.3178 | 0.6856 | 0.5119 | 3.81 |
| 4 | 3.7612 | 4.9702 | 15.47 | 0.7729 | 0.7312 | 0.6293 | 0.4652 | 3.81 |
| 5 | 3.4287 | 5.2504 | 15.75 | 0.8378 | 0.7882 | 0.6667 | 0.5538 | 3.81 |
| 6 | 3.1891 | 4.1320 | 13.68 | 0.7965 | 0.7588 | 0.7009 | 0.6563 | 3.81 |
| 7 | 2.9029 | 3.0562 | 11.84 | 0.8535 | 0.8037 | 0.7456 | 0.7342 | 3.81 |
| 8 | 2.6380 | 2.6401 | 10.98 | 0.8679 | 0.7964 | 0.7678 | 0.7718 | 3.81 |
| 9 | 2.3701 | 2.8854 | 11.18 | 0.8706 | 0.7995 | 0.7450 | 0.7452 | 3.81 |
| 10 | 2.1951 | 2.5817 | 10.08 | 0.9164 | 0.8360 | 0.7652 | 0.7668 | 3.81 |
| 11 | 2.0823 | 2.6235 | 9.78 | 0.9167 | 0.8191 | 0.7639 | 0.7562 | 3.81 |
| 12 | 2.0510 | 2.2536 | 9.51 | 0.9064 | 0.8227 | 0.7857 | 0.7821 | 3.81 |
| 13 | 1.9132 | 2.4730 | 9.88 | 0.9164 | 0.8529 | 0.7516 | 0.7419 | 3.81 |
| 14 | 1.9259 | 2.3484 | 9.15 | 0.9026 | 0.8401 | 0.7609 | 0.7683 | 3.81 |
| 15 | 1.8331 | 2.2363 | 9.28 | 0.8794 | 0.6533 | 0.7868 | 0.7541 | 3.81 |
| 16 | 1.8571 | 2.3254 | 9.32 | 0.8930 | 0.8217 | 0.7682 | 0.7702 | 3.81 |
| 17 | 1.7126 | 2.1986 | 9.45 | 0.9090 | 0.8098 | 0.7726 | 0.7857 | 3.81 |
| 18 | 1.6924 | 2.2184 | 9.41 | 0.8908 | 0.7050 | 0.7592 | 0.7524 | 3.81 |
| 19 | 1.6747 | 2.2150 | 9.30 | 0.8944 | 0.8247 | 0.7725 | 0.7658 | 3.81 |
| 20 | 1.5712 | 2.3365 | 9.14 | 0.8996 | 0.8258 | 0.7429 | 0.7372 | 3.81 |
| 21 | 1.5537 | 2.1715 | 8.97 | 0.8940 | 0.8219 | 0.7800 | 0.7548 | 3.81 |
| 22 | 1.5293 | 2.1572 | 9.72 | 0.8570 | 0.6697 | 0.7565 | 0.7584 | 3.81 |
| 23 | 1.4710 | 2.6739 | 9.34 | 0.8866 | 0.8046 | 0.7564 | 0.7557 | 3.81 |
| 24 | 1.4783 | 2.5395 | 9.20 | 0.8861 | 0.7493 | 0.7559 | 0.7500 | 3.81 |
| 25 | 1.4386 | 2.0371 | 8.93 | 0.8833 | 0.8168 | 0.7887 | 0.7738 | 3.81 |
| 26 | 1.4275 | 2.2038 | 8.64 | 0.8906 | 0.8221 | 0.7888 | 0.7632 | 3.81 |
| 27 | 1.4271 | 2.4666 | 9.09 | 0.8824 | 0.8036 | 0.7903 | 0.7351 | 3.81 |
| 28 | 1.3991 | 2.1688 | 8.85 | 0.8823 | 0.8163 | 0.8000 | 0.7475 | 3.81 |
| 29 | 1.3904 | 2.2364 | 8.85 | 0.8807 | 0.8188 | 0.8122 | 0.7525 | 3.81 |
| 30 | 1.3852 | 2.2293 | 8.82 | 0.8782 | 0.8135 | 0.8043 | 0.7484 | 3.81 |

## End-to-end fold-0 evaluation

- Strict CUDA-only evaluation covered all `70/70` studies and `1723` slices with zero failures.
- Pre-registered `threshold=0.5, top-3, p90`: MAE `2.1349 mm`, RMSE `3.3029 mm`, bias `+0.5790 mm`, combined Macro-F1 `0.4921`.
- Locked candidate `relative_component, ratio=0.3, gate=0.5, q=0.75`: MAE `1.9094 mm`, RMSE `2.9627 mm`, bias `+0.2175 mm`, combined Macro-F1 `0.5896`.
- The locked candidate was selected on fold 0 and must be tested unchanged on another OOF fold before it is treated as an unbiased estimate.

## Error decomposition and decision

- Slice selector AUC against annotated target slices: `0.9099`.
- At least one annotated target was retrieved in top-1 for `100%` of annotated fold-0 studies.
- Oracle annotated-target pooling reached MAE `1.3022 mm` with p90, showing that keypoint measurement is substantially better than the original top-3 pooling result.
- The principal remaining fold-0 error is target-span reconstruction/pooling, not failure to find the target region.
- Experiment 1 is accepted as the new MLS baseline. The old broken checkpoint was about `6.66 mm` MAE end-to-end; this run reduces it to `1.91 mm` with the locked candidate.

Detailed artifacts are in `end_to_end/metrics.json`, `end_to_end/decomposition.json`, and `end_to_end/postprocessing_search.json`.
