# MLS experiment: mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots

- Status: `completed`
- Updated UTC: `2026-08-28T01:04:50.462285+00:00`
- MLflow run id: `85a9cba212fa45a19ebc6f972106a802`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 0, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "selector_target_mode": "peak_aware_soft", "selector_peak_base": 0.75, "selector_peak_power": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 5, "selector_threshold": 0.6, "negative_value_mm": 0.1, "aggregation": "relative_component", "selector_relative_ratio": 0.3, "aggregation_quantile": 0.75, "aggregation_probability_weighted": true, "anchor_window_radius": 3, "min_active_slices": 3, "heatmap_guard_ratio": 0.0, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 30, "snapshot_start_epoch": 13, "snapshot_every_n_epochs": 2, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 10.0,
  "train_loss": 2.0235927414541544,
  "lr": 8.43120818934367e-05,
  "peak_vram_gb": 4.54556131362915,
  "val_loss": 1.7823293457428615,
  "selector_auc": 0.9082017605141819,
  "selector_f1": 0.8374384236453202,
  "selector_accuracy": 0.8303341902313625,
  "selector_peak_auc": 0.8085110324767957,
  "selector_positive_mean": 0.6595325816750102,
  "selector_negative_mean": 0.2159896863713151,
  "keypoint_mae_px": 9.378605615592438,
  "mls_mae_mm": 2.152767816528616,
  "mls_rmse_mm": 3.2549629458601084,
  "mls_f1_3mm": 0.7526881720430108,
  "mls_f1_5mm": 0.7909967845659164,
  "study_mls_mae_mm": 1.0999055201240953,
  "study_mls_f1_3mm": 0.7719298245614035,
  "study_mls_f1_5mm": 0.85,
  "study_boundary_f1": 0.8109649122807017,
  "selection_objective": 1.523874815305601,
  "train_spatial_loss": 5.172258598306483,
  "train_coordinate_loss": 0.029720136558149843,
  "train_mls_loss": 0.8172101453457729,
  "train_threshold_loss": 0.29771604822529907,
  "train_selector_loss": 0.4815938781788521
}
```

## Epoch history

| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7730 | 36.3806 | 4.0333 | 0.0000 | 133.59 | 0.4037 | 0.3247 | 4.76 |
| 2 | 4.3614 | 16.6622 | 4.0333 | 0.0000 | 26.16 | 0.3311 | 0.3708 | 4.55 |
| 3 | 3.7709 | 3.3300 | 4.0333 | 0.0000 | 13.96 | 0.2316 | 0.2573 | 4.55 |
| 4 | 3.1094 | 3.0874 | 4.0333 | 0.0000 | 12.58 | 0.7592 | 0.6413 | 4.55 |
| 5 | 2.7239 | 2.7898 | 4.0333 | 0.0000 | 12.25 | 0.8189 | 0.7363 | 4.55 |
| 6 | 2.4822 | 2.2513 | 4.0333 | 0.0000 | 9.48 | 0.8349 | 0.7317 | 4.55 |
| 7 | 2.4194 | 2.1328 | 4.0333 | 0.0000 | 10.43 | 0.8230 | 0.7746 | 4.55 |
| 8 | 2.3048 | 2.4496 | 3.3442 | 0.2693 | 9.54 | 0.8682 | 0.7790 | 4.55 |
| 9 | 2.2288 | 2.2724 | 1.2485 | 0.8098 | 9.49 | 0.8838 | 0.7856 | 4.55 |
| 10 | 2.0236 | 2.1528 | 1.0999 | 0.8110 | 9.38 | 0.9082 | 0.8085 | 4.55 |
| 11 | 2.0004 | 2.2447 | 2.0564 | 0.6659 | 9.39 | 0.8926 | 0.7542 | 4.55 |
| 12 | 1.9519 | 2.0439 | 1.3863 | 0.7739 | 9.33 | 0.9123 | 0.7589 | 4.55 |
| 13 | 1.9676 | 2.1444 | 1.1807 | 0.8202 | 8.83 | 0.9097 | 0.7795 | 4.55 |
| 14 | 1.8438 | 2.4389 | 1.2701 | 0.7812 | 9.24 | 0.9125 | 0.7866 | 4.55 |
| 15 | 1.8941 | 2.1197 | 1.6641 | 0.7891 | 9.16 | 0.9091 | 0.7904 | 4.55 |
| 16 | 1.7884 | 2.1861 | 1.4034 | 0.8043 | 9.14 | 0.9065 | 0.7987 | 4.55 |
| 17 | 1.7176 | 2.2798 | 2.4837 | 0.4739 | 9.20 | 0.8872 | 0.7546 | 4.55 |
| 18 | 1.7339 | 2.0703 | 1.4752 | 0.7544 | 9.00 | 0.8955 | 0.7872 | 4.55 |
| 19 | 1.6562 | 2.1713 | 1.1430 | 0.8173 | 9.03 | 0.8913 | 0.7712 | 4.55 |
| 20 | 1.6266 | 2.1634 | 1.2065 | 0.7870 | 9.13 | 0.9015 | 0.7739 | 4.55 |
| 21 | 1.6122 | 1.9930 | 1.2736 | 0.8210 | 8.71 | 0.8936 | 0.7961 | 4.55 |
| 22 | 1.5790 | 2.0308 | 1.2558 | 0.7474 | 8.86 | 0.8980 | 0.7999 | 4.55 |
| 23 | 1.5288 | 1.8589 | 1.1179 | 0.8070 | 8.48 | 0.8835 | 0.7920 | 4.55 |
| 24 | 1.5283 | 2.1069 | 1.3653 | 0.7900 | 8.76 | 0.8724 | 0.7589 | 4.55 |
| 25 | 1.5564 | 2.0341 | 1.2596 | 0.7943 | 8.44 | 0.8893 | 0.7851 | 4.55 |
| 26 | 1.5149 | 2.1013 | 1.2989 | 0.7963 | 8.57 | 0.8773 | 0.7753 | 4.55 |
| 27 | 1.5089 | 2.1308 | 1.2505 | 0.7828 | 8.58 | 0.8825 | 0.7838 | 4.55 |
| 28 | 1.4991 | 2.0832 | 1.2776 | 0.7778 | 8.41 | 0.8816 | 0.7837 | 4.55 |
| 29 | 1.4797 | 2.0794 | 1.2617 | 0.8029 | 8.49 | 0.8832 | 0.7828 | 4.55 |
| 30 | 1.4898 | 2.1138 | 1.2605 | 0.7851 | 8.38 | 0.8792 | 0.7801 | 4.55 |

## Post-training full-study CUDA audit

- Audited 12 checkpoint states: best objective, best selector AUC, epochs
  13/15/17/19/21/23/25/27/29, and final epoch 30.
- Each state evaluated all 70 fold-0 studies with CUDA hard-required.
- Total: 840 study-checkpoint evaluations, zero failures, no CPU fallback.
- Best frozen robust single-checkpoint result: epoch 13, MAE `1.381863 mm`,
  Boundary-F1 `0.835069`.
- Best diagnostic single-checkpoint result on the common 6048-profile grid:
  epoch 21, MAE `1.233394 mm`, Boundary-F1 `0.834836`. This is in-fold and
  profile-sensitive, so it is not a production estimate.
- Exp05 peak-aware baseline on the same grid: best MAE `1.288706 mm`; balanced
  MAE `1.318348 mm`, Boundary-F1 `0.828226`.

## Snapshot/cross-model blend audit

- Built 14 validated slice-level blends from Exp08 snapshots and the Exp05
  peak-aware baseline without rerunning model inference.
- Best balanced blend: `75% Exp05 + 25% Exp08 epoch21`.
  - MAE: `1.236911 mm`
  - Boundary-F1: `0.847368`
  - Combined macro-F1: `0.639221`
  - Selection objective: `1.542175`
- The same blend under the previously frozen Exp05 balanced profile achieved
  MAE `1.250444 mm`, a `5.15%` improvement over Exp05 under the identical
  profile (`1.318348 mm`). Boundary-F1 decreased by only `0.009783`.
- Robustness: 16 profiles were within `0.025 mm` of the blend's best MAE and
  31 within `0.05 mm`, versus 11 and 20 for Exp05.
- Decision: train hybrid-soft fold1 with the same target/config family and a
  shortened snapshot schedule through epoch 23. Fold2 is conditional on fold1
  transfer evidence. Final selection requires strict cross-fold validation.

Artifacts:

- `end_to_end_checkpoint_audit/`
- `checkpoint_pooling_expanded/checkpoint_pooling_summary.json`
- `snapshot_blends/`
- `snapshot_blend_pooling_expanded/checkpoint_pooling_summary.json`
