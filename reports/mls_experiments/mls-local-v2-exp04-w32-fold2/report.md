# MLS experiment: mls-local-v2-exp04-w32-fold2

- Status: `completed`
- Updated UTC: `2026-08-27T12:03:19.148006+00:00`
- MLflow run id: `c78a77479f3e4492be0ed9cb54e707e2`
- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.
- Config: `{"fold": 2, "use_competition_folds": true, "backbone": "hrnet_w32", "input_channels": 3, "image_size": 512, "heatmap_sigma": 3.0, "mls_loss_weight": 0.25, "threshold_loss_weight": 0.1, "softargmax_temperature": 1.0, "threshold_temperature_mm": 0.5, "use_selector": true, "selector_loss_weight": 1.0, "spatial_loss_weight": 0.25, "coordinate_loss_weight": 0.5, "gradient_accumulation_steps": 1, "dataset_variant": "multitask_v2", "learning_rate": 0.0001, "weight_decay": 0.001, "head_dropout": 0.1, "epochs": 30, "batch_size": 5, "val_split": 0.2, "top_k_slices": 7, "selector_threshold": 0.9, "negative_value_mm": 0.1, "aggregation": "joint_component", "selector_relative_ratio": 0.5, "aggregation_quantile": 0.9, "anchor_window_radius": 2, "min_active_slices": 3, "heatmap_guard_ratio": 0.5, "rotation_deg": 8.0, "translation": 0.03, "intensity_jitter": 0.03, "augment_prob": 0.6, "early_stopping_patience": 8, "lr_scheduler_patience": 5, "use_amp": false, "num_workers": 2, "seed": 42}`

## Best validation

```json
{
  "epoch": 19.0,
  "train_loss": 1.5066449501374686,
  "lr": 3.5659838364445505e-05,
  "peak_vram_gb": 4.545804977416992,
  "val_loss": 1.7390785014374988,
  "selector_auc": 0.8985235235235235,
  "selector_f1": 0.8228228228228228,
  "selector_accuracy": 0.8236173393124065,
  "selector_positive_mean": 0.7962662785491618,
  "selector_negative_mean": 0.1979000440087042,
  "keypoint_mae_px": 8.526450526129615,
  "mls_mae_mm": 2.2273336071409173,
  "mls_rmse_mm": 3.2040098329231483,
  "mls_f1_3mm": 0.8775981524249422,
  "mls_f1_5mm": 0.8224852071005917,
  "selection_objective": 2.683164437971748,
  "train_spatial_loss": 5.0193152546247415,
  "train_coordinate_loss": 0.017591993877793238,
  "train_mls_loss": 0.3296027398420304,
  "train_threshold_loss": 0.15289228651988893,
  "train_selector_loss": 0.14533022839450285
}
```

## Epoch history

| epoch | train loss | val MLS MAE | kp MAE | selector AUC | selector F1 | F1@3 | F1@5 | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6677 | 58.5185 | 110.04 | 0.2422 | 0.0000 | 0.7925 | 0.6897 | 4.76 |
| 2 | 4.0380 | 7.2503 | 18.11 | 0.2055 | 0.0000 | 0.8224 | 0.7582 | 4.55 |
| 3 | 3.6401 | 3.4155 | 12.92 | 0.8345 | 0.0000 | 0.8159 | 0.7429 | 4.55 |
| 4 | 3.0051 | 4.3259 | 14.90 | 0.7921 | 0.0000 | 0.8190 | 0.7756 | 4.55 |
| 5 | 2.7964 | 2.6823 | 11.80 | 0.8049 | 0.1127 | 0.8399 | 0.8012 | 4.55 |
| 6 | 2.5258 | 2.7088 | 12.12 | 0.8341 | 0.7912 | 0.8393 | 0.8000 | 4.55 |
| 7 | 2.3734 | 2.4475 | 9.94 | 0.8347 | 0.7735 | 0.8604 | 0.8138 | 4.55 |
| 8 | 2.1804 | 2.4072 | 9.40 | 0.8801 | 0.7561 | 0.8618 | 0.8208 | 4.55 |
| 9 | 2.0972 | 2.3582 | 9.51 | 0.8989 | 0.7988 | 0.8802 | 0.8621 | 4.55 |
| 10 | 1.9997 | 2.4101 | 9.75 | 0.9046 | 0.8235 | 0.8565 | 0.7903 | 4.55 |
| 11 | 1.9273 | 2.4664 | 9.36 | 0.9132 | 0.8264 | 0.8612 | 0.8012 | 4.55 |
| 12 | 1.8721 | 2.5395 | 9.73 | 0.9031 | 0.8000 | 0.8591 | 0.8195 | 4.55 |
| 13 | 1.8804 | 2.1762 | 8.96 | 0.8618 | 0.7311 | 0.8832 | 0.8201 | 4.55 |
| 14 | 1.8066 | 2.5589 | 9.93 | 0.9085 | 0.8127 | 0.8451 | 0.8152 | 4.55 |
| 15 | 1.7145 | 2.3266 | 8.95 | 0.9176 | 0.8447 | 0.8784 | 0.8352 | 4.55 |
| 16 | 1.6701 | 2.3524 | 8.98 | 0.9142 | 0.8336 | 0.8532 | 0.8094 | 4.55 |
| 17 | 1.6744 | 2.2981 | 9.07 | 0.9054 | 0.8180 | 0.8664 | 0.8363 | 4.55 |
| 18 | 1.6059 | 2.3213 | 9.38 | 0.8875 | 0.7595 | 0.8844 | 0.8314 | 4.55 |
| 19 | 1.5066 | 2.2273 | 8.53 | 0.8985 | 0.8228 | 0.8776 | 0.8225 | 4.55 |
| 20 | 1.4904 | 2.2458 | 8.77 | 0.9085 | 0.8208 | 0.8514 | 0.8235 | 4.55 |
| 21 | 1.4534 | 2.3116 | 8.85 | 0.8635 | 0.7685 | 0.8650 | 0.8412 | 4.55 |
| 22 | 1.4107 | 2.4221 | 8.83 | 0.8818 | 0.7584 | 0.8545 | 0.7866 | 4.55 |
| 23 | 1.3776 | 2.4114 | 8.62 | 0.8927 | 0.7922 | 0.8696 | 0.8024 | 4.55 |
| 24 | 1.3668 | 2.6381 | 8.62 | 0.8828 | 0.7828 | 0.8643 | 0.8260 | 4.55 |
| 25 | 1.3481 | 2.3679 | 8.38 | 0.8878 | 0.7994 | 0.8545 | 0.8095 | 4.55 |
| 26 | 1.3826 | 2.3161 | 8.22 | 0.8896 | 0.8050 | 0.8552 | 0.8107 | 4.55 |
| 27 | 1.3559 | 2.3641 | 8.25 | 0.8824 | 0.7987 | 0.8688 | 0.8036 | 4.55 |

## Post-training checkpoint audit

All 67 fold-2 studies were evaluated end-to-end on CUDA with zero failures.
The three retained checkpoints behave differently after study-level pooling:

| Checkpoint | Selection epoch | Frozen-profile MAE mm | RMSE mm | Bias mm | F1@3 | F1@5 |
|---|---:|---:|---:|---:|---:|---:|
| Best objective | 19 | 2.2017 | 3.6082 | -0.3252 | 0.8519 | 0.8261 |
| Best slice MAE | 13 | 2.9264 | 5.7432 | -2.6818 | — | — |
| Best selector AUC | 15 | **2.1037** | **3.4083** | **-0.1804** | 0.7667 | **0.8936** |

The best-slice-MAE checkpoint is not the best study model: its selector is not
calibrated for the frozen `0.9` gate, causing severe underestimation. The AUC
checkpoint is the fold-2 deployment candidate among the retained states.

### Selector/measurement decomposition of the AUC checkpoint

- Annotated-target-vs-other selector AUC: `0.9322`.
- Target retrieval recall: top-1=`1.000`, top-3=`1.000`, top-5=`1.000`.
- Oracle annotated-target median MAE: `2.0849 mm`.
- Oracle annotated-target p90 MAE: `1.3658 mm`.
- Predicted-selector top-3 median MAE: `2.2936 mm`.

The selector is no longer the dominant error source. The large gain from
oracle median to oracle p90 proves that study-level measurement distribution
and aggregation are now the bottleneck. This also explains why selecting a
checkpoint only by slice MAE can fail end-to-end.

## Three-fold decision after fold 2

The robust three-fold profile is `anchor_window(radius=2)`, absolute selector
gate `0.9`, minimum active slices `3`, unweighted `q=0.9`, with heatmap guard
ratio `0.5`. Its fold MAEs are `1.4337 / 1.4791 / 2.0224 mm`, mean `1.6451 mm`.
This improves the previous `joint_component` three-fold mean (`1.6691 mm`) but
also confirms that fold 2 is substantially harder. The two-fold `1.4519 mm`
estimate must therefore not be treated as the expected hidden-set error.

Detailed aggregate artifacts are stored in `end_to_end_objective/`,
`end_to_end_mae/`, `end_to_end_auc/`, `decomposition_auc/`, and
`../threefold_w32_pooling_auc/`. Per-study/per-slice CSVs remain local and are
excluded from MLflow uploads by policy.
