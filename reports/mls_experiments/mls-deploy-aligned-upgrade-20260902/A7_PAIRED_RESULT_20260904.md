# A7 paired translation experiment — completed, not promoted

Both fixed arms completed 15 epochs / 8115 optimizer steps. The evaluator rebuilt matching initialization, per-epoch augmented-input exposure and embedded checkpoint provenance. No validation images were used for training or epoch selection. Both checkpoints were evaluated once on the fixed 70-study fold0, seed42, epoch15 canonical CUDA pipeline with the qualified same-runtime reference. No pooling, threshold or precision tuning was performed.

| Metric | Same-runtime baseline | Paired control | Paired consistency |
|---|---:|---:|---:|
| Study MAE mm | 1.4705651353 | 1.6882281184 | 1.5590386116 |
| F1 at 3 mm | 0.8196721311 | 0.7457627119 | 0.8125000000 |
| F1 at 5 mm | 0.7368421053 | 0.7777777778 | 0.7804878049 |
| Boundary F1 | 0.7782571182 | 0.7617702448 | 0.7964939024 |
| Selection objective (lower better) | 1.9140508989 | 2.1646876288 | 1.9660508067 |

## Interpretation and decision

Consistency improved its matched control: MAE decreased 0.1291895069 mm, 3-mm F1 increased 0.0667372881, Boundary F1 increased 0.0347236576, and objective decreased 0.1986368221. This is single-seed evidence for benefit within this paired-training setup, not proof of generalization across seeds/folds.

Against the qualified baseline, consistency still increased MAE by 0.0884734763 mm and worsened objective by 0.0519999078; 3-mm F1 fell by 0.0071721311. Its better 5-mm and Boundary F1 do not override the failed preregistered gates. Neither arm passed the resource screen. No automatic replication, promotion, Champion replacement or submission ZIP is allowed from this result. Final triage Macro-F1/Urgent F1 improvement has NOT been established.

The control degradation means the paired-view training setup itself cannot be assumed harmless. The regularizer partly recovers that loss but does not establish superiority over the original training recipe. Do not blindly increase its weight, select another epoch or tune pooling on these 70 held-out studies. Any next experiment requires a distinct, training-only-supported hypothesis and a preregistered comparison; retain current Champion.

## Artifacts and recovery

Remote root: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a7_paired_translation_20260904`.

- Control checkpoint: `control/mls_multitask_epoch_015.pth`; SHA256 `a843b370f4e92798586a84f823f06f29aaf100df948b86bd5c5e24dba86b5820`; MLflow run `0f5b17c509714f7fa3da96726c59cfb6`.
- Consistency checkpoint: `consistency/mls_multitask_epoch_015.pth`; SHA256 `4ed54985e07e0a7bf6a88f70b924dc144becedbe42cd85a90cd8f1f08826484b`; MLflow run `5746a793c5d04da994935651fbaae5d4`.
- Final aggregate: `canonical_pair_audit/pair_aggregate_summary.json`.
- Control audit aggregate SHA256: `abb440ad36bf75af68d59258aa5dc46520627ce9e53822cb633a22e64f7ae657`.
- Consistency audit aggregate SHA256: `5fe47b1f38d6a94ed09ceaf47dadcfd0c9eb0bbe3b69e2fb03999581d72a4a15`.
- Training durations: control 3580.635 s; consistency 3597.475 s.

Completion observer session 78065 and evaluation session 78157 both exited successfully. Do not restart either training or evaluation. Private per-study predictions remain server-only. At report creation, final aggregate transfer/checksum verification and final audit MLflow archival are still pending; no claim is made that these final artifacts have already been archived locally or verified in MLflow. Training MLflow run IDs alone do not prove final artifact delivery.

## Subsequent archival verification

The final aggregate was subsequently transferred to this report directory as `A7_PAIR_AGGREGATE_20260904.json`. Local and remote SHA256 both equal `2869776a2b793542326446387e54c1ca66c3537b2a6c28c43b7d723f64d8eaf1`.

Both existing MLflow runs now contain `reports/a7_final_audit/aggregate_summary.json` (their respective arm) and `reports/a7_final_audit/pair_aggregate_summary.json` (the paired comparison). All four uploaded artifacts were independently downloaded on the server and their SHA256 matched the source, using the aggregate hashes recorded above. Archival session 72439 exited successfully. This verifies final audit-summary delivery, not a new claim about checkpoint-artifact download verification. No private rows were transferred. No additional training was launched.
