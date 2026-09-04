# A7 mechanism diagnostic: stability improved, release gates still failed

Completed on the server GPU, session59193 exit0. Fixed 128 positive training slices, sample SHA256 `b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1`; no held-out images, no model updates. All three models received identical image/target bytes (exposure SHA256 `70b85b6d786f019f09e4dd88292a9e617009a79b04b6346b086673b1fed034a5`). Model parameters and buffers were hashed before/after and remained unchanged. Source SHA256 `2c245fd2e50ce33d3fc67f4a632c9ce9125ae6b4f99f551da51a723aa7b0b110`; compile check passed on server. Baseline aggregates reproduce the original fixed probe.

| Aggregate | Baseline | A7 control | A7 consistency |
|---|---:|---:|---:|
| Original training-slice MLS MAE mm | 1.455074 | 1.271845 | 1.158305 |
| Mean absolute MLS change, +8px horizontal | 0.785377 | 0.736053 | 0.432926 |
| Mean absolute MLS change, +8px vertical | 0.356826 | 0.480860 | 0.284071 |
| Horizontal overlap heatmap JS | 0.00187710 | 0.00220742 | 0.000358854 |
| Vertical overlap heatmap JS | 0.00162692 | 0.00141415 | 0.000178338 |
| Mean original landmark error mm | 3.408017 | 2.875240 | 2.981452 |
| Horizontal 3mm prediction crossings /128 | 13 | 7 | 9 |
| Horizontal 5mm prediction crossings /128 | 7 | 8 | 0 |

The regularizer substantially reduced mean heatmap and geometric translation instability on these training samples. It did not simply have zero practical effect despite its tiny initial gradient. However, improved average stability is not universal robustness: consistency horizontal maximum MLS change is 5.344013 mm versus baseline2.605244/control4.207992, and horizontal selector maximum change is0.335510 versus baseline0.183330. Consistency also has worse original mean landmark error than its paired control and more horizontal 3mm crossings than that control.

Control improves original training-slice MLS fit yet loses held-out study accuracy; consistency improves training fit and stability further yet still fails the fixed held-out baseline gates. This is compatible with a generalization or slice-to-study aggregation mismatch, not proof of its exact cause. Training slices and held-out studies are different units; their MAEs must NOT be subtracted as a formal train-validation gap. No negative-slice specificity or final Urgent F1 was measured by this positive-only probe.

Decision: retain the failed-resource-screen classification. No TTA, weight sweep, epoch reselection, automatic replication or submission. A7's stability mechanism worked in this limited diagnostic but is insufficient for the requested final triage improvement. Further work should investigate errors contributing to the final study decision and generalization rather than assuming stronger translation consistency must win.

Remote aggregate: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a7_paired_translation_20260904/training_translation_comparison.json`. At creation of this report, transfer and MLflow archival of this new diagnostic aggregate are pending; the earlier final resource-screen audit was already archived and download-verified. No private per-slice rows were saved by this diagnostic.

## Archival completed subsequently

Local aggregate: `A7_TRAIN_TRANSLATION_COMPARISON_20260904.json`. Local, server-source and MLflow-downloaded SHA256 all match `a1553ea7a792b3acdaedec00eefaabd385c96d8a92686ca92d7aded72458fafd`. MLflow run `5746a793c5d04da994935651fbaae5d4`, artifact `reports/a7_mechanism/training_translation_comparison.json`. Transfer session27576 and archive verification session1554 exited0. No model checkpoint was promoted or deleted.
