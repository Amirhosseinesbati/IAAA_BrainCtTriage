# A6 postmortem: better training geometry, worse held-out study outcome

The fixed training-only diagnostic completed on the3090, with CUDA forward
passes only, in9.785 seconds excluding MLflow transfer. Both baseline reference
MAEs reproduced exactly; all128 preselected positive training slices /89studies
matched the original sample digest. No validation image, parameter search or
optimization was used. This is mechanistic training evidence, not generalization.

| Training-only metric | Baseline Exp16 epoch15 | A6 epoch15 |
|---|---:|---:|
| DARK slice MLS MAE mm | 1.461234212 | 1.016691923 |
| Global softargmax slice MLS MAE mm | 0.921112597 | 0.812771797 |
| Local softargmax slice MLS MAE mm | 1.101026297 | 0.665201426 |
| DARK slice F1 at3mm | 0.854961832 | 0.884057971 |
| DARK slice F1 at5mm | 0.869565217 | 0.926315789 |
| Local-versus-DARK absolute MLS gap mm | 0.672004819 | 0.678498983 |
| Third landmark target outside radius6 window | 15.625% | 8.59375% |

DARK mean landmark errors were [2.034939,1.805857,6.387321]mm for baseline
and [1.980506,1.613197,5.289494]mm for A6. The first two target-outside-window
fractions were [0.78125%,0%] for baseline and [0%,0%] for A6. Despite training
improvement, the independently completed fixed70-study A6 resource audit failed
all five gates (study MAE2.561580288mm). A6 stays rejected.

## What this changes

The simple explanation that A6 failed because it could not learn localization
even on the training positives is not supported by this fixed sample. The
training/deployment local-DARK gap also remains roughly0.68mm rather than being
eliminated. The more relevant unresolved questions are generalization to held-out
anatomy, selection/calibration on irrelevant or negative slices, study pooling,
and processed-PNG versus raw-DICOM representation. None is uniquely established
as the cause by this positive-only sample.

Do not compare the training slice MAE directly against held-out study MAE as a
numerical generalization gap: the populations and units of aggregation differ.
Do not infer that better training F1 improves triage. Do not promote a posthoc
decoder swap. The baseline run and A6 are each one seed; this is not a multi-seed
causal estimate or an independent validation result.

## Next bounded investigation, before new training

The existing audit code already preserves per-slice probabilities and MLS values
server-only in `study_slice_predictions.csv`. Inspect only schema/provenance,
then consider a fixed2x2 diagnostic: baseline/A6 geometry crossed with baseline/A6
selector probabilities, aligned by exact study and slice keys, using unchanged
0.5/top3/p90. Reproduce both native aggregate results first. Export only aggregate
statistics. This can separate sensitivity to the selector from geometry without
another training run or threshold search; mixed results remain retrospective and
cannot rescue A6. If compatible baseline cache is absent, explicitly scope the
necessary fixed baseline inference before running it; never assume alignment.

## Preservation

Aggregate and terminal JSON files are local under
`server_aggregate/a6_seed42_resource_screen_20260904/` and remote under
`/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/` with names
`a6_training_geometry_postmortem_20260904.json` and `.status.json`.

- Result SHA256: `15f61521a052d1ed6f285d2734b1efd44652f4f9e8a55bcf2f106ac7d153e518`.
- Status SHA256: `c2c1bf379dec7ceb6c752390d7ae51d027b4b72ea742b88d06f4d4ab7b057feb`.
- Probe source SHA256: `ee868f2e83fa48ce32b14c45eb3b1821c87f290523853c52a93f9a3af186d9b0`.
- Sample SHA256: `b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1`.

MLflow logging succeeded on run `9b8e9fc5996a42549e3aca5aa40763d7`, artifact
directory `reports/a6_training_postmortem`; diagnostic metrics use the separate
`postmortem_` prefix. For JSON interpretation, the reused summary function stores local-softargmax statistics under
`local_vs_dark.decoders.softargmax`; global values are under `global_vs_dark`.
No checkpoint was changed, promoted or transferred as a best model. No monitoring
schedule was reactivated, and the server was not stopped.
