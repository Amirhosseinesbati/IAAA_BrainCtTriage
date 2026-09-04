# A6 factor-swap diagnostic: neither simple swap improves baseline

The fixed canonical-pooling cache diagnostic completed on the server without
training, model inference or threshold search. Both native combinations exactly
reproduced the corrected70-study fold0 metrics. Eight target-server tests passed.

| Scalar geometry | Selector probabilities | MAE mm | F1 at3mm | F1 at5mm | Objective |
|---|---|---:|---:|---:|---:|
| Baseline | Baseline | 1.470959 | 0.819672 | 0.736842 | 1.914444 |
| A6 | Baseline | 1.500135 | 0.786885 | 0.761905 | 1.951345 |
| Baseline | A6 | 1.609999 | 0.754098 | 0.789474 | 2.066427 |
| A6 | A6 | 1.480787 | 0.754098 | 0.809524 | 1.917165 |

No mixed combination dominates baseline or meets all resource requirements.
Do not release a hybrid, restart A6 replication, or sweep blending weights.

## Boundary behavior

At3mm the native baseline has TP25/FP5/FN6/TN34. Native A6 and baseline geometry
with A6 selector each have TP23/FP7/FN8/TN32; both change4 decisions versus
baseline, with0 corrections and4 new errors. A6 geometry with baseline selector
has TP24/FP6/FN7/TN33, changing2 decisions, both new errors. These equal aggregate
counts do not prove the changed patients are identical.

At5mm native baseline has TP14/FP4/FN6/TN46. Native A6 has TP17/FP5/FN3/TN45:
4 baseline errors corrected,2 new errors. A6 geometry with baseline selector
corrects2 and introduces2 errors; baseline geometry with A6 selector corrects2
and introduces0. Thus the5mm gain and3mm regression cannot be summarized as a
uniform improvement or uniform failure of either subsystem.

## Interpretation and next research question

Changing the selector alone is sufficient to reproduce the adverse aggregate
3mm confusion counts in this retrospective construction. Geometry changes also
hurt3mm with the old selector. This supports examining the boundary-specific
training objective, target construction and effective sampler exposure rather
than declaring that a simple component swap fixes the model.

Before another run, inspect how the current dataset, sampler and loss represent
studies/slices around3mm versus5mm, and reconcile with previously tried ordinal,
dual-selector and study-bag experiments. If proposing boundary weighting or a
regularizer, predefine one controlled change with matched optimization and the
corrected inference signature. Do not repeat an earlier intervention under a new
name or claim that weighting must work. Research is useful where it informs
this specific mechanism, not as a reason to launch an unfocused architecture grid.

## Limitations and provenance

Fold0 has been repeatedly used; this is not independent validation. Scalar MLS
and both presence/peak probabilities were swapped, not the learned networks.
Geometry also carries heatmap peak, but guard0 makes it irrelevant to selection.
The caches align by series and sorted index, not independently checked SOP UID.
Any future hybrid claim would require fresh shared-reader CUDA/identity parity;
no such claim is made here. Nothing in this result establishes triage improvement.

Source commit7387e01; source SHA256
1fc241b49c2deaaaa9c7202a28b4f5b81772fc78825ff9bd98dd133e75f59ed1.
Protocol SHA256
41a567932db8df8deec638ae8fed8967ecc1202a188bedb95554d109469adee4.
Input hashes and canonical profile are pinned by the comparator-correction
protocol. Only aggregate result/receipt files are retained locally; original
private study/slice values stay on the server. No scheduler was reactivated.

Aggregate result `a6_canonical_factor_swap_20260904.json` SHA256:
`8ed2f8f076c255106ff5ad5294530261ac48a938a4fc26bfa6ba6df1bcad6420`.
MLflow receipt `a6_canonical_factor_swap_mlflow_receipt_20260904.json` SHA256:
`dd9a5fae4aa63b2393b4cf1b03571f39449c56b5796dc6a1cea4a4f5168bb85b`.
Both are retained alongside this report; remote originals are under
`/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/`.
MLflow run `9b8e9fc5996a42549e3aca5aa40763d7` remains FINISHED. All 12 diagnostic
metrics were read back and matched; artifacts are under
`reports/canonical_factor_swap`. This is diagnostic evidence, not a release.
