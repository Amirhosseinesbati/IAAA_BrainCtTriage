# Same-pooling correction: A2-A6 resource results

## Outcome

The comparator defect is confirmed and repaired for the existing cached studies.
All legacy profile metrics reproduced, and the canonical baseline predictions
matched the immutable three-seed audit **for every one of 70 studies**.
All six caches share exact fold0 study identities, truths and ordered slice
indices. Checkpoint configuration, epoch15/seed42 identity and all input hashes
were verified. No new model inference or training was performed.

The valid single-seed comparison is:

| Model | MAE mm | F1 at3mm | F1 at5mm | Boundary F1 | Objective |
|---|---:|---:|---:|---:|---:|
| Baseline Exp16 | 1.470959 | 0.819672 | 0.736842 | 0.778257 | 1.914444 |
| A2 signed | 1.528905 | 0.786885 | 0.789474 | 0.788179 | 1.952546 |
| A3 study bag | 1.746995 | 0.781250 | 0.700000 | 0.740625 | 2.265745 |
| A4 pair rank | 1.654460 | 0.825397 | 0.717949 | 0.771673 | 2.111115 |
| A5 detached rank | 1.631716 | 0.754098 | 0.820513 | 0.787306 | 2.057105 |
| A6 local geometry | 1.480787 | 0.754098 | 0.809524 | 0.781811 | 1.917165 |

None passes every original gate under the repaired comparison. Bounds are
unchanged, with 1e-8 tolerance only for decimal-rounding noise. Baseline itself
fails only the improvement objective, as expected: that gate requires baseline
objective minus0.01, not just reproducing baseline.

For A6, corrected changes versus baseline are MAE **+0.009828mm**, F1@3
**-0.065574**, F1@5 **+0.072682**, objective **+0.002720**. This is a different
and more nuanced finding than the earlier impression of a large MAE collapse.
Its gain at5mm does not compensate for failing the required3mm noninferiority.
No candidate has established improved frozen-context triage or release readiness.

## Why the earlier interpretation was wrong

The original resource scripts demanded 0.5/top3/p90 from candidates but compared
them with baseline metrics using its frozen checkpoint policy. The intended A2
plan required matching baseline inference. The baseline itself has MAE2.470454
with the old audit profile, versus1.470959 with its actual checkpoint pooling.
Thus prior all-fail statements described the old screen, not a valid matched
comparison. This was an evaluation-process error, not evidence of broken data.

Canonical policy, recovered from both checkpoint and three-seed evaluator:
threshold0.6, relative_component, ratio0.3, quantile0.75, probability weighting,
min_active3, top5, anchor_radius3, heatmap_guard0, negative0.1, then clamp[0,30].
It was frozen before corrected candidate outcomes. No thresholds were selected
using the new results, and every completed A2-A6 candidate was included.

Historical JSONs and metrics remain intact. Their human-readable reports now
point here. Do not reuse the legacy A2-A6 resource evaluators as a scientifically
valid same-pooling gate. Any future training/evaluation launcher must assert the
candidate and comparator inference signatures match, including clipping.

## Evidence, limitations and next action

Source commit: d667486. Five target-server tests passed; no local test or model
execution. Reconstruction JSON SHA256:
0dd707cb86b39ef888ce56ee9fb29f455f6b6cfd4b5b870a32f6eeaa7dcf1b70.
Protocol SHA256:
15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee.
Script SHA256:
60d5c1e2da3607bccbfeb4a6482670ec3aa6c46e613fca41ba5edd4869ae1ec2.

Local aggregate: ALIGNED_CACHED_RESOURCE_RECONSTRUCTION_20260904.json.
Remote aggregate: /workspace/iaaa_artifacts/mls_deploy_aligned_20260902/aligned_cached_resource_reconstruction_20260904.json.
Private slice outputs remain server-only; no individual predictions were printed.
An independent MLflow correction run preserves the revised evidence separately
from historical runs; its receipt is stored beside the remote aggregate.

MLflow correction run `8478b358f7b84f47b41f3b0ca882152d` was read back FINISHED,
with54 aggregate metrics and matching A6 MAE1.480786694117955. Its URL is
https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/8478b358f7b84f47b41f3b0ca882152d.

This is still one repeatedly examined development fold, not independent
generalization proof. A4/A5 retain their separate batch/update-count confound.
The training-positive postmortem is valid, but its interpretation must use the
corrected study outcome, not the old mismatched reference.

Next bounded discriminator: the planned baseline/A6 scalar-geometry-versus-
selector 2x2 swap, now under the **canonical** policy above, with exact keys and
native-outcome reproduction. This is retrospective sensitivity analysis only;
it cannot promote a hybrid or choose a threshold. Do not start another training
run before this resolves which intervention is scientifically justified.
All original multi-seed/cross-fold/frozen-triage hard gates remain mandatory.
The user's automatic-monitor cancellation remains in effect.
