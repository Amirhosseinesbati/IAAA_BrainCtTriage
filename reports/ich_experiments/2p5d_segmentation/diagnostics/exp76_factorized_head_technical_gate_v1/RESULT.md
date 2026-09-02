# Exp76 result — factorized architecture technical gate passed

Exp76 completed on preregistration commit `8ef9788` without patient images,
optimizer steps or train/calibration/outer/test inference. All 90 related tests
passed on the server. MLflow run: `107a5f810cb9482daccb97ecac0f310e`.

The real Exp61 checkpoint loaded into the factorized wrapper with maximum
six-class probability difference `1.1920929e-7`, mean difference `4.1294763e-12`,
zero hard-argmax mismatch and exactly zero auxiliary-classification difference.
Both residual outputs were exactly zero at initialization. Maximum
foreground→subtype and subtype→foreground cross-gradients were respectively
`1.5768011e-8` and `5.5879354e-9`, comfortably below the locked `1e-6` gate.

The locked scope exposed `2,837,996` trainable decoder/spatial-head parameters,
while encoder and classification trainable counts were both zero and all module
train/eval modes matched the preregistration. All nine technical gates passed.

Decision: `authorize_preregistered_calibration_smoke`. This proves only algebraic
identity, checkpoint compatibility and gradient/scope isolation; it is not
evidence of performance improvement. The next run must remain bounded and must
compare its epoch-zero calibration identity before interpreting any update.

Aggregate artifact SHA-256:
`d29fa93a4cc22ac71aa64dc3207e1368dfdc5368adab2f87876ca360aa418a36`.
