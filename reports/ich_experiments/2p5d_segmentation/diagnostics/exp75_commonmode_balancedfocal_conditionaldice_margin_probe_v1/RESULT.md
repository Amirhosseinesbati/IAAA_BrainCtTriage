# Exp75 result — loss-only hierarchy branch closed

Exp75 completed on commit `f4e0039` with 24 train-only batches, no optimizer and
no held-out access. All 86 related tests passed. MLflow run:
`8d71c4c4a9d14f6fb733579af6bd0d1c`.

The morphology-aware objective retained safe background pressure (`0.462x`) and
positive decoder/head alignment (`0.510`). IPH margin was controlled (`0.959x`),
but the required rare/diffuse margins failed: IVH=`0.538x`, SDH=`0.059x`,
EDH=`0.470x`, and SAH=`0.911x`.

Decision: `reject_exact_loss_weighting_before_calibration_or_outer`. Exp72–Exp75
now close the loss-only hierarchy branch on the incumbent six-class head. The
next experiment must change output architecture and trainable representation. A
factorized head can reproduce all six incumbent probabilities exactly at
initialization while representing `P(foreground)` and
`P(subtype | foreground)` as algebraically separate factors.

Aggregate artifact SHA-256:
`4479434c695cacd1dfd3c830d2016e54406c557df89f318dbb3143f089c42fbe`.
