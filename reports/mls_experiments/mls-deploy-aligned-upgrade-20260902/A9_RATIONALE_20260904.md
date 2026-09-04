# A9 frozen-baseline refiner rationale

A8 showed that the reference-conditioned head can reduce continuous MLS error relative to a matched control, but full-network retraining degraded the control and the candidate lost threshold F1. A9 changes one scientifically material factor: it starts from the exact qualified baseline and freezes every pre-existing parameter and buffer. Only the zero-initialized 47,617-parameter outer-point refiner is optimized.

At initialization A9 must be bitwise identical to the qualified baseline. Consequently, any later change is attributable to the refiner and cannot arise from backbone, BatchNorm, coarse heatmap or selector drift. The historical spatial, coordinate, MLS and triage-threshold losses remain unchanged; only their gradient path is restricted. This is a preservation-first test, not a post-hoc repair of A8.

The run is fixed at 10 epochs / 1690 updates, batch16, seed42, fold0 and strict float32 CUDA. No validation inference, epoch selection, threshold tuning or pooling tuning is permitted during training. A single canonical 70-study audit follows only after all provenance and frozen-state checks pass. Failure of any baseline resource gate rejects the model without replication or ZIP creation.

The CUDA real-batch preflight subsequently passed. Initial refined and baseline heatmaps were bitwise identical; one optimizer step changed the refiner while every frozen baseline parameter and buffer stayed bitwise unchanged. The preflight receipt is `A9_PREFLIGHT_20260904.json`, local/server SHA256 `e7bcd33d2461221521a715ef15b00ccd4cac281e6e6a94d6bb6b47ca6e062ab6`.
