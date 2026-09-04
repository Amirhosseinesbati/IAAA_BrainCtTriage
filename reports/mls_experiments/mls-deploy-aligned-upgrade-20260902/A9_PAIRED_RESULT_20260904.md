# A9 frozen-baseline reference refiner — canonical result

## Decision

**Rejected at the fold0/seed42 resource gate.** The A9 checkpoint is not a
release candidate, is not eligible for replication, and must not be used for a
submission ZIP. The checkpoint remains archived in MLflow only; it is not copied
to the local checkpoint/mls release area.

This decision is not based on train loss or an informal validation split. It is
from one fixed, CUDA-only, 70-study held-out fold0 screen against the
runtime-qualified baseline and its frozen prospective bounds.

## What A9 tested

A9 loaded the exact qualified baseline checkpoint with SHA-256
c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062,
enabled a zero-initialized outer reference refiner, and trained only that
refiner for a fixed 10 epochs / 1,690 optimizer steps. All baseline parameters
and buffers were frozen.

The final provenance preflight independently verified:

- All 1,938 baseline state-dict tensors are bitwise unchanged.
- The only additional state keys are the six outer_refinement.refine tensors.
- The training preflight, source/data manifest, history, checkpoint, MLflow run,
  fixed epoch, and qualified runtime reference all match their recorded SHA-256
  contracts.
- The dedicated epoch-10 evaluator has an executable inference body identical to
  the qualified epoch-15 evaluator after normalizing only the epoch guard and
  output epoch metadata.

No private study predictions were copied locally or uploaded to MLflow.

## Fixed canonical screen

| Metric | Qualified baseline | A9 | A9 − baseline |
| --- | ---: | ---: | ---: |
| MAE (mm) | 1.470565 | 1.448133 | -0.022432 |
| RMSE (mm) | 2.440431 | 2.435763 | -0.004668 |
| F1 @ 1 mm | 0.820513 | 0.825000 | +0.004487 |
| F1 @ 3 mm | 0.819672 | 0.786885 | **-0.032787** |
| F1 @ 5 mm | 0.736842 | 0.810811 | +0.073969 |
| Boundary-F1 | 0.778257 | 0.798848 | +0.020591 |
| Selection objective | 1.914051 | 1.850437 | -0.063614 |

The candidate passed MAE, F1@5, Boundary-F1, and objective gates, but failed the
required F1@3 gate. A mixed result cannot justify a final-triage promotion:
the 3-mm boundary is clinically and competition-relevant, and the screen was
pre-registered to reject any deterioration there.

## Interpretation

The frozen refiner can improve coarse localization and the 5-mm boundary while
shifting enough studies around the 3-mm decision threshold to reduce F1@3. This
is evidence that the hypothesis has signal, not evidence that the model is
deployable. The next MLS design must explicitly preserve the 3-mm threshold
behavior; repeating the same A9 recipe would not answer that question and is
prohibited.

## Evidence

- Training run: bb4a898d61d544c9a450bfcd4ccb4b79
- Candidate checkpoint SHA-256:
  853dc584f0c2baef731ddcf8d8b0ba2eef0914606285c7375039a1ea7d6bd8fe
- Canonical candidate audit SHA-256:
  81f3950211252783ee42c6c069cc9c5fd0550d0756009392122f7a91b0a6094f
- Final pair receipt SHA-256:
  d67d0e9b1cd3fa84e31e053c11fc521b75e636ff25d72eef884b8a4de594e0d9
- Training runtime: 426.52 seconds; canonical CUDA audit runtime: 46.51 seconds.

The corresponding public JSON receipts are
A9_TRAINING_SUMMARY_20260904.json, A9_TRAINING_HISTORY_20260904.json,
A9_CANONICAL_AUDIT_AGGREGATE_20260904.json, and
A9_PAIR_AGGREGATE_20260904.json.
