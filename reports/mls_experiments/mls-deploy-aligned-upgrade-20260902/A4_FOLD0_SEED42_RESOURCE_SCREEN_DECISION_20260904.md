# A4 fold-0 / seed-42 resource-screen decision

> Comparator correction (2026-09-04): the historical resource screen mixed
> candidate 0.5/top3/p90 with baseline checkpoint pooling. Preserve its numerical
> record, but use `ALIGNED_CACHED_RESOURCE_CORRECTION_RESULT_20260904.md` for
> the valid same-pooling comparison. The old rejection alone did not establish
> matched-deployment inferiority. No promotion is authorized by this annotation.

## Decision

**Rejected — stop A4 expansion.**  The pair-ranking intervention completed its
fixed epoch-15 CUDA-only resource screen, but it failed all five
pre-registered gates.  A4 must not start seeds `2026` or `3407`, the three-seed
audit, any cross-fold run, pooling/threshold rescue, ensemble, promotion, or a
submission ZIP.

This is a scientifically useful negative result, rather than evidence that the
infrastructure, data transfer, or CUDA evaluator is broken: training exited
zero; the held-out CUDA audit completed zero-failure inference for exactly 70
studies; and the result was evaluated with the locked profile, checkpoint, and
gates.  It rejects this *shared-backbone, same-study RankNet auxiliary-loss*
formulation at its fixed weight and recipe.

## Immutable execution evidence

- Run: `mls-vast-da-a4-pair-rank-fold0-seed42`; training started
  `2026-09-04T06:05:24Z` and ended `2026-09-04T06:48:42Z` with exit code `0`.
- Fixed candidate: fold 0, seed 42, epoch 15; no epoch selection occurred.
- CUDA audit: `NVIDIA GeForce RTX 3090`, CUDA-only/no CPU fallback, 70 held-out
  studies, zero inference failures, and audit exit code `0`.
- Audit interval: `2026-09-04T07:08:35.539345+00:00` to
  `2026-09-04T07:10:35.590560+00:00`.
- MLflow aggregate-only run: `c68bbdcab8f54384a77f5c7557412cb5`.  Per-study
  predictions remain on the server and were not transferred or registered.

## Fixed-profile gate result

The only decision profile is the pre-registered selector threshold `0.5`,
top-3 selection, and p90 aggregation.  The table reports the exact locked
comparison, not a re-tuned profile.

| Metric | Locked requirement | A4 observed | Margin | Result |
|---|---:|---:|---:|---|
| MAE | <= 1.470959 mm | 1.878737 mm | +0.407778 mm | fail |
| F1 @ 3 mm | >= 0.819672 | 0.794521 | -0.025152 | fail |
| F1 @ 5 mm | >= 0.736842 | 0.697674 | -0.039168 | fail |
| Boundary F1 | >= 0.778257 | 0.746097 | -0.032160 | fail |
| Selection objective | <= 1.904444 | 2.386542 | +0.482098 | fail |

The decision JSON records `rejected_stop_a4_expansion`, `promotion_eligible:
false`, and `submission_zip_allowed: false`.

## What the aggregate diagnostics do — and do not — show

The fixed p90 profile has a +0.421015 mm mean signed error and an MAE of
1.878737 mm.  A diagnostic `relative_component` profile happens to reduce
in-sample MAE to 1.540498 mm and change the bias to -0.128414 mm, but it is
explicitly an in-sample diagnostic and did not supply a boundary-F1 decision.
It is therefore **not** a valid rescue, replacement aggregation rule, or reason
to continue A4.  The locked evaluator itself labels such best-profile output as
diagnostic only.

The careful conclusion is limited: applying pair ranking to the shared model
did not produce a deploy-aligned improvement at the locked operating point and
made the final estimate sensitive to the choice of aggregation.  It does not
identify a single causal defect in the data or implementation.  The next
hypothesis should avoid repeating this formulation unchanged and should first
isolate any ranking signal from geometry supervision (for example by a
separately pre-registered selector-only/frozen-feature experiment), then earn a
fresh fold-0 resource screen before replication.

## Local aggregate-artifact verification

Only these server-generated aggregate artifacts were copied into
`server_aggregate/a4_seed42_resource_screen_20260904`; the local SHA-256 value
matches the remote value in every case.  The rejected checkpoint and all
per-study files remain server-only.

| Local artifact | SHA-256 |
|---|---|
| `A4_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json` | `91ffa56cb4644ce591f819d7b26ba9f43281218daa40fce2c044e6f78420adcf` |
| `audit_launcher_status.json` | `6673f1776f77c62f2d679106146373add6b8507033c78535476a2cf33eaa4c46` |
| `audit_status.json` | `a427c09052b930bc612cee949861b9cff82627b8261ad333272f8924471f7562` |
| `epoch015_metrics.json` | `acbb3e3ceab087deb198cd108284a1fdb8f7474e9dd8712179621ba4e0a76dd1` |

## Next research constraint

Do not modify A4's pooling, thresholds, loss weight, sampling, or checkpoint
after observing this result.  A future A5 candidate must be separately
pre-registered, preserve the same leak-free deploy-aligned decision path, and
demonstrate one narrow mechanism before consuming replication or cross-fold
budget.
