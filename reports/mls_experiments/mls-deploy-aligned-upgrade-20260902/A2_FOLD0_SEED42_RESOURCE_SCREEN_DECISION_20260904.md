# A2 signed-geometry — fold-0 / seed-42 resource-screen decision

> Comparator correction (2026-09-04): the historical resource screen mixed
> candidate 0.5/top3/p90 with baseline checkpoint pooling. Preserve its numerical
> record, but use `ALIGNED_CACHED_RESOURCE_CORRECTION_RESULT_20260904.md` for
> the valid same-pooling comparison. The old rejection alone did not establish
> matched-deployment inferiority. No promotion is authorized by this annotation.

## Decision

**Rejected.** `mls-vast-deploy-aligned-a2-signed-geometry-fold0-seed42` may
not start seeds 2026 or 3407, expand to another fold, enter a triage-promotion
comparison, or create a submission ZIP. This is a scientific rejection of the
pre-registered A2 intervention—not an infrastructure failure.

## Immutable evidence

- Training launcher: `completed`, exit code `0`; start
  `2026-09-04T02:30:53Z`, finish `2026-09-04T03:18:09Z`.
- Fixed checkpoint: `mls_multitask_epoch_015.pth`; SHA-256
  `40b09e9d985e9aab46f27c6ac86e740aebdf97d81d8793eabebd90dd5c74a656`.
- CUDA audit: exactly one `epoch015` candidate, fold 0, 70 held-out studies,
  zero failures, CUDA-only; start `2026-09-04T03:33:13Z`, finish
  `2026-09-04T03:35:18Z`.
- Audit-status SHA-256:
  `09aa6f9cba0f688a69db57d0e884d89950caa1b99b0e5456a14c24e97286c0c3`.
- Aggregate metrics SHA-256:
  `c66894ef9473fee143846c7fec1ef815b102549af469c583951146d54ab25cb4`.
- MLflow run `5e421fd5159e4bbe870414fa8342f813` received the aggregate audit
  metrics and decision after the import-path fix. No raw per-study prediction
  CSV was logged or transferred.

## Fixed-profile result versus pre-registered resource gates

The only profile evaluated is the pre-registered selector threshold `0.5`,
top-3, `p90` study pooling profile. The reference values are the locked
fold-0/seed-42 baseline encoded before A2 training.

| Metric | Locked requirement | A2 result | Delta from requirement | Gate |
|---|---:|---:|---:|---|
| MAE (mm) | <= 1.470959 | 2.016416 | +0.545457 | fail |
| F1 @ 3 mm | >= 0.819672 | 0.787879 | -0.031793 | fail |
| F1 @ 5 mm | >= 0.736842 | 0.761905 | +0.025063 | pass |
| Boundary-F1 | >= 0.778257 | 0.774892 | -0.003365 | fail |
| selection objective | <= 1.904444 | 2.466632 | +0.562188 | fail |

The candidate also has RMSE `3.483170 mm` and bias `+0.560900 mm` on the
fixed end-to-end audit. It fails four of five resource gates. A lone F1@5
improvement is insufficient and is not a reason to relax the pre-registration.

## What the result does—and does not—show

At epoch 15, the inner validation proxy looked favorable (study MAE
`1.169988 mm`, Boundary-F1 `0.826458`, selection objective `1.563223`), while
the full, fixed 70-study CUDA audit was substantially worse. This is direct
evidence that the inner slice/validation proxy cannot select deployment-ready
MLS checkpoints for this intervention.

The signed-offset auxiliary loss was present and finite at epoch 15
(`train_signed_offset_loss=0.465289`, with pre-registered weight `0.10`). The
single controlled screen cannot prove that the sign target itself is malformed
or that a different weight would fail; it does prove that **this exact
signed-geometry implementation and weight do not improve deploy-aligned MLS**.
Accordingly, no A2 weight sweep, alternate epoch, pooling change, or extra
seed may be used to rescue it.

## Next scientific boundary

Future MLS work must begin from a newly pre-registered hypothesis and retain
the same full-study CUDA resource screen. It must not treat the favorable inner
validation values above as evidence for promotion. The current A2 epoch-15
checkpoint remains on the server with its recorded checksum for reproducible
forensics, but it is not a validated model and is deliberately not transferred
to the local `checkpoint/mls` release area.
