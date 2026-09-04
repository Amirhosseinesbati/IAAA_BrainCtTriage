# A3 fold-0 / seed-42 resource-screen decision

> Comparator correction (2026-09-04): the historical resource screen mixed
> candidate 0.5/top3/p90 with baseline checkpoint pooling. Preserve its numerical
> record, but use `ALIGNED_CACHED_RESOURCE_CORRECTION_RESULT_20260904.md` for
> the valid same-pooling comparison. The old rejection alone did not establish
> matched-deployment inferiority. No promotion is authorized by this annotation.

## Decision

**Rejected — stop A3 expansion.**  The fixed epoch-15 A3 checkpoint completed
the CUDA-only 70-study fold-0 audit, but failed every pre-registered resource
gate.  It cannot start the two additional A3 seeds, cross-fold training,
pooling/threshold rescue, an ensemble, promotion, or a submission ZIP.

This decision is about the study-bag auxiliary intervention, not about a
generic inability to improve MLS.  It is a valid negative result that narrows
the next scientific hypothesis.

## Immutable execution evidence

- Run: `mls-vast-da-a3-study-bag-fold0-seed42`; 23 epochs completed with
  launcher exit code `0` on the RTX 3090.
- Training interval: `2026-09-04T04:25:06.570522+00:00` to
  `2026-09-04T05:08:35.350320+00:00`.
- Fixed checkpoint: epoch 15, SHA-256
  `8693ecfc05e427959ab800223eebb8fab9f9598c43eb8192434dd663887db04f`.
- Inference audit: CUDA-only, RTX 3090, fold 0, exactly 70 held-out studies,
  zero candidate inference failures, fixed selector gate `0.5`, top-3, p90,
  and no checkpoint search.
- MLflow run: `0be09e666c5940ed890a11a714ef41a7`; it contains aggregate gate
  evidence only.  Per-study predictions remain private on the server.

## Gate results

| Metric | Locked requirement | A3 observed | Result |
|---|---:|---:|---|
| MAE | <= 1.470959 mm | 2.221690 mm | fail |
| F1 @ 3 mm | >= 0.819672 | 0.739726 | fail |
| F1 @ 5 mm | >= 0.736842 | 0.681818 | fail |
| Boundary F1 | >= 0.778257 | 0.710772 | fail |
| Selection objective | <= 1.904444 | 2.800145 | fail |

The result also regresses against rejected A2 on the same fixed resource
screen: A3 has +0.205274 mm MAE and -0.064120 boundary F1 relative to A2
(A2 MAE 2.016416 mm; boundary F1 0.774892).

## Aggregate-only error analysis

The diagnostic contains no identifiers or individual values.  Its key pattern
is bidirectional rather than a removable global offset:

- Low truth values are strongly overestimated: the <1-mm stratum (32 studies)
  has MAE 1.911739 mm and mean signed error +1.911739 mm; the 1--<3-mm stratum
  (7 studies) has MAE 3.077679 mm and bias +2.453782 mm.
- Severe values remain underestimated: 5--<10 mm has bias -1.152470 mm and
  >=10 mm has bias -1.631229 mm.
- At 3 mm there are 15 false positives and 4 false negatives; at 5 mm there
  are 9 false positives and 5 false negatives.  Only 38.57% of studies are
  within 1 mm.

The intended effect of the positive-study softmax bag loss was to reduce
spurious high local estimates in low-MLS studies and prioritize truly severe
slices.  The aggregate results point in the opposite direction: false-positive
pressure increased in low-MLS strata while high-MLS underestimation persisted.
This does not prove a single causal mechanism, but it decisively rejects this
weighting/selection formulation as a resource-worthy A3 continuation.

## Operational reconciliation

The first audit launch refused before GPU work because the existing runner
looked for the aggregate training report only in the clean worktree; the
trainer had written it to the explicit canonical workspace.  Runner fix
`1969a21` accepts exactly one of those known report locations and refuses an
ambiguous pair.  After that fix, CUDA inference and the aggregate decision both
completed successfully.  A detached tmux exit left the launcher status stale;
the durable audit and decision artifacts were used to reconcile it to terminal
`completed_reconciled`.  Future runner fix `c1f0459` records evaluator failures
as terminal metadata rather than leaving a `running` status.

## Local aggregate artifact verification

Only the following server-generated aggregate artifacts were copied locally;
all corresponding SHA-256 values match.  No checkpoint or raw prediction CSV
was copied because the candidate is rejected.

| Local artifact | SHA-256 |
|---|---|
| `A3_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json` | `b60f04f6924e33ea4a14ba377f4017c8fcc42faf4cdf10018aa47334837321db` |
| `A3_FOLD0_SEED42_AGGREGATE_DIAGNOSTIC_20260904.json` | `03c897bacfa0ae655b68982b3874db20d9ef25803b03b0d5fbb4e8f5f7e23ac1` |
| `A3_FOLD0_SEED42_CUDA_AUDIT_STATUS_20260904.json` | `a49f617cae5ca4aac0c7cb18f87020a29c5d554f1c18e3a2a8d4670bcb869a91` |
| `A3_FOLD0_SEED42_CUDA_AUDIT_METRICS_20260904.json` | `1600b92acaeae7af342ff6a2de6426f0304c5dabdb427a20bf3dc67e9589a53c` |
| `A3_FOLD0_SEED42_CUDA_AUDIT_LAUNCHER_RECONCILIATION_20260904.json` | `6f80bdc7e5a1308dbbbaf2353d857e59c31fbfb5c29630a04f59219b3db6ab2c` |

## Next research constraint

Do not revisit the rejected study-balanced/hybrid samplers or this A3
attention-weighted bag-loss formulation.  Any next MLS candidate must be
pre-registered before training, preserve the deploy-aligned audit, and address
the observed non-monotonic error shape without a post-hoc pooling or threshold
search.  It must again earn fold-0 resource-screen passage before any
additional seeds or folds are authorized.
