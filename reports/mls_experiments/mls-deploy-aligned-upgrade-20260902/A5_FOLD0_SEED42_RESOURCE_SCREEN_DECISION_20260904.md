# A5 terminal decision: rejected

> Comparator correction (2026-09-04): the historical resource screen mixed
> candidate 0.5/top3/p90 with baseline checkpoint pooling. Preserve its numerical
> record, but use `ALIGNED_CACHED_RESOURCE_CORRECTION_RESULT_20260904.md` for
> the valid same-pooling comparison. The old rejection alone did not establish
> matched-deployment inferiority. No promotion is authorized by this annotation.

## Outcome

The fixed epoch-15 CUDA audit completed successfully on all 70 fold-0 studies,
with zero inference failures. A5 failed all five preregistered resource gates.
Decision: `rejected_stop_a5_expansion`. No A5 seed replication, cross-fold run,
pooling/threshold change, promotion, packaging, or submission follows this
result. The current released champion is unchanged.

| Metric | A5 observed | Required | Pass |
|---|---:|---:|---|
| Study MAE, mm | 1.9533821280 | <= 1.4709586392 | no |
| F1 at 3 mm | 0.7323943662 | >= 0.8196721311 | no |
| F1 at 5 mm | 0.7317073171 | >= 0.7368421053 | no |
| Boundary F1 | 0.7320508416 | >= 0.7782571182 | no |
| Selection objective | 2.4892804447 | <= 1.9044444028 | no |

The fixed profile remains selector threshold 0.5, top-k 3, p90 aggregation.
Its RMSE is 2.5818873392 mm and signed bias is +0.5558038632 mm.
`combined_macro_f1` is null: this screen provides no final triage Macro-F1 or
Urgent-class improvement evidence and must not be presented as leaderboard
validation.

## Interpretation relative to A4

A4 used the same fold, seed, fixed epoch and evaluation profile. A5 changed
only the rank auxiliary's gradient path, retaining selector gradients while
detaching the backbone and preserving training-mode normalization.

| Metric | A4 | A5 | A5 minus A4, approximately |
|---|---:|---:|---:|
| Study MAE, mm | 1.878737 | 1.953382 | +0.074645 |
| F1 at 3 mm | 0.794521 | 0.732394 | -0.062126 |
| F1 at 5 mm | 0.697674 | 0.731707 | +0.034033 |
| Boundary F1 | 0.746098 | 0.732051 | -0.014047 |
| Selection objective | 2.386542 | 2.489280 | +0.102739 |

The 5-mm improvement comes with worse 3-mm F1, MAE and combined objective.
This does not support isolated rank-backbone gradients as a sufficient remedy
for the A4 failure. One seed cannot establish the precise causal mechanism or
rule out every possible ranking approach. It does establish that this frozen
A5 recipe fails the agreed resource screen and should not consume replication
resources. Future work needs a distinct, evidence-backed hypothesis; changing
this run's checkpoint, pooling or gates after observing the result is not a
valid rescue.

## Execution and tracking

- Training completed with exit code 0 at `2026-09-04T08:30:29Z`, after
  40 minutes 55 seconds.
- CUDA audit ran from `2026-09-04T09:11:47Z` to `2026-09-04T09:13:56Z`.
- Resource launcher completed with exit code 0 at `2026-09-04T09:14:27Z`.
- Terminal state was observed at `2026-09-04T09:42:52Z`; the audit tmux session
  was absent and no CUDA compute process was listed.
- MLflow reports `logged` for run `6983886c0bea419696087a39cc6c8478`.
- The corrected decision has replication permission, promotion eligibility,
  and submission permission all explicitly false.
- The completed A5 heartbeat monitor `mls-vast-milestone-monitor` was deleted.
  This did not stop or destroy the server. The overall MLS goal remains open.

Checkpoint retained on the server (not copied to the local champion folder):
`/workspace/IAAA_BrainCtTriage/models/checkpoints/mls_multitask/mls-vast-da-a5-detached-rank-fold0-seed42/mls_multitask_epoch_015.pth`.
SHA-256: `8be0c30b7b089ff83e8d7813a5920128228ea4c90dec0d2ac6eb12bd20289ad2`.

## Local aggregate evidence

Five final JSON files were copied to
`server_aggregate/a5_seed42_resource_screen_20260904/` alongside this report.
Every local SHA-256 matched its remote counterpart. No private per-study
prediction file or checkpoint was transferred.

| File | SHA-256 |
|---|---|
| `A5_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json` | `91264ea712cccdbdefaf9c60478864f802173cdbad3f6cac502e5a86d1817e60` |
| `audit_status.json` | `58658ac12854e3272ee36229db51514be9f1d606bbe587a10b5ed884627bb4bf` |
| `audit_launcher_status.json` | `c528cdb5bd129925f48187ee9078ac4e5293e27320b9db5f475ebfc6d8d2949c` |
| `metrics.json` | `138e2e8507c99ee295f19487ab463b89b7050f6874008b14d80c530a5ad2173b` |
| `status.json` | `a83c1d6f22f1605e99c04262c79807518a60c2f25bc36a0e99cda2f22be5e192` |
