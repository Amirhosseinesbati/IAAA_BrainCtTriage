# A6 final resource decision: rejected

Training finished normally at 2026-09-04 11:21:17 UTC (43m46s). Fixed epoch15
CUDA audit finished at 11:25:44 UTC and its gate launcher at 11:26:02 UTC.
The terminal check at 11:26:46 UTC found no audit session or CUDA process.
All70 fold0 studies were evaluated, with zero inference failures.

| Fixed metric | Observed | Required | Decision |
|---|---:|---:|---|
| Study MAE mm | 2.5615802875 | <=1.4709586392 | fail |
| F1 at3mm | 0.6956521739 | >=0.8196721311 | fail |
| F1 at5mm | 0.6808510638 | >=0.7368421053 | fail |
| Boundary F1 | 0.6882516189 | >=0.7782571182 | fail |
| Selection objective | 3.1850770498 | <=1.9044444028 | fail |

Fixed DARK profile: selector0.5, top3, p90, epoch15. Study RMSE4.2210722471mm,
bias+0.9872868992mm. Combined triage Macro-F1 is null: no final triage gain
was established or inferred from these MLS metrics.

Decision: `rejected_stop_a6_expansion`. Both additional seed replications and
cross-fold expansion are closed under the preregistered contract. No checkpoint,
pooling or threshold search is allowed to rescue A6; no release/submission ZIP.
The audit metrics JSON contains inherited in-sample profile diagnostics. They
were exposed when reading that file, but are exploratory only and are not used
to override the fixed-profile rejection or select a deployment configuration.
Future reports should project only fixed-profile fields to avoid this exposure.

## Interpretation and remaining uncertainty

This is negative evidence for the tested local-geometry intervention at the
fixed budget. Matching batch5 and schedule removed the earlier batch10 budget
confound; it does not establish a unique cause for A6's failure. Passing numerical
tests proves finite execution, not favorable optimization or generalization.
Potential mechanisms include committing geometry gradients to an incorrect peak,
mode switching and the local expectation still differing from DARK. The aggregate
study result cannot distinguish these from selector/pooling effects.

Before any new training, a training-only postmortem will compare baseline and
A6 on the same fixed128 positive training slices used in the earlier diagnostic.
Its separate protocol is frozen before running it. A6 remains rejected regardless
of that diagnostic's outcome; the diagnostic is not an additional validation gate.

## Checkpoint and MLflow provenance

Checkpoint remains on server (not promoted or copied as a best model):
`/workspace/IAAA_BrainCtTriage/models/checkpoints/mls_multitask/mls-vast-da-a6-local-geometry-fold0-seed42/mls_multitask_epoch_015.pth`.
SHA256: `88b094341b260d48c90a2a1e12772c5bd5d82ac898e509db7ed7762d0b44aec6`.

[MLflow run](https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/9b8e9fc5996a42549e3aca5aa40763d7)
was read back as FINISHED, with all five matching resource metrics, all gates0
and tag `rejected_stop_a6_expansion`. No private predictions were read or copied.

## Local/server aggregate preservation

Local directory: `server_aggregate/a6_seed42_resource_screen_20260904/` beside
this report. All five transferred files matched the server SHA256:

| File | SHA256 |
|---|---|
| decision.json | `5bf7ef86aff030b7aa293f94ae34e8f91afd2fef7aa06b1eea83121ba5196212` |
| audit_status.json | `d586300797f1f4b8449d0398262966ffa590c0a7d26964cea71b5699802112c9` |
| audit_launcher_status.json | `b357ab7319fe94f299002384ea4bfa1b29db02401d6220431f302cc41da5cf04` |
| metrics.json | `be6d8de71a995684c0d1648730934c220ed40d62a34a9ac1e5130bfa378b182d` |
| training_status.json | `56df47a353d75a82c9322a5fa1d39f537bd6508d734367cacb312c90e8c37507` |

The 15-minute heartbeat was paused at the user's request and remains paused.
No replacement schedule was created. The server and overarching goal remain
active; no improved MLS model has yet been established by A6.
