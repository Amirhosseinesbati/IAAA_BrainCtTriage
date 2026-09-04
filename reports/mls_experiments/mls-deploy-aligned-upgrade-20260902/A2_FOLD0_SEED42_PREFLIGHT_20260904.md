# A2 signed-geometry — fold-0 / seed-42 preflight record

## Immutable experiment identity

- Candidate: `mls-vast-da-a2-signed-geometry-fold0-seed42`.
- Scientific change: `signed_offset_loss_weight=0.10` only; production
  keypoint decoding and study pooling are unchanged.
- Fixed audit state: epoch 15, 70 held-out fold-0 studies, CUDA-only inference.
- Manifest SHA-256:
  `59b688378a36cb78dd6280c817ffb0e44bb0782a71d9008af490ce32bdebc666`.
- Local implementation commits, in order:
  `a618a7b` (feature/tests/reports), `1b25c50` (materialized run), and
  `fa17691` (direct-server provenance tag).
- GitHub push was unavailable because the local non-interactive environment
  has no GitHub credential. The exact six non-sensitive commit files were
  directly synchronized only after their server base blob hashes matched, and
  the resulting two source blobs were verified as
  `d26d125ab80d1746aafd6ec9e76c6ed969e320ee` and
  `73f7c6b8b63d25337de64bc0f8c0e59ee9be3296`.

## Server readiness

- Host: Vast RTX 3090, CUDA available through the project uv environment.
- Data: `Data/processed` in the clean worktree is a symlink to the transferred
  canonical workspace; `mls_labels_multitask.csv` has 3,485 lines (header plus
  3,484 contract rows).
- Disk before launch: 62.61 GiB free.
- GPU compute processes: none; campaign lock and A2 artifact root: absent.
- MLflow: five allowlisted DagsHub variables only were installed in the
  root-only server secrets file (mode `600`); read-only MLflow connectivity
  passed. No unrelated local `.env` values were copied.

## CUDA memory preflight

Synthetic forward/backward smoke under strict deterministic settings passed:

| Batch | Peak allocated VRAM |
|---:|---:|
| 8 | 6.655 GiB |
| 12 | 9.993 GiB |
| 16 | 13.254 GiB |

The first scientific A2 resource screen deliberately retains batch size 5 and
FP32 to isolate the loss intervention from optimizer changes. The 3090 still
provides substantially faster CUDA execution. A larger batch may only be used
in a separately registered later replication.

## Launch and monitoring

The durable launcher started at `2026-09-04T02:30:53Z` in tmux session
`mls_da_a2_f0_s42`, writing to
`/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a2_fold0_seed42`.
It owns the global GPU lock, snapshots the manifest bytes before training, and
writes terminal status independently of the tmux session. Monitoring is
milestone-only: no run-log, tqdm, or capture-pane inspection while active.

## Resource-screen stop rule

Only this fold-0/seed-42 run is authorized now. It may authorize two more
fixed seeds only if its epoch-15 full-study CUDA audit is non-inferior to the
locked seed-42 reference on MAE, F1@3, F1@5, Boundary-F1, and selection
objective, as specified in `A2_SIGNED_GEOMETRY_PREREGISTERED_PLAN_20260904.md`.
It cannot create a submission or promotion claim on its own.
