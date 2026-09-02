# Exp20 preflight and launch status

## Frozen purpose

Exp20 is the third-fold replication of the fixed dual-selector regression
component recipe. The normalized training configuration differs from Exp19 in
exactly one field: held-out `fold`, from 0 to 2. The primary post-training test
is fixed before launch:

- baseline: promoted Exp15r `epoch017`;
- challenger: Exp20 `epoch21` only;
- component mode: `regression_only`;
- blend: 90% Exp15r plus 10% Exp20 `mls_mm`;
- Exp15r retains selector, peak/ranking probability and heatmap;
- no alpha, checkpoint, threshold or pooling retune.

The required gates over all 67 fold2 studies are MAE no worse than
`1.5483543317709396mm`, Boundary-F1 no worse than `0.8925925925925926`, and
objective at most `1.7531691465857544`.

## Preflight evidence

- Manifest, plan and audit runner SHA-256 values match between local and Vast.
- `bash -n` passed for `scripts/run_vast_exp20_primary_transfer_audit.sh`.
- Both Exp19 and Exp20 manifests validate through `MLSHeatmapConfig`; their
  normalized training-config diff is exactly `{'fold': [0, 2]}`.
- Processed MLS data validated without loading a model: 3484/3484 resolved
  rows, 338 studies, 1781 positive and 1703 negative rows, complete positive
  spacing and study truth.
- Exp15r baseline predictions contain 67 studies from a completed CUDA-only
  audit with zero failures.
- Exp15r epoch17 checkpoint is present with SHA-256
  `e4c5f91c4e9fb97b766477615f6e42244bed2ee53f85c98f4f1353146cb6e16e`.
- Identity parity using the production evaluator reproduced baseline MAE
  `1.5483543317709396`, Boundary-F1 `0.8925925925925926` and objective
  `1.7631691465857544`; residual numerical deltas were at most `2.22e-16` under
  the fixed `1e-9` tolerance. This step used saved CUDA predictions and no
  model/image inference.
- A light CUDA tensor forward/backward completed on the RTX3060 with finite
  loss and gradient and peak allocation `0.004884GB`.
- The first ad-hoc CUDA smoke omitted the required host-driver library path and
  a second command suffered shell quoting; both failed before model/training
  execution. The corrected smoke used the project's host-libcuda workaround
  and passed. No Exp20 training state was created during these checks.
- Root-only secrets mode is `0600`; a read-only MLflow probe succeeded without
  printing credentials.
- Before launch there is no Exp20 training status, lock, checkpoint directory
  or tmux session. GPU is idle at 35MiB and the 40GB workspace has about 9.4GB
  free.
- The Vast instance retains `auto_destroy=false` and must not be stopped or
  destroyed without user coordination.

## Durable execution contract

Training must use `scripts/launch_vast_mls_tmux.py` with session
`mls_exp20_fold2_dual`. Model forward, backward and validation are CUDA-only;
CPU fallback is forbidden. The run must complete all 23 epochs unless a genuine
terminal failure occurs.

After successful terminal training, only epoch21 is audited. The durable audit
runner is `scripts/run_vast_exp20_primary_transfer_audit.sh`; it refuses to
overlap training, requires exact finite epochs 1 through 23, evaluates all 67
studies on CUDA, applies the frozen 90/10 gate and uploads only allowlisted
aggregate artifacts to the training MLflow run. A scientific gate failure is
retained as a completed result and cannot be rescued by another checkpoint.

## Durable launch

- Training commit: `cb9c3c4ad79eb1b05770a583050faad7c570a425`.
- Session: `mls_exp20_fold2_dual`.
- Started UTC: `2026-09-02T12:35:47.996130+00:00`.
- MLflow run ID: `aa4d88acea4246a8a7e5c27a0a33a6c6`.
- Durable status:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp20-w32-fold2-dual-selector-thirdfold-replication/status.json`.
- Durable log:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp20-w32-fold2-dual-selector-thirdfold-replication/train.log`.
- Initial live GPU check: 89-98% utilization and approximately 5.3GB used
  VRAM; tmux present and status `running`.

Epoch1 completed with finite metrics and peak VRAM `4.647613GB`:

- train loss `4.705378`;
- validation loss `3.578214`;
- presence AUC `0.629701`;
- peak AUC `0.320915`;
- keypoint MAE `38.768px`;
- slice MAE `19.0467mm`;
- online study MAE `4.881576mm`;
- online Boundary-F1 `0.0`.

These are warm-up diagnostics with no checkpoint-selection authority. Training
continued through the frozen 23 epochs without changing the recipe.

## Terminal training result

- Finished UTC: `2026-09-02T14:03:27.457082+00:00`.
- State: `completed`; exit code `0`; 23/23 finite epochs; tmux absent.
- Compute policy: CUDA-only with no model fallback to CPU.
- Peak allocated VRAM: `4.647613GB`.
- MLflow run: `aa4d88acea4246a8a7e5c27a0a33a6c6`, independently
  verified `FINISHED`.
- Primary epoch21 checkpoint SHA-256:
  `34f6a251fc0355ce4ef6ccc0a3bd5b977ab106de947e0ff51cd2ef320f3042de`.

## Frozen audit and post-failure result

The primary epoch21 audit completed 67/67 studies on the RTX3060 with zero
failures. The fixed 90/10 regression-only transfer improved MAE from
`1.548354332` to `1.533389346`, but Boundary-F1 fell from `0.892592593` to
`0.884848485` and objective worsened from `1.763169147` to `1.763692376`.
The primary gate therefore failed.

The preregistered alpha sensitivity screen found no eligible alpha. The one
allowed named-best checkpoint (epoch11) also completed 67/67 CUDA studies with
zero failures, but its fixed hybrid produced MAE `1.539041557`, Boundary-F1
`0.884848485` and objective `1.769344587`; it failed and checkpoint diagnostics
stopped. No Exp20 checkpoint was copied into the local release directory.

## Conservative three-fold OOF

The frozen fallback retained Exp15r unchanged on fold2 and used only the
independently passing 10% regression components on fold0 and fold1. Across 204
disjoint held-out studies it passed all aggregate gates:

- MAE: `1.472591075 -> 1.461521959` (`-0.011069116mm`);
- Boundary-F1: `0.850206612 -> 0.855888430` (`+0.005681818`);
- objective: `1.772177852 -> 1.749745100` (`-0.022432753`).

The paired 2,000-replicate bootstrap assigned `0.9775` probability to objective
improvement, with 95% delta interval `[-0.049041088, -0.000690376]`. Aggregate
artifacts and nine OOF metrics were uploaded to the Exp20 MLflow run; raw
study-level CSVs remained excluded. This is a package candidate, not yet a
leaderboard-proven release.

## Conservative package integration and terminal CUDA audit

The actual five-checkpoint submission package was built as
`/workspace/iaaa_artifacts/packages/iaaa_brain_ct_triage_mls_conservative_five_20260902.zip`.
It contains 42 stored files, is 812,453,997 bytes, and has SHA-256
`660770225b53e5389ba0e8dde70cc7e1a65f732ca854887aba6ba8deff1d490b`.

The extracted package runtime completed all 204 held-out studies on the RTX3060
with CUDA-only model forward and zero failures. All seven audit gates passed:
index, slice MLS, selector, peak-selector, heatmap, member aggregation and OOF
metric parity. The packaged result exactly reproduced MAE `1.461521959`,
Boundary-F1 `0.855888430` and objective `1.749745100`; maximum member aggregation
residual was `0mm`. Runtime was `552.441s` and peak MLS-only VRAM was
`0.927000GiB`.

Only aggregate audit artifacts and nine package metrics were uploaded to MLflow;
the raw per-study CSV remains on Vast. The package is internally accepted and
ready for a limited official leaderboard submission, but is not yet a
leaderboard-proven release. The Vast instance remains running and must not be
stopped or destroyed without user coordination.
