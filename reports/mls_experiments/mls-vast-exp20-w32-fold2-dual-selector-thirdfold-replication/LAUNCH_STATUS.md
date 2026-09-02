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
