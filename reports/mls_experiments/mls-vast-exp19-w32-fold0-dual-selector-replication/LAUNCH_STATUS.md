# Exp19 launch status

## Frozen purpose

Exp19 is an independent fold0 replication of Exp18. The normalized training
configuration differs from Exp18 in exactly one field: `fold`, from 1 to 0.
The primary post-training test was fixed before launch:

- baseline: Exp16 `best_selector_auc`/epoch16;
- challenger: Exp19 epoch21 only;
- mode: `regression_only`;
- blend: 90% Exp16 + 10% Exp19 `mls_mm`;
- locked production pooling unchanged;
- primary gate: MAE and Boundary-F1 no worse than Exp16 and objective improved
  by at least 0.01.

No other checkpoint, alpha or profile may rescue the primary test.

## Preflight evidence

- Server commit: `cb7ccf21d56b1331ce2b2c75630bd9cd1680bfa8`.
- Normalized Exp18/Exp19 training-config diff: exactly `fold: 1 -> 0`.
- Processed data contract: 3484/3484 resolved rows, 338 studies, 1781 target
  and 1703 non-target rows, complete positive spacing and study truth; validator
  loaded no model.
- Fold0 baseline CSV: 70 rows, 70 unique studies and zero recorded errors.
- CUDA smoke: RTX3060, selector shape `[1, 2]`, finite loss/backward, nonzero
  finite gradient for both selector rows, peak VRAM 0.936GB.
- Remote MLflow probe: `dagshub.com` reachable using the root-only 0600 secrets
  file without printing credentials.
- Before launch: RTX3060 at 35MiB and 0% utilization; workspace had about 12GB
  free; no Exp19 status, lock, checkpoint directory, epoch history or tmux
  session existed.
- Auto-destroy remains disabled; the Vast instance must not be stopped or
  destroyed without user coordination.

## Durable launch

- Run: `mls-vast-exp19-w32-fold0-dual-selector-replication`.
- Session: `mls_exp19_fold0_dual`.
- Started UTC: `2026-09-02T10:36:49.927841+00:00`.
- MLflow run ID: `5383a78d31bf4a79a5bf6aff3c086e8c`.
- Durable status:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp19-w32-fold0-dual-selector-replication/status.json`.
- Durable log:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp19-w32-fold0-dual-selector-replication/train.log`.
- Compute policy in report: model forward/backward/validation CUDA-only, no CPU
  fallback.

The first authoritative live checks found status `running`, tmux present,
MLflow system monitoring active, finite losses, throughput about 2.74 batches/s,
GPU utilization 98-100% and approximately 5.3GB used VRAM.

The post-training runner is now prepared at
`scripts/run_vast_exp19_primary_transfer_audit.sh`. It cannot overlap training:
it requires the training session to be absent, status `completed` with exit
code zero, CUDA-only policy, and exactly 23 finite epoch-history rows. It audits
only the preregistered epoch21 checkpoint. A scientific gate failure is retained
as a valid completed result and its aggregate report is still sent to MLflow.

## First completed epoch

Epoch1 completed durably with peak VRAM 4.647613GB and finite metrics:

- train loss: 4.744462;
- validation loss: 3.929777;
- presence AUC: 0.768325;
- peak AUC: 0.249551;
- keypoint MAE: 145.22px;
- slice MLS MAE: 70.96mm;
- online study MAE: 4.03331mm;
- online Boundary-F1: 0.0.

These are warm-up metrics and have no checkpoint-selection authority. Exp18 and
Exp16 already proved online metrics can rank full-study checkpoints incorrectly.
The run continues through the preregistered 23 epochs unless a genuine failure
occurs.

## Frozen transfer evaluator

`scripts/evaluate_mls_fixed_component_transfer.py` was tested locally with an
identity fixture and on the real Exp09/Exp18 server artifacts. The latter
reproduced MAE 1.248084723, Boundary-F1 0.831034483 and objective 1.586015757
exactly. It accepts only one baseline, challenger, component mode and alpha and
returns nonzero when the fixed gate fails. It performs no model inference.
