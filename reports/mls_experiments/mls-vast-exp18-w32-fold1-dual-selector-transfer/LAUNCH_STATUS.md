# Exp18 launch status

## Preflight evidence

- Server commit: `441eba6f699ccbc07bc958116571bbaf179001b9`.
- CUDA smoke: RTX 3060, dual logits shape `[1, 2]`, finite forward/loss/backward,
  nonzero finite gradient for both selector rows, peak VRAM 0.936 GiB.
- Backward compatibility: the modified pooling code reproduced Exp09 epoch15
  frozen MAE `1.258664866792622` with absolute delta `0.0`; Boundary-F1 remained
  `0.823728813559322`.
- Dataset: 3484/3484 resolved rows, 338 studies, 1781 target and 1703 nontarget
  rows, complete positive spacing and study truth; no model was loaded by the
  validator.
- MLflow: remote `dagshub.com` connectivity probe succeeded using the root-only
  secrets file without printing credentials.
- GPU immediately before launch: RTX 3060, 35 MiB used, 0% utilization.
- Workspace disk immediately before launch: 14 GiB free.
- No pre-existing Exp18 status, run lock or tmux session existed.

## Durable launch

- Run: `mls-vast-exp18-w32-fold1-dual-selector-transfer`.
- Session: `mls_exp18_fold1_dual`.
- Started UTC: `2026-09-02T07:26:28.361938+00:00`.
- MLflow run ID: `18474f1d10234ca5900caefe3f62c2eb`.
- Durable status:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp18-w32-fold1-dual-selector-transfer/status.json`.
- Durable log:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp18-w32-fold1-dual-selector-transfer/train.log`.
- Auto-destroy: false; the instance must not be stopped or destroyed without
  user coordination.

The initial live check found state `running`, the expected tmux session,
MLflow system monitoring active, one CUDA process using about 5.3 GiB, GPU
utilization 100%, finite losses and throughput about 2.7 batches/s during
epoch 1. No CPU model fallback is allowed.

The next live check confirmed two complete, durable epoch rows and seven
nonempty selection/recovery checkpoints, including the new best-peak-AUC
checkpoint. Exp18 was actively training epoch 3. Epoch 2 remained an early
warm-up point: presence AUC 0.5694, peak AUC 0.4089, keypoint MAE 26.23 px and
slice MLS MAE 17.01 mm; all values were finite. No metric decision is permitted
before the preregistered recovery gates and full-study CUDA audit.

## Terminal training evidence

- Training finished UTC: `2026-09-02T08:54:04.758225+00:00`.
- Durable state: `completed`; exit code: `0`; tmux training session: absent.
- Epoch history: complete `1..23`, with no NaN, OOM or model CPU fallback.
- Peak training VRAM: `4.647613 GiB`.
- Best online objective occurred at epoch 12: study MAE `0.8998149905 mm`,
  Boundary-F1 `0.8476621418`, objective `1.2549402529`, presence AUC
  `0.8991009080` and peak AUC `0.7960730707`.
- Final epoch 23 was weaker than epoch 12, confirming that last-checkpoint
  selection is not valid for this run.
- MLflow closed normally under run ID
  `18474f1d10234ca5900caefe3f62c2eb`; aggregate evaluation remains pending.

## Full-study CUDA audit launch

- Audit session: `mls_exp18_audit`.
- Started UTC: `2026-09-02T09:08:01Z`.
- Audit root:
  `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/end_to_end_checkpoint_audit`.
- Durable audit log:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp18-w32-fold1-dual-selector-transfer/audit/audit.log`.
- Twelve preregistered candidate files were verified present and SHA256-distinct:
  six named checkpoints (including the new `best_peak_auc`) plus epochs
  13/15/17/19/21/23. Therefore the full plan requires `12 x 67 = 804` genuine
  CUDA study-checkpoint evaluations; no duplicate inference can be removed.
- Immediately before audit: RTX 3060 idle, 12 GiB workspace disk free, MLflow
  remote probe passed and the pending-event queue contained zero rows.
- The first authoritative poll found `best_objective` running, 13/67 studies
  already logged, the tmux session present and live GPU utilization.
- After the CUDA audit, the same durable script runs the 6048-profile grid for
  every candidate, applies the frozen production gate, and uploads only
  allowlisted aggregate reports to the existing MLflow run. Raw per-study
  predictions and medical data are explicitly excluded from MLflow upload.

## Git synchronization note

The exact MLS implementation was committed independently on the server. The
main remote branch had advanced by nine commits while this server had two
local commits, so no force push or unsafe merge was attempted in the dirty
concurrent workspace. A safe push to a dedicated branch was attempted but the
server has no GitHub username/token. The same source files exist in the local
workspace; synchronization must preserve concurrent ICH/submission changes.
