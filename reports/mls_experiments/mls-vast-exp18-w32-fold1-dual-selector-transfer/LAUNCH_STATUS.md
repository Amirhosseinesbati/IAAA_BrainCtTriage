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
- MLflow training closed normally under run ID
  `18474f1d10234ca5900caefe3f62c2eb`; the later aggregate audit upload also
  completed successfully with a zero-row pending queue.

## Full-study CUDA audit launch

- Audit session: `mls_exp18_audit`.
- Started UTC: `2026-09-02T09:08:01Z`.
- Audit root:
  `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/end_to_end_checkpoint_audit`.
- Durable audit log:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp18-w32-fold1-dual-selector-transfer/audit/audit.log`.
- Twelve preregistered candidate files were verified present and SHA256-distinct:
  six named checkpoints (including the new `best_peak_auc`) plus epochs
  13/15/17/19/21/23. File-level hash uniqueness does not by itself prove that
  every stored model state is different, because checkpoint metadata can also
  change the hash. The preregistered audit therefore preserves all 12 labels
  and runs `12 x 67 = 804` CUDA study-checkpoint evaluations.
- Immediately before audit: RTX 3060 idle, 12 GiB workspace disk free, MLflow
  remote probe passed and the pending-event queue contained zero rows.
- The first authoritative poll found `best_objective` running, 13/67 studies
  already logged, the tmux session present and live GPU utilization.
- After the CUDA audit, the same durable script runs the 6048-profile grid for
  every candidate, applies the frozen production gate, and uploads only
  allowlisted aggregate reports to the existing MLflow run. Raw per-study
  predictions and medical data are explicitly excluded from MLflow upload.

## Full-study CUDA audit terminal evidence

- CUDA audit finished at `2026-09-02T09:23:40Z`; the complete grid/gate/upload
  pipeline finished at `2026-09-02T09:33:09Z`.
- Coverage was `12 x 67 = 804/804` study-checkpoint evaluations with zero
  failures and the required `cuda_only_no_cpu_fallback` policy.
- Pooling produced 6048 profiles per candidate, or 72576 data rows total.
- The locked production profile selected epoch21 as Exp18's strongest
  checkpoint: MAE `1.392991942`, Boundary-F1 `0.771236`, objective `1.850521`.
- Required gates were MAE `<=1.2586648668`, Boundary-F1 `>=0.82`, and objective
  `<=1.6112072397`. No Exp18 candidate passed all gates; the durable decision is
  to retain Exp09/epoch15.
- Same-fold diagnostic tuning found MAE `1.127073515`, Boundary-F1 `0.809942`
  and objective `1.507190` for `best_objective`, but it remains below the
  boundary floor and is not an unbiased production estimate.
- Aggregate artifacts were uploaded to MLflow. Raw study prediction CSVs were
  excluded, and the pending-event queue ended at zero.
- No Exp18 checkpoint was copied into the local release-model directory.

## Cross-run component-screen preregistration

- Purpose: test whether Exp18 regression is complementary to the trusted Exp09
  gate/ranking path before spending GPU time on a new training run.
- Inputs: Exp09/epoch15, Exp18/epoch21 (best locked), and Exp18
  `best_objective`/epoch12 (best online).
- Four fixed modes progressively retain Exp09 selector, peak ranking, and
  heatmap; the final mode blends regression only.
- Challenger alphas are fixed at `0.10`, `0.25`, `0.50`, `0.75`, and `1.00`.
- The production profile and numerical promotion gates remain frozen. A second
  guarded profile is diagnostic only.
- Exp09 predictions are regenerated on CUDA on the existing server because
  transferring the local per-study prediction CSV was rejected as sensitive.
  The Exp09 checkpoint transfer was accepted and verified byte-for-byte:
  124890565 bytes, SHA256
  `98923f724b2d61c4a8671ef0405ab7c205913e0829b72c0e32ae317ef23cfccb`.
- Per-study CSVs stay on the Vast server. Only explicitly allowlisted aggregate
  grid, summary and Markdown report may be uploaded to MLflow.
- First execution correctly stopped before any blend because regenerated Exp09
  MAE was `1.2590358221` versus historical `1.2586648668` (delta
  `+0.0003709553 mm`). Boundary-F1 was exactly unchanged at `0.8237288136`;
  objective moved by the same `+0.0003709553`. The new CSV has the same 67
  studies and 1346 slices, while the larger byte size is explained by the new
  backward-compatible `peak_probability` field, exactly equal to selector for
  all 1346 Exp09 slices.
- A distinct cross-runtime parity tolerance of `0.001` is therefore fixed before
  observing any blend. It only guards baseline reproduction; historical release
  gates remain numerically unchanged.
- The first complete 82-row grid (41 candidates x two profiles) found seven
  frozen-gate-eligible blends, all at alpha `0.10`. For Exp18/epoch21, all four
  modes were numerically identical: MAE `1.248084723`, Boundary-F1 `0.831034483`,
  objective `1.586015757`. This proves the gain is already obtained by blending
  only `mls_mm`; selector/peak/heatmap changes do not contribute at this alpha.
- The original alphabetical tie selected `baseline_selector`. A deterministic
  minimum-complexity tie-break is applied after observing exact equality, so the
  diagnostic selected label becomes `regression_only` without changing any
  prediction or metric.

## Cross-run component-screen terminal evidence

- Exp09 re-audit completed `67/67` studies on CUDA with exit code zero and zero
  failures at `2026-09-02T10:15:26Z`.
- The final minimum-complexity screen and MLflow upload completed at
  `2026-09-02T10:25:49Z`; tmux session `mls_exp09_component_screen` is absent.
- Final selected diagnostic is
  `exp18_epoch21__regression_only__a0p1`: MAE `1.2480847228`, RMSE
  `1.9581349086`, Boundary-F1 `0.8310344828`, objective `1.5860157573`.
- Relative to the same-runtime Exp09 baseline, deltas are MAE
  `-0.0109510993 mm`, Boundary-F1 `+0.0073056692`, and objective
  `-0.0255624377`.
- Seven candidates passed all frozen numerical gates; all used alpha `0.10`.
  No alpha at or above `0.25` passed.
- The allowlisted grid, summary and Markdown report were uploaded to MLflow run
  `18474f1d10234ca5900caefe3f62c2eb`. All `study_slice_predictions.csv` files
  and `screen_selected_predictions.csv` were explicitly excluded.
- Five aggregate artifacts were synced locally and verified; the local grid has
  exactly 82 rows and the summary selects the regression-only candidate above.

## Git synchronization note

The exact MLS implementation was committed independently on the server. The
main remote branch had advanced by nine commits while this server had two
local commits, so no force push or unsafe merge was attempted in the dirty
concurrent workspace. A safe push to a dedicated branch was attempted but the
server has no GitHub username/token. The same source files exist in the local
workspace; synchronization must preserve concurrent ICH/submission changes.
