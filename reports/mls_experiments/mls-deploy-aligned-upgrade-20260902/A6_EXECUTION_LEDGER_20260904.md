# A6 execution ledger — 2026-09-04

## State and scientific scope

A6 fold0/seed42 training started on the target RTX 3090 at **10:37:31 UTC**.
No held-out A6 result exists yet. No improved model, promotion or submission
is claimed. This run tests peak-local training geometry; deployment DARK and
all five resource gates remain fixed. See the preregistered A6 plan.

Code/preregistration commit: `b40b308`. Normalized training configuration
differs from the baseline only in `training_geometry_decoder`. Batch 5,
workers 2, 23 epochs and fixed audit epoch 15 preserve the baseline ordinary
optimizer budget, unlike the earlier batch-10 rank trials.

## Execution safety and verification

All Python, tests and CUDA execution occurred on the server, not the user's
computer. Local operations were source edits, git recording, artifact transfer
and hashing. The two existing shared remote files were backed up under
`/workspace/iaaa_artifacts/server_source_backups/before_a6_b40b308`.
An initial overwrite was denied; complete read-only content comparison then
proved both remote files exactly matched the parent commit (normalizing line
endings), with no unrelated changes. Transfer proceeded only after that check.
All 11 A6 source/config/test/preregistration files matched SHA256 after SCP.

Target-server validation:

- Both bash launchers passed `bash -n`.
- `python -m unittest tests.test_mls_local_geometry_cuda tests.test_evaluate_mls_a6_fold0_resource_screen`: **10 tests passed**, 0.731 seconds test time.
- Tests cover exact legacy decoding, Gaussian coordinate scale, local-gradient
  support, distant secondary peaks, borders/flat maps, negative-only loss,
  one-factor normalized config and fail-closed aggregate resource decisions.
- Full synthetic HRNet-W32 forward, actual multitask loss, backward, finite
  gradient check, clipping and AdamW update passed with batch 5 on CUDA.
- Synthetic preflight peak allocated VRAM: **4.253228 GiB**; gradient norm
  **4.694874**; all updated parameters finite. This is not an accuracy result.
- Strict deterministic mode enabled, AMP disabled; no real data/checkpoint
  used in synthetic tests. Preflight used random initialization, not pretrained
  weights. Training retains the ordinary pretrained initialization.

## Evidence hashes

| Artifact | SHA256 |
|---|---|
| A6 manifest | `28689c91e9c94a30c1543dde119633a608ff3594e22b9a0ae928fc58ea71822e` |
| A6 preregistration | `06f68b073589222590ffcc6267016fa63875fd3b6c66b1d15072ec3c0acde844` |
| CUDA preflight JSON, remote and local | `118396254ac1436fe871bd5ca8a13c7b52471e1eb0b34de2bafbc84d2db0b97e` |
| Processed labels | `01512662b62bcaf484f99cb872c40e28e2cfb300adee60db40957db0d06001ad` |
| Competition folds | `d3c4640aec8fbfd8a912286bbf40ee39a7f48756c899cafcf8d976ce664ce2b8` |
| config_models.py | `bc4a2263a02f69143d903d4de41dfa1740b782500b8bf8ddb7d4c0b39eeee089` |
| train_multitask.py | `5135d38b193a59079c763ea8003e66e7d5da1a81be94355b82fead1c4eb81cc3` |
| geometry_decoding.py | `984b0cc0da1ea78d188d09dbf2e095b63cfe5fdf7a8a047e7d4061bf7d11ebb5` |

Local preflight: `A6_CUDA_PREFLIGHT_20260904.json` beside this ledger.
Remote preflight: campaign root plus `a6_fold0_seed42_cuda_preflight.json`.

## Verified launch and next milestone

Server project: `/workspace/IAAA_BrainCtTriage_mls_da`.
Campaign root: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902`.
Training run: `mls-vast-da-a6-local-geometry-fold0-seed42`.
Tmux session: `mls_da_a6_f0_seed42`; launcher PID 35487.
At **10:37:57 UTC**, status was running with the matching manifest hash,
and Python PID **35512** was alive and present in NVIDIA's compute process
list, using **5504 MiB**. No epoch/log/validation output was inspected.
Training status is `<campaign>/<run>/status.json`.

The launcher sources the existing MLflow secrets file (permissions verified
600), uses the established training tracker, and disables automatic server
destruction. Actual A6 MLflow receipt/run ID still requires metadata or terminal
verification; file presence is not claimed as proof of remote logging.

Subsequent server-side metadata-only query confirmed exactly one A6 run in
MLflow experiment 16: `9b8e9fc5996a42549e3aca5aa40763d7`, status `RUNNING`,
start_time `1788518265226`. No intermediate metrics were inspected. Final
metric/artifact delivery still needs terminal verification.

Heartbeat `mls-a6-vast-milestone-monitor` is active every 30 minutes, quiet on
unchanged state and constrained to the fixed training/audit sequence below.

Next check is a 30-minute terminal milestone, approximately **11:08 UTC**.
Do not inspect intermediate epoch metrics or launch another GPU workload.
After training exits 0 and its process ends, run only
`scripts/run_vast_mls_a6_fold0_seed42_resource_screen.sh` in session
`mls_da_a6_f0_resource`, auditing epoch15 on all70 fold0 studies.
Decision: report directory plus `A6_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json`.
Audit directory: `<campaign>/a6_fold0_seed42_cuda_audit`.
Any failed gate stops A6 expansion; a pass only supports a separately
preregistered replication. Full triage promotion rules remain unchanged.

## Training terminal milestone and shorter monitoring cadence

At the user's requested check, **2026-09-04 11:22:48 UTC**, training status was
`completed`, exit code 0. It records completion at **11:21:17 UTC**, duration
**43 minutes 46 seconds**. The training tmux session and PID35512 were absent,
and NVIDIA reported no compute process. Manifest SHA256 still matched.
No training logs or intermediate metrics were opened. This proves normal
execution completion, not improved held-out accuracy.

The user requested shorter timing on the 3090. This was interpreted explicitly
as monitoring frequency, reduced from 30 to **15 minutes**; training epochs,
fixed checkpoint and scientific acceptance gates were not changed. This cadence
supersedes the earlier monitoring intervals in this ledger and the conditional
continuation document, without changing their scientific protocol.

The fixed epoch15/all70-study resource-screen launcher was dispatched only
after the clean training termination and absence of an existing A6 audit.
