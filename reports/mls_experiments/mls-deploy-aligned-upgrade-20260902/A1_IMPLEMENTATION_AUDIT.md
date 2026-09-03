# A1 ordinal MLS implementation audit

## Scope

This audit verifies that the A1 experiment is an MLS-improvement experiment, not
an unrelated workload, and that its training-only ordinal head can influence the
deployed MLS path without changing the historical inference contract.

## Static implementation findings

- `HRNetHeatmapModel.forward_multitask_extended()` sends the same non-detached
  `feat_1_4` backbone tensor to the heatmap, selector, and ordinal heads.
- `multitask_loss()` applies the ordinal loss only to target slices with all
  keypoints valid. Its labels are the preregistered official boundaries
  `MLS >= 1/3/5 mm`, with weights `0.75/1.0/1.25` and a monotonic-order penalty.
- The weighted ordinal term is part of the scalar loss before AMP scaling and
  `.backward()`. Therefore it updates both the ordinal head and the shared
  backbone; there is no `detach()` on this path.
- Production inference intentionally calls the historical
  `forward_multitask()` and consumes only heatmap and selector outputs. The
  ordinal head is training-only; the deployed MLS value remains derived from
  decoded keypoints.
- The inference loader reconstructs the ordinal architecture from checkpoint
  config and uses `load_state_dict(..., strict=True)`. Missing or unexpected
  ordinal parameters fail closed rather than being silently ignored.
- Resume validation includes every ordinal model/loss field and strict state
  loading, preventing an A1 run from resuming from an incompatible baseline.
- A tiny contract test now exercises the real `forward_multitask_extended()`
  method with a synthetic shared backbone and proves that ordinal-only
  backpropagation produces a non-zero gradient in that backbone. The complete
  ordinal suite passed 7/7 tests via `python -m unittest` in 0.575 seconds on
  2026-09-03; no dataset, production backbone, or training workload was loaded.

Conclusion: no implementation defect or inference-contract drift was found in
the reviewed A1 gradient path. The scientific hypothesis remains unproven until
fixed-epoch, three-seed, deploy-aligned evaluation is complete.

## Three-seed manifest parity

An exact textual comparison on 2026-09-03 confirmed that the seed-42, seed-2026,
and seed-3407 A1 manifests differ only in `run_name`, the provenance `seed` tag,
and `training_config.seed`. Fold, data contract, architecture, losses, training
schedule, aggregation, and fixed audit epoch are identical.

- seed 42 SHA-256: `1aa878e08b5d887e4a5d44a032dd5c320ed16d08d474e16757448b6633f116d1`
- seed 2026 SHA-256: `802d8b7e1893a2a29b9a52254967490e5626ea8cbbeeab23813816e0d4d6a25`
- seed 3407 SHA-256: `f1c58b921e13ab42e72b36cc1e20bca62c434ef48a966442af40c47d31ea8485`

This closes accidental configuration drift as a confounder in the preregistered
three-seed median comparison.

## Runtime milestone snapshot

- Snapshot time: `2026-09-03T00:15:16Z`
- Vast instance: `49527185`; GPU: NVIDIA GeForce RTX 3060 12 GiB
- Session: `mls_da_a1_f0_s42` present
- Active job wrapper: `mls-vast-da-a1-ordinal-fold0-seed42`
- GPU state: 100% utilization, 5,339 MiB used, 57 C
- Completed epoch rows: not measured by this snapshot. The initial probe looked
  under the external artifact root, while the trainer's canonical history is
  `reports/mls_experiments/<run>/epoch_metrics.jsonl` inside the project. Absence
  at the non-canonical path is not evidence that zero epochs had completed.
- Free workspace bytes: 3,267,231,744

No transfer, second heavy job, checkpoint selection, or adaptive threshold
selection was performed during this snapshot.

### Epoch-11 runtime milestone

- Snapshot time: `2026-09-03T00:43:26Z`
- Session: `mls_da_a1_f0_s42` present; the seed-42 training process was live
- GPU state: 87% utilization, 5,339 MiB used, 58 C
- Canonical metric history: 11 completed epoch rows
- Latest completed epoch: 11
- Epoch-11 study MAE: `1.2403016553` mm
- Epoch-11 study F1 at 3 mm / 5 mm: `0.8666666667` / `0.85`
- Epoch-11 study boundary-F1: `0.8583333333`
- Epoch-11 selection objective: `1.5754557169`
- Fixed epoch-15 checkpoint: not yet present
- Free workspace bytes: `3,266,228,224` (93% filesystem use)

These epoch-11 figures are an encouraging training diagnostic, but they are not
eligible for model selection or an A1-vs-baseline claim. The preregistered screen
remains the fixed epoch-15 median over seeds 42, 2026, and 3407, followed by the
deploy-aligned triage gate. No new run, transfer, deletion, checkpoint selection,
or threshold tuning was performed at this milestone.

### Fixed epoch-15 seed-42 materialization

- Snapshot time: `2026-09-03T01:02:23Z`
- Training remained live and had completed 16 epoch rows
- Fixed checkpoint bytes: `124,934,257`
- Fixed checkpoint SHA-256:
  `7f4b4daee6935618ed2740464036971ae665b7acf798bf1220e34e066892aab8`
- A1 study MAE: `1.3684332163` mm
- A1 study F1 at 3 mm / 5 mm: `0.8771929825` / `0.7179487179`
- A1 study boundary-F1: `0.7975708502`
- A1 selection objective: `1.8242306564`
- Free workspace bytes: `3,140,231,168`

The exact same-seed, same-fold, same-epoch control is historical Exp16. A1 minus
control deltas are: MAE `+0.2103315105` mm (worse), F1 at 3 mm
`+0.0297353553` (better), F1 at 5 mm `-0.0820512821` (worse), boundary-F1
`-0.0261579634` (worse), and selection objective `+0.2675943090` (worse).
This is a materially negative single-seed diagnostic with a narrow 3-mm gain,
not a valid A1 rejection decision. The locked protocol requires fixed epoch-15
median inference over all three seeds before the deploy-aligned screen is
consumed. Training therefore continues without adaptive checkpoint selection,
threshold tuning, or architecture changes.

## Disk-safety gate for subsequent seeds

The transferred baseline fixed-epoch checkpoints are 124,898,853 bytes each,
while a completed-run optimizer resume checkpoint is approximately 354,860,303
bytes. The trainer can retain six rolling best checkpoints, the fixed epoch-15
checkpoint, a final checkpoint, and one resume checkpoint. A conservative
per-seed requirement is therefore about 1.35 GB before logs and temporary
atomic-write headroom; A1 is marginally larger because of its ordinal head.

The snapshot above had 3,267,231,744 free bytes. This is sufficient for the
already-active seed but is not evidence that either later seed can be launched
safely. The authoritative launcher itself requires 4 GiB free by default, so
post-run reclamation must restore at least that threshold; it must not be lowered
to fit the remaining disk.
Before starting seed 2026 or seed 3407, the terminal run must pass this sequence:

1. inventory every checkpoint and record SHA-256 for fixed epoch 15;
2. verify launcher/MLflow terminal state and artifact upload;
3. preserve the preregistered epoch-15 checkpoint and reports;
4. only then reclaim completed-run optimizer state and non-fixed artifacts that
   are both durably uploaded (when applicable) and explicitly ineligible for the
   fixed audit;
5. require at least the launcher's 4 GiB preflight threshold for the next run.

No active-run checkpoint may be removed, and this audit does not authorize any
deletion before terminal-state verification.

## Post-run transfer-chain verification

On 2026-09-03, the existing fail-closed launcher and checksum-transfer suites
passed 11/11 tests in 0.590 seconds using the standard-library unittest runner.
The covered contracts include: refusal of failed or incomplete runs, mandatory
fixed epoch 15, complete epoch history, immutable manifest SHA, rejection of a
wrong or incomplete transfer manifest, the campaign-wide GPU lock, the minimum
disk preflight, and CUDA-only/no-CPU-fallback launcher behavior. No model or
dataset workload was executed.

Terminal artifacts must therefore be packaged with
`scripts/build_mls_run_transfer_manifest.py` and verified after transfer with
`scripts/verify_mls_run_transfer.py`; ad-hoc copies are not sufficient evidence.

## Audit and promotion-chain verification

The current branch passed 22/22 metadata-only decision tests in 0.542 seconds on
2026-09-03. They cover three-seed recipe parity, checkpoint-hash-bound resume,
all official 1/3/5 mm boundaries, immutable fold membership, recomputation of
stored seed medians, binding private predictions to their aggregate audit,
strict (not tied) Urgent-F1 improvement, refusal of single-fold promotion, and
authorization only from checksum-bound full five-fold OOF evidence. No model or
dataset computation was performed.

This was the final local preflight before runtime milestones; further work must
be driven by completed training evidence rather than additional speculative
contract changes.

## Decision rule

The experiment is useful only if its fixed epoch-15 three-seed median improves
the final deploy-aligned triage comparison, especially Macro-F1 and Urgent-F1,
while satisfying every preregistered non-inferiority gate. Fold 0 is an early
rejection screen only and cannot authorize a release or submission ZIP.

## Seed-42 terminal materialization and transfer

The first A1 replicate completed all 23 locked epochs at
`2026-09-03T01:27:49Z` with launcher exit code 0. The canonical history has
exactly epochs 1 through 23, and the server was GPU-idle after completion. The
fixed epoch-15 checkpoint remained byte-identical to the preregistered value:

- run: `mls-vast-da-a1-ordinal-fold0-seed42`;
- checkpoint bytes: `124,934,257`;
- checkpoint SHA-256:
  `7f4b4daee6935618ed2740464036971ae665b7acf798bf1220e34e066892aab8`;
- MLflow run: `21735244d12c42c8b7594d1bda1e627f`, status `FINISHED`;
- MLflow artifact metadata included five model snapshots plus the report,
  complete epoch history, run summary, resolved configuration, runtime
  metadata, and source snapshot; no private medical predictions were uploaded.

The strict transfer manifest has SHA-256
`e41846f4f7409da439d1d50afe5ddafc879180216544c650ff1bdec13c851242`.
All six declared artifacts verified locally, including the fixed checkpoint,
manifest, launcher status, report, 23-row history, and run log. The local
verification record is under
`checkpoint/mls/mls-vast-da-a1-ordinal-fold0-seed42/transfer_verification.json`.

No checkpoint was deleted during this materialization. Server free space after
transfer was `3,015,540,736` bytes, below the immutable 4-GiB launcher gate.
The eight non-fixed seed-42 files occupy `1,229,532,610` bytes in aggregate
(seven inference snapshots plus a `354,966,975`-byte resume file); their
removal requires explicit approval even though they are ineligible for the
fixed audit. Removing them alone would leave `4,245,073,346` bytes, still
`49,893,950` bytes below 4 GiB. A further small reproducible or ineligible
artifact must therefore be reclaimed before seed 2026 can launch. Until that
approval and gate are satisfied, no second GPU job may start.

The user subsequently authorized removal of files that were genuinely
ineligible for the fixed audit, with recoverability retained. Before deletion,
all nine files were bound to exact SHA-256 values and verified MLflow artifact
paths. Three previously unlogged seed-42 artifacts (`best_study_boundary`,
`final`, and `resume_latest`) were archived under MLflow run
`21735244d12c42c8b7594d1bda1e627f`; the baseline-seed2026 `final` artifact was
archived under run `560b53753e304e218036bc8cdcc8061a`. The other five
seed-42 snapshots already existed in MLflow. The complete recovery map is
stored locally beside the fixed checkpoint in
`nonfixed_cleanup_recovery_manifest.json`.

Exactly eight non-fixed seed-42 files and one non-fixed baseline-seed2026 final
checkpoint were then removed. The fixed seed-42 epoch-15 file remained the
only checkpoint in its server run directory and retained SHA-256
`7f4b4daee6935618ed2740464036971ae665b7acf798bf1220e34e066892aab8`.
Server free space rose to `4,369,993,728` bytes, above the immutable 4-GiB
launcher gate. A1 seed2026 subsequently passed preflight (`4,267,572` KiB
available versus `4,194,304` KiB required) and launched at
`2026-09-03T02:02:26Z` from manifest SHA-256
`802d8b7e1893a2a29b9a52254967490e5626ea8cbbeeeab23813816e0d4d6a25`.
Its warm-up showed 95% GPU utilization and 5,339 MiB allocated VRAM, with the
CUDA-only/no-CPU-fallback launcher contract active.

## Seed-2026 terminal materialization, recovery archive, and seed-3407 launch

The second A1 replicate completed all 23 locked epochs at
`2026-09-03T03:29:13Z` with launcher exit code 0. MLflow run
`4d701a93db3b4feba7bdab9411ecc4f8` is `FINISHED`, the six-item local transfer
verified against manifest SHA-256
`3821f1a0e616a92b2cdfd2da3a1166f15907325e81bacade7e59ac6563908cd9`, and the
fixed epoch-15 checkpoint SHA-256 is
`b9f93c91aaf7cb7bd57c88a236b0c3b8e4a9417939a372705b831d62a4337ecb`.

Its fixed-epoch diagnostics were MAE `1.497669 mm`, F1@3 `0.785714`, F1@5
`0.722222`, boundary-F1 `0.753968`, selection objective `2.038606`, and selector
AUC `0.902254`. This replicate is diagnostically weak and is not promoted, but
the preregistered three-seed median still requires seed 3407 before the A1
candidate can be accepted or rejected.

With explicit user authorization, the three seed-2026 artifacts that were not
already logged (`best_study_boundary`, `final`, and `resume_latest`) were
uploaded and size-verified under MLflow run
`4d701a93db3b4feba7bdab9411ecc4f8` at `models/nonfixed_archive/`. The baseline
seed-3407 `final` checkpoint was likewise archived under run
`e51ba117f0c84724bbe5215062de27cb`. Exact hashes, byte sizes, original runs,
and recovery paths are recorded in
`checkpoint/mls/mls-vast-da-a1-ordinal-fold0-seed2026/nonfixed_cleanup_recovery_manifest.json`.
The other five seed-2026 non-fixed snapshots already existed in the standard
`models/` artifact namespace.

Exactly eight non-fixed seed-2026 checkpoints and the one archived baseline
seed-3407 final checkpoint were then removed from the shared server model
store. The fixed seed-2026 epoch-15 checkpoint remained byte-identical and was
the only file left in that run directory. Free space increased to
`4,367,593,472` bytes, above the immutable 4-GiB launcher gate.

The third and final preregistered A1 replicate passed recipe parity, idle-GPU,
global-lock, destination, and disk preflight checks and launched at
`2026-09-03T07:20:45Z` in tmux session `mls_da_a1_f0_s3407`. It uses manifest
SHA-256 `f1c58b921e13ab42e72b36cc1e20bca62c434ef48a966442af40c47d31ea8485`
from server commit `b321de5d11f4fd2b00b80eeef25eb56e068a015c`. The one-time post-launch check
confirmed an active CUDA process using 5,296 MiB VRAM. Monitoring now returns
to the 60--90 minute milestone cadence; no fourth experiment is authorized.
