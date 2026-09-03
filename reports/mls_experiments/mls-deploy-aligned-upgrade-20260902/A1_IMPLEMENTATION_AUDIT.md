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
