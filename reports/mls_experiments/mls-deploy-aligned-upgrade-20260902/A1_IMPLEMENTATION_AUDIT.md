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

Conclusion: no implementation defect or inference-contract drift was found in
the reviewed A1 gradient path. The scientific hypothesis remains unproven until
fixed-epoch, three-seed, deploy-aligned evaluation is complete.

## Runtime milestone snapshot

- Snapshot time: `2026-09-03T00:15:16Z`
- Vast instance: `49527185`; GPU: NVIDIA GeForce RTX 3060 12 GiB
- Session: `mls_da_a1_f0_s42` present
- Active job wrapper: `mls-vast-da-a1-ordinal-fold0-seed42`
- GPU state: 100% utilization, 5,339 MiB used, 57 C
- Completed epoch rows: 0 (the first epoch had not yet reached its atomic metric
  write at snapshot time)
- Free workspace bytes: 3,267,231,744

No transfer, second heavy job, checkpoint selection, or adaptive threshold
selection was performed during this snapshot.

## Decision rule

The experiment is useful only if its fixed epoch-15 three-seed median improves
the final deploy-aligned triage comparison, especially Macro-F1 and Urgent-F1,
while satisfying every preregistered non-inferiority gate. Fold 0 is an early
rejection screen only and cannot authorize a release or submission ZIP.
