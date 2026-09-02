# Exp19 preregistered plan: independent fold0 dual-selector replication

## Why this run exists

Exp18 was not a valid standalone replacement for Exp09. Its completed 804-case
CUDA audit failed all frozen gates. A later preregistered component screen did,
however, find a narrow and interpretable complementarity result on fold1:

- baseline: Exp09/epoch15;
- challenger: Exp18/epoch21;
- only `mls_mm` is blended;
- baseline weight `0.90`, challenger weight `0.10`;
- MAE `1.248084723 mm`;
- Boundary-F1 `0.831034483`;
- objective `1.586015757`.

All four component modes were identical at this alpha, proving that selector,
peak and heatmap changes were unnecessary. Seven grid candidates passed, but
all used alpha 0.10. Because this was selected on fold1, it is not yet an
unbiased release result.

## Single training factor

Exp19 copies the Exp18 training recipe exactly and changes only the held-out
competition fold from 1 to 0. In particular, it retains:

- HRNet-W32, 512px, three input channels;
- dual presence/peak selector and normalized selector loss;
- official 3484-row multitask-v2 contract;
- slice-class-balanced sampler;
- seed 42 and strict CUDA determinism;
- optimizer, LR, weight decay, all losses and augmentations;
- 23 epochs and snapshots 13/15/17/19/21/23;
- no warm start and no CPU model fallback.

## Frozen primary transfer test

The primary fold0 test is fixed before training and cannot be changed after
seeing Exp19 metrics:

1. Exp16 `best_selector_auc`/epoch16 is the fold0 baseline.
2. Exp19 `epoch21` is the only primary challenger checkpoint.
3. Exp16 supplies selector probability, peak/ranking probability and heatmap.
4. Only `mls_mm` is blended as `0.90 * Exp16 + 0.10 * Exp19`.
5. The production profile remains `severity_window`, radius 3, selector gate
   0.5, minimum 3 active slices, weighted q0.75, no heatmap guard.
6. No alpha, checkpoint, offset, threshold or pooling retune is allowed.

Exp16's authoritative same-runtime reference is:

- MAE `1.6044777010 mm`;
- Boundary-F1 `0.8273325590`;
- objective `1.9498125829`.

Primary replication passes only if all are true:

- hybrid MAE is no worse than `1.6044777010 mm`;
- hybrid Boundary-F1 is no worse than `0.8273325590`;
- hybrid objective is at most `1.9398125829`, an improvement of at least 0.01;
- all 70 fold0 studies complete CUDA inference with zero failures.

The historical release floors (MAE 1.664553, Boundary-F1 0.82, objective
2.020027) are secondary; merely passing those weaker gates does not prove the
Exp18 component recipe transferred.

## Resource gate and stopping rule

Training is justified because a complete saved-prediction screen produced a
positive, mechanistically simple hypothesis. It is not a seed sweep. Peak VRAM
for Exp18 was 4.648GB on the same RTX3060, so the run fits the 12GB device.

After training, only epoch21 needs primary full-study CUDA inference. If the
fixed fold0 transfer gate fails, no claim of cross-fold complementarity is
allowed. Other checkpoints may be audited for failure analysis, but cannot
replace epoch21 in the primary result. If it passes, repeat the same fixed test
on a third fold before any leaderboard-facing two-model deployment.

All run metrics and aggregate reports go to MLflow. Raw medical data and
per-study prediction CSVs do not. The Vast instance remains active and is never
stopped or destroyed without user coordination.
