# A5: selector-only same-study ranking screen

- Status: preregistered before any A5 CUDA preflight, training, or metric is read.
- Scope: one RTX 3090 run, fold 0 / seed 42, fixed epoch 15 only.
- Compute: CUDA-only; one GPU workload at a time; no CPU model fallback.

## Motivation

A4's shared-backbone RankNet update was technically valid but failed every
fixed resource gate.  The A4 result is not evidence of faulty transfer or
evaluation: its epoch-15 CUDA audit completed zero-failure inference on 70
held-out studies.  The rejected model also showed a large difference between
the fixed p90 decision profile and an explicitly non-decision diagnostic
profile, so post-hoc pooling changes are prohibited.

The new mechanism is deliberately narrower than a new architecture: preserve
all ordinary slice supervision exactly, while preventing the sparse pair-only
rank update from changing the geometry representation or its BatchNorm
statistics.  The pair loss still teaches the existing selector head on frozen
backbone features.  This directly tests whether rank information can improve
slice ordering without the negative transfer suggested by A4.  It is not an
attempt to rescue A4 with a new threshold, loss weight, or checkpoint.

The premise is supported only as a hypothesis, not a promised gain.  Work on
multi-task optimisation documents that auxiliary gradients can interfere with
a primary task, motivating explicitly controlled auxiliary updates:
[PCGrad](https://arxiv.org/abs/2001.06782) and
[Auxiliary Task Update Decomposition](https://doi.org/10.48550/arXiv.2108.11346).
The MLS-specific reason to retain the rank signal is unchanged: deployment
uses selector ordering before fixed local aggregation.

## Fixed intervention

At each scheduled pair update, `forward_selector_only_detached_backbone`:

1. places only the backbone in evaluation mode and records every module's
   original train/eval state;
2. calculates its features under `torch.no_grad()`;
3. restores those states, then runs the existing selector head in its ordinary
   training mode on the detached features.

Consequently, the A5 pair loss can create gradients for selector-head weights,
but cannot create gradients for the backbone or heatmap head and cannot update
backbone BatchNorm buffers.  The ordinary slice batch remains exactly the
full-gradient A4/baseline path.  A CUDA regression test proves all of those
properties; a CUDA preflight uses the exact W32, 512-pixel, batch-10 recipe.

All remaining training values are frozen to the A4 template: rank weight
`0.10`, minimum gap `1 mm`, temperature `1`, cadence every four ordinary
batches, HRNet-W32, strict determinism, 23 epochs, fold 0, seed 42, epoch 15,
selector and geometry losses, sampling, and deployment aggregation.  A3 bag
loss remains zero.  Batch size stays 10 despite unused 3090 memory so that
batch/optimiser dynamics are not a second experimental factor.

## Resource gate and stop rule

The unmodified 70-study fold-0 resource contract decides the A5 epoch-15
checkpoint.  All five limits must pass:

| Metric | Required |
|---|---:|
| study MAE | <= 1.470959 mm |
| F1 at 3 mm | >= 0.819672 |
| F1 at 5 mm | >= 0.736842 |
| boundary F1 | >= 0.778257 |
| selection objective | <= 1.904444 |

Any failure writes `rejected_stop_a5_expansion`; it authorizes no A5 seed
replication, cross-fold run, pooling/threshold change, ensemble, promotion, or
submission.  A pass is only a resource-screen success.  It does not create a
release or a leaderboard submission and requires a separately frozen
replication and triage plan before further GPU work.

## Privacy and provenance

MLflow receives training aggregates and fixed-audit aggregate metrics only.
Raw per-study predictions remain on the server.  Before any training, source,
manifest, tests, and CUDA preflight must be SHA-256 matched between the clean
local worktree and clean server worktree; replaced remote files are backed up.
