# A3 — selection-aligned positive-study auxiliary loss

## Decision context

The A2 signed-offset intervention is rejected on its fixed fold-0 resource
screen. Its 70-study, fixed `gate=0.5 / top3 / p90` audit yielded MAE
`2.016416 mm`, F1@3 `0.787879`, F1@5 `0.761905`, and boundary F1 `0.774892`.
It missed four of five expansion gates, so no A2 seed, fold, pooling search,
promotion, or submission is permitted.

The retrospective aggregate-only diagnostic locates a bidirectional failure:
the candidate overestimates low truth strata (especially 1--<3 mm) and
underestimates the 5+ mm strata. This produces 9 false positives and 5 false
negatives at 3 mm, and 6 false positives and 4 false negatives at 5 mm. It is
not evidence for changing the locked A2 pooling profile.

## Hypothesis

Current training optimizes independently sampled slices. Although local MLS and
relative peak labels are supervised, no loss connects the peak-selector ranking
and predicted local MLS to the official *study maximum* that deployment must
recover. A small positive-study auxiliary loss should suppress isolated high
local estimates in low-MLS studies and direct peak mass towards truly severe
slices, while retaining the successful local heatmap objective.

This is not the already rejected study-balanced sampler. The primary
`slice_class_balanced` loader remains unchanged. A3 additionally consumes one
positive target-slice bag every four ordinary batches; its 0.20 weight is the
only new optimization signal. Historical full study-balanced and hybrid
samplers are explicitly not re-run.

## Frozen implementation and runtime contract

- Manifest: `config/experiments/mls-vast-deploy-aligned-a3-study-bag-template.yaml`.
- Fold/seed/schedule: fold 0, seed 42, 23 epochs; audit exactly epoch 15.
- Base model and local objectives: HRNet-W32, 512 px, original slice-balanced
  sampling, selector, local geometry, 1/3/5-mm threshold and augmentation
  configuration are unchanged from the A2 template except that rejected signed
  loss is reset to zero.
- Auxiliary target: all annotated target slices from one *training* study;
  peak logits form a softmax-weighted local MLS value and it is supervised
  against that study's official maximum plus a 1/3/5-mm boundary BCE.
- Deployment remains the existing absolute keypoint geometry with the frozen
  `selector_threshold=0.5`, `top_k=3`, `p90`, and `min_active=1` audit path.
  A3 does not tune or replace it.
- The terminal audit launcher is
  `scripts/run_vast_mls_a3_fold0_seed42_resource_screen.sh`; it refuses any
  checkpoint other than epoch 15, any incomplete training state, GPU overlap,
  unsafe secret permissions, failed inference, or a changed audit contract.
- Before launch, `smoke_mls_study_bag_cuda.py` must complete a forward/backward
  pass on the largest training positive bag using CUDA. No CPU fallback is
  accepted. Batch size remains 5 rather than changing a second optimization
  variable; the 3090's additional VRAM is reserved for the whole study bag.

## Preflight outcome (recorded before launch)

- Source implementation commit: `be3056e`; it was copied directly to the
  server because noninteractive GitHub credentials are unavailable. The server
  worktree base remains `b321de5`; the manifest and MLflow tags carry the local
  source commit to prevent this distinction being lost.
- Every transferred A3 file matched its local SHA-256 before preflight.
- Largest positive fold-0 training bag: 23 annotated target slices across 138
  positive training-study bags.
- CUDA-only forward/backward: `ok`, finite loss; peak allocated VRAM
  `19.080746 GiB` on the RTX 3090. The preflight result is stored only on the
  server at `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a3_fold0_seed42/preflight_largest_positive_bag.json`, SHA-256
  `9623f00306fc245f352eb9d5bcebd7cf0511a23473d7ab360a4b46ea9be181d9`.
- Therefore the controlled primary slice batch remains 5: raising it would
  consume the headroom required by the worst-case study-bag update and would
  introduce a second optimization change.

## Fixed decision rule

After completion, run exactly one full CUDA audit on the 70 held-out fold-0
studies and exactly checkpoint epoch 15. The A3 candidate can unlock only two
predeclared new-seed replications if all of these are met relative to the frozen
resource baseline:

| Metric | Required |
|---|---:|
| MAE | <= 1.470959 mm |
| F1 @ 3 mm | >= 0.819672 |
| F1 @ 5 mm | >= 0.736842 |
| Boundary F1 | >= 0.778257 |
| Objective | <= 1.904444 |

Failing any gate stops A3 expansion. Validation proxy values, in-fold pooling
grids, alternate checkpoints, and ensemble experiments cannot rescue it.
Passing this screen is not a promotion or submission authorization: it only
permits the predeclared seed replications, followed by cross-fold and complete
triage validation.

## Locked evidence chain from MLS screen to triage claim

The 70-study resource screen is deliberately an inexpensive MLS gate; it is
not evidence that final triage Macro-F1 or Urgent F1 improved.  Before its
outcome is known, the following non-adaptive chain is fixed:

1. Only a pass unlocks the two already named fold-0 seed replications (`2026`
   and `3407`).  They retain the A3 manifest in every model-affecting field,
   fixed epoch 15, and frozen deploy pooling; seed is the sole allowed training
   difference.
2. The three fixed fold-0 epoch-15 checkpoints (42/2026/3407) are audited by
   `evaluate_mls_three_seed_fold_cuda.py`, which refuses CPU fallback,
   non-distinct seeds, model-affecting configuration differences, incomplete
   studies, or non-fixed epochs.  Its per-study member predictions remain
   private on the server; the aggregate audit binds their SHA-256.
3. The existing `evaluate_mls_deploy_aligned_seed_medians.py` then compares the
   candidate and immutable baseline fold-0 three-seed medians in both frozen
   Champion-branch and oracle contexts.  No threshold, pooling, checkpoint, or
   triage decision rule may be searched at this point.  A positive MLS metric
   alone never authorizes wider training.
4. Cross-fold expansion is allowed only if the frozen-context Macro-F1 and
   Urgent F1 directions are positive while accuracy, Normal/Critical F1,
   3/5-mm F1, catastrophic-error, bootstrap, and oracle-direction safeguards
   satisfy the canonical aggregate gate.  Otherwise A3 stops with no rescue
   change.
5. A submission claim requires the complete five-fold, three-seed OOF protocol
   over all 338 studies and a pass from
   `assert_mls_final_promotion_gate.py`.  That final gate requires checksum-bound
   sources for all five folds and explicitly forbids clean ZIP creation before
   both Macro-F1 and Urgent F1 improve against the frozen Champion context.

## Privacy and tracking

The per-study/per-slice audit CSV remains private to the server. MLflow receives
only normal aggregate training metrics and aggregate gate evidence. Raw study
predictions are never uploaded, copied locally, or used for pooling searches.
