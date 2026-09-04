# Confirmed defect: resource-gate comparator pooling mismatch

Discovered during the A6 cache-alignment investigation, before reading any
corrected candidate result. This changes interpretation of the earlier resource
decisions; it does not claim that any candidate is good or authorize training.

The A2 intended protocol explicitly references the frozen baseline seed42 from
the three-seed audit and says inference settings are unchanged. Its thresholds
were derived from that baseline's checkpoint-configured aggregation. However,
`evaluate_mls_a2_fold0_resource_screen.py` enforces candidate0.5/top3/p90, and
A3-A6 reuse it. Thus the comparison mixed two distinct inference policies.

Confirmed source: `evaluate_mls_three_seed_fold_cuda._aggregate` uses all frozen
checkpoint pooling fields plus clipping to[0,30]mm. Baseline checkpoint metadata
was read on the server (no model forward): threshold0.6, top5, relative_component,
relative_ratio0.3, quantile0.75, probability_weighted=true, min_active3,
anchor_radius3, heatmap_guard0, negative0.1. The config template agrees.

The same baseline epoch15/SHA256 c2427320... has:

- Canonical pooling from the three-seed audit: MAE1.4709586392, F1@3 .8196721311,
  F1@5 .7368421053, objective1.9144444028.
- Old audit cache with0.5/top3/p90: MAE2.4704536704, F1@3 .7142857143,
  F1@5 .7142857143.

The numerical reference itself was not fictitious, but it was attached to the
wrong candidate pooling. A gate can be numerically fail-closed yet scientifically
invalid. The previous all-fail results remain valid descriptions of the old
profile; they do not prove matched-deployment inferiority against baseline.

## Corrective procedure frozen before corrected outcomes

Use `POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json` and
`reconstruct_mls_aligned_cached_screen.py`. Include baseline and every completed
A2-A6 epoch15 candidate, not just the most promising one. No retraining, no new
inference, checkpoint search or threshold sweep. Keep all original gate bounds
(objective still requires0.01 improvement); use1e-8 numerical tolerance solely
for stored decimal rounding.

Verify source/checkpoint/cache hashes, frozen fold0 identities, unique keys,
truth and slice alignment; reproduce every native0.5/top3/p90 result and baseline
canonical values for each study against its immutable three-seed private CSV.
Only then compute candidates with exactly the baseline pooling. Missing or
inconsistent coverage invalidates reconstruction; do not fill or drop cases.
Only aggregate outputs may leave the server.

Historical JSON decisions are never overwritten. Label the old comparator as
defective and publish separate corrected results. Previously claimed conclusions
such as "A6 rejected therefore worse than baseline" require this correction.
No candidate is promoted, automatically replicated, or packaged from this repair.
The original multi-seed, cross-fold, frozen-context triage hard gates remain.
The proposed geometry/selector2x2 diagnostic is deferred until the comparator
repair is resolved. Automatic monitoring remains paused by user request.
