# Exp89r2 — fresh-cache retry of the locked Exp89 OOF suite

Exp89r1 completed outer folds 0 and 1 and selected the fold2 temporal checkpoint,
then stopped because the shared Exp53 cache namespace contained an older fold2
outer cache whose metadata did not match the current strict `split=outer` contract.
The validator correctly refused silent reuse.

Exp89r2 changes no scientific choice: checkpoint mapping and hashes, patient
splits, Exp88 architecture, seed, optimizer, epoch selection, outer timing,
bootstrap, primary gate and strong gate remain exactly those preregistered in
`exp89_subtype_temporal_residual_oof_v1/PREREGISTERED_PLAN.md`. The nullable
single-class fold evaluator fix from Exp89r1 is retained.

The sole operational change is a fresh cache root
`/workspace/cache/ich_temporal_exp89r2_fresh`. No old feature cache will be copied,
deleted or reused. All train/calibration/outer frozen features will be regenerated
from the five locked accepted spatial checkpoints, and outer extraction remains
after checkpoint selection. Exp89 and Exp89r1 directories are preserved.

This is still development OOF, not a final unseen test. No gate may be adjusted
after observing Exp89r1's partial fold results.
