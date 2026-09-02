# Exp89r1 — code-only retry of Exp89

Exp89 stopped on an evaluator edge case after fold0 outer inference: a rare subtype
had a single-class outer fold, for which AUC is correctly undefined. Conversion of
that `null` fold metric to `float` crashed before a fold summary or pooled result was
written.

Exp89r1 keeps the exact checkpoint mapping, hashes, patient splits, temporal
architecture, optimizer, seed, epoch selection, bootstrap and every primary/strong
gate from `exp89_subtype_temporal_residual_oof_v1/PREREGISTERED_PLAN.md`. The only
change is that undefined *fold-level* subtype AUC deltas remain JSON `null`. Fold
macro AUC continues to average the available subtype AUCs, as before. Pooled OOF
contains both classes for every subtype, must reproduce the locked baseline, and
still uses the original strict delta implementation.

Existing deterministic feature caches may be reused only after their checkpoint,
manifest, split and file hashes pass the existing cache validator. The failed Exp89
directory is preserved and Exp89r1 writes to a new directory. This remains
development OOF, not a final unseen test or leaderboard result.
