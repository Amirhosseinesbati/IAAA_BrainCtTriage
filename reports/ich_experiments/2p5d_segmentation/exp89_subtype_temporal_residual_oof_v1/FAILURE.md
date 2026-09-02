# Exp89 execution failure — evaluator edge case

The first execution stopped after fold0 checkpoint selection and outer inference.
The spatial/temporal models and GPU remained healthy. Fold0 had only one class for
at least one rare subtype, so `auc_summary` correctly returned `null` for that
fold-specific AUC. The generic strict delta helper then attempted `float(null)` and
raised `TypeError` before any fold summary or pooled result was written.

No hyperparameter, checkpoint mapping, promotion gate or observed model metric is
being changed. Exp89r1 is a code-only retry: fold-level undefined subtype deltas are
preserved as `null`; pooled OOF still requires all subtype AUCs and uses the original
strict delta/gates. The failed directory is retained for provenance and is not
overwritten.
