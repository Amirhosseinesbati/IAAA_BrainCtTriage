# Exp88 result — Any-invariant temporal subtype residual

Decision: `advance_to_patient_disjoint_five_fold_development_oof`.

Exp88 kept the frozen spatial model's Any-ICH logit bit-exact and trained only a
42,437-parameter temporal residual for the five subtype logits over ordered slice
features. The recipe and promotion gate were fixed before execution. On
calibration fold 1, best epoch 13 improved macro subtype AUC from `0.897887` to
`0.912073` (`+0.014186`) while Any-ICH AUC remained exactly `0.920251` (delta
`0.0`). The corresponding selection proxy improved `+0.002128`.

Subtype AUC deltas were EDH `+0.034483`, SDH `+0.021073`, IVH `+0.012903`, SAH
`+0.011905`, and IPH `-0.009434`. Thus all five preregistered checks passed:
exact Any invariance, macro gain at least `0.01`, proxy gain at least `0.0015`, no
subtype loss below `-0.01`, and at least three improved subtypes.

This is a calibration-screen success, not final evidence. Outer2 was deliberately
not inferred and spatial Dice/volume metrics are unchanged by design. The next
authorized step is a patient-disjoint five-fold development OOF implementation
with fold-specific spatial checkpoints. Runtime was `10.61s`, peak VRAM
`0.039 GiB`. Diagnostic checkpoint SHA256:
`c7a80e9eb44f76e9a5c7ba4eb89c121bf55fcd1adc2ed93d367090f86f47600d`.
