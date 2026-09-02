# Exp88 — Any-invariant temporal subtype residual calibration screen

## Rationale fixed before execution

Exp53 proved that a small bidirectional temporal head over frozen 2.5D encoder
features contains transferable subtype signal. On its one-shot outer2 evaluation,
macro subtype AUC improved by `+0.02908` and the selection proxy by `+0.00342`.
Expansion was rejected only because Any-ICH AUC fell `-0.00314`, just beyond its
locked `-0.002` safety margin. The six-output residual and joint loss allowed the
shared temporal trunk to alter Any ranking.

Exp88 removes that failure mode by architecture: the temporal head emits only
five subtype residuals, concatenated with the bit-exact frozen incumbent Any
logit. Any is also absent from both slice and study loss. This is a direct causal
successor to Exp53, not a post-hoc channel blend.

## Locked execution

- Base model: exp22 Unet++/EfficientNet-B2 checkpoint for outer2/calibration1.
- Schema-v4 manifest and the exact batch-16 frozen feature caches previously
  audited for Exp53 are reused only after hash/provenance validation.
- Temporal architecture remains LayerNorm → Linear `352→64` → GELU → BiGRU
  hidden `32` → dropout `0.2`; only the residual output changes from six to five
  subtype channels.
- Same Exp53 recipe: focal gamma `1`, study-loss weight `0.5`, AdamW `5e-4`,
  weight decay `1e-3`, batch eight studies, maximum 20 epochs, patience four,
  positive-weight cap `20`, seed `42`.
- Epoch selection uses the original score weights. Since Any delta is exactly
  zero, only `0.15 × macro-subtype-AUC delta` can change the proxy.
- Calibration1 is used for checkpoint selection. Outer2 is not inferred again.
  No threshold search, row-level prediction persistence, external MLflow,
  Telegram, GitHub push or accepted-checkpoint promotion occurs.

## Locked promotion gate

Every condition is conjunctive:

1. Any-ICH AUC delta is exactly zero within `1e-12`;
2. macro subtype AUC improves by at least `+0.01`;
3. selection proxy improves by at least `+0.0015`;
4. no subtype AUC falls by more than `0.01`;
5. at least three of five subtype AUCs improve.

Passing authorizes a patient-disjoint five-fold *development OOF* implementation
with fold-specific base checkpoints. It does not authorize reuse of outer2 as a
fresh confirmation, accepted-checkpoint promotion or leaderboard claims.
