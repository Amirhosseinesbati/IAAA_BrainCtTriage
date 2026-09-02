# Exp81 — factorized loss-causality train-only probe

## Question fixed before execution

Exp80 proved that the BF16-exact factorized architecture preserves Exp61 at
initialization, but its first full objective rapidly favored IVH/IPH while
suppressing diffuse SDH/SAH. This diagnostic asks whether the causal pressure is
primarily conditional class-weighted focal CE, conditional subtype Dice, their
interaction, or the shared foreground/decoder update.

## Locked scope and provenance

- Exp61 checkpoint SHA-256
  `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- Schema4 manifest SHA-256
  `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.
- Outer fold `2`, calibration fold `1`, seed `42`; train split only.
- No calibration/outer/OOF inference, no model promotion and no row-level
  prediction artifacts.
- Same `unetplusplus/tu-efficientnetv2_rw_s` BF16 factorized warm start as
  Exp80, with encoder/classifier frozen and decoder/spatial heads trainable.

## Locked diagnostic

1. On 24 foreground-containing train batches, decompose the exact Exp80 loss
   into foreground support (`0.325 Dice + 0.175 focal + 0.05 hard-empty`),
   conditional class-weighted focal CE (`0.175`, gamma `2`) and conditional
   subtype Dice (`0.325`). Record per-branch parameter-gradient norms,
   pairwise gradient cosines and per-class subtype-margin attraction.
2. From the identical zero-residual initialization and identical ordered train
   batches, run eight AdamW updates at `5e-5` for four variants:
   `full_exp80`, `without_conditional_dice`, `without_conditional_focal`, and
   `foreground_only`.
3. Measure pre/post drift on four fixed, non-updated train-distribution probe
   batches: hard/soft Dice, true-pixel target probability, conditional target
   probability, subtype margin, predicted soft volume and background
   foreground probability. These are mechanistic train diagnostics, not
   estimates of generalization.

## Locked interpretation rule

- If full-objective mean diffuse (SAH/SDH) soft-Dice drift is not worse than
  `-0.001`, return `short_horizon_drift_not_reproduced`.
- Removing one conditional component identifies it as the primary suspect only
  if it rescues diffuse mean soft Dice by at least `0.005` versus the full
  objective and exceeds the other removal's rescue by at least `0.002`.
- If both removals rescue by at least `0.005`, classify the failure as a
  conditional-loss interaction.
- Otherwise classify it as shared-foreground/decoder pressure or inconclusive.

The result can authorize design of a new preregistered calibration recipe, but
cannot authorize outer evaluation or checkpoint promotion.
