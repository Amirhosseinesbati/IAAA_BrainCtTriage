# Exp89 — patient-disjoint five-fold Any-invariant temporal subtype OOF

## Rationale fixed before execution

Exp88 improved calibration1 macro subtype AUC by `+0.014186` and its classification
selection proxy by `+0.002128`, while preserving the spatial model's Any-ICH logit
exactly. Four of five subtypes improved; IPH declined `-0.009434`, close to the
locked safety boundary. A five-fold outer evaluation is therefore necessary before
any deployment integration or checkpoint promotion.

This suite uses the exact five spatial checkpoints already packaged as the accepted
hard-pixel OOF candidate. It does not use the later IVH-center checkpoints, because
mixing a different spatial model with the temporal intervention would confound the
attribution.

| outer | calibration | spatial checkpoint | locked SHA256 prefix |
|---:|---:|---|---|
| 0 | 1 | exp20 | `62d696d4d5f4` |
| 1 | 2 | exp21b | `b3125e8c8a09` |
| 2 | 1 | exp22 | `c63e609c652c` |
| 3 | 1 | exp18 | `cef60d76040c` |
| 4 | 1 | exp19 | `a5c968856345` |

The pooled frozen baseline must reproduce exactly 338 studies, 320 patients,
Any-ICH AUC `0.9345288326300984`, and macro subtype AUC
`0.8291593268101609`. A mismatch aborts the suite instead of producing a result
against a different model.

## Locked protocol

- Every fold uses its checkpoint's original patient-disjoint training, calibration
  and outer partitions. Pairwise patient overlap must be zero.
- The Exp88 recipe is unchanged: frozen encoder features, LayerNorm → Linear 64 →
  GELU → bidirectional GRU hidden 32 → dropout 0.2 → five subtype residuals;
  AdamW `5e-4`, weight decay `1e-3`, focal gamma 1, study loss 0.5, batch eight,
  maximum 20 epochs, patience four, positive-weight cap 20 and seed 42.
- The Any-ICH logit bypasses the temporal residual and must remain bit-exact.
- Checkpoint selection sees only that fold's calibration partition. Its outer cache
  is opened or created only after the temporal checkpoint is fixed.
- Outer predictions are held only in memory for paired aggregation. No row-level
  medical predictions, external MLflow/DagsHub artifacts, Telegram payloads or
  GitHub push are produced by this development run.
- Paired uncertainty uses 2,000 patient-cluster bootstrap samples.
- This is development OOF. Hyperparameters were motivated by Exp88, so this is not
  a final unseen test and cannot itself justify a leaderboard claim.

## Primary advance gate

All conditions are conjunctive:

1. exact study/patient coverage, locked-baseline reproduction and zero leakage;
2. Any logits/AUC unchanged within `1e-12`;
3. pooled macro subtype AUC delta at least `+0.005`;
4. classification selection proxy delta at least `+0.00075`;
5. no pooled subtype AUC delta below `-0.01`;
6. at least three subtypes improve;
7. at least three of five fold macro deltas are nonnegative and the worst is at
   least `-0.025`;
8. patient-bootstrap probability of macro improvement at least `0.90`, with at
   least `95%` valid bootstrap samples.

Passing authorizes replication/deployment-prototype work, not accepted-checkpoint
promotion. Strong support additionally requires at least four nonnegative folds,
worst fold at least `-0.01`, bootstrap probability at least `0.95`, and a strictly
positive 95% bootstrap lower bound. Gates will not be lowered after observation.
