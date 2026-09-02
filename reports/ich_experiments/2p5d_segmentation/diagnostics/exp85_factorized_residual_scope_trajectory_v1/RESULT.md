# Exp85 result — confirmatory gate failed; exploratory scope attribution completed

## Formal decision

`reproduction_failed_no_scientific_interpretation`

No candidate passed the locked Exp84 gate and no checkpoint was written. Outer
fold 2 was not inferred. The current joint foreground/subtype residual recipe is
closed; it must not be continued by sweeping epochs or learning rates.

## Reproduction audit

Recorded batch identities matched exactly across all three parameter scopes.
However, the joint scope did not reproduce Exp83 step 4 or Exp84 epoch 1 within
the preregistered maximum aggregate tolerance `1e-6`:

- versus Exp83 step 4, maximum difference was `0.03195 mL` in total-volume bias;
- versus Exp84 epoch 1, maximum difference was `0.00640 mL` in volume MAE.

Dice and checkpoint-score differences were much smaller (`1e-4` order), which
is consistent with small GPU/data-order numerical variation, but the locked
tolerance was conjunctive. Therefore the trajectory cannot be labelled a
confirmatory replication. The within-run randomized-control-style scope
comparison remains useful as exploratory causal evidence because sampler batch
identities were identical.

## Exploratory attribution

All variants used the same one-epoch sequence and were evaluated at steps 4,
16, 32, 64, 128, 192 and 303.

- **Both heads (870 parameters):** after step 4 the score was only `+0.000033`
  above baseline. It then declined monotonically; by step 303, SAH Dice was
  `+0.00858` but SDH Dice was `-0.03256`, volume MAE `+0.3449 mL`, and score
  `-0.00558`.
- **Foreground only (145 parameters):** step 4 was effectively identity. By
  step 303 it reduced SAH by `-0.00323`, SDH by `-0.02242`, mean Dice by
  `-0.00532`, worsened MAE by `+0.3547 mL`, and reduced score by `-0.00595`.
  This head is the dominant source of support/volume degradation and provides
  no SAH benefit.
- **Subtype only (725 parameters):** this was consistently the least harmful
  and temporarily beneficial scope. Its best checkpoint score occurred at step
  128: score `+0.000660`, selection `+0.000579`, mean Dice `+0.001053`, SAH
  `+0.009794`, IVH `+0.001014`, IPH `+0.000359`, EDH `+0.000365`, and volume
  MAE `-0.01427 mL`; SDH nevertheless fell `-0.006266`. At step 303 SAH reached
  `+0.01279`, but SDH loss grew to `-0.01062` and score gain fell to
  `+0.000518`.

No milestone met the material-gain gate. The best subtype-only point improved
every monitored direction except SDH, but the magnitude was too small and SDH
crossed its safety limit.

## Consequence for the next architecture

The foreground residual branch should be removed from further work on this
factorization. A generic five-way subtype residual is also insufficient because
its SAH gain is purchased by reallocating SDH pixels. Any justified successor
must enforce class-selective invariants—e.g. prevent correct incumbent SDH/EDH
decisions from being relabelled—rather than merely retune the same objective.

The historical class-safe SAH adapters must not be repeated blindly: Exp65 made
no hard change; Exp66 achieved only score `+0.00054` and SAH Dice `+0.00444`;
Exp67 achieved score `+0.00118` and SAH Dice `+0.01014` while preserving SDH,
but failed its preregistered SAH-volume-MAE gain (`0.0216 mL` versus required
`0.10 mL`). This means the remaining path needs either better support selection
or a target/evaluation design that improves spatial extent, not merely subtype
relabeling.

## Integrity

- Aggregate artifact SHA-256:
  `e76ff428809ae3d2f8e87c24519d8fe96555a13d5d9945f77bc2ff40f5c3d422`.
- Runtime: 343.58 seconds; peak allocated VRAM: 0.94 GiB.
- External reporting disabled; no row-level predictions were persisted.
