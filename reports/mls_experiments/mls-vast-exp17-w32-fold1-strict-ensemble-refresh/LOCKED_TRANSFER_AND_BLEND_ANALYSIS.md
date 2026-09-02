# Exp17 locked-transfer and blend analysis

Exp17 is an infrastructure success but not a production promotion. All 23
epochs, 11 CUDA-only checkpoint audits (737 study-checkpoint evaluations) and
66,528 pooling rows completed without model fallback or evaluation failure.

## Frozen gate

Under severity-window radius 3, selector gate 0.5, minimum three active slices,
weighted q0.75 and guard 0, the best Exp17 state produced MAE 1.399901,
Boundary-F1 0.793443 and objective 1.813016. It failed all gates against Exp09
epoch15 (1.258665 / 0.823729 / 1.611207), so Exp09 remains the fold1 member.

## Leakage-safe cross-fold test

The same snapshot/profile was selected using only fold0 Exp16 and fold2 Exp15r,
then applied unchanged to fold1. The chosen epoch17 severity-window profile had
radius 2, presence gate 0.3, minimum three active slices, q0.65, unweighted and
heatmap guard 0.5.

- historical Exp09 fold1: MAE 1.484237, Boundary-F1 0.742308, objective 1.999622;
- challenger Exp17 fold1: MAE 1.817143, Boundary-F1 0.698396, objective 2.420352.

Therefore the failure is not explained by fitting the wrong fold1 pooling row.

## Complementarity test

Four slice-level blends were screened from saved CUDA predictions. Under the
frozen production profile, none beat Exp09:

| Candidate | MAE | Boundary-F1 |
|---|---:|---:|
| Exp09 | 1.258665 | 0.823729 |
| 90% Exp09 + 10% Exp17 | 1.264027 | 0.805556 |
| 75% Exp09 + 25% Exp17 | 1.276441 | 0.796552 |
| 50% Exp09 + 50% Exp17 | 1.311159 | 0.793443 |
| 25% Exp09 + 75% Exp17 | 1.347769 | 0.789831 |

Exp17 is not a useful ensemble correction for the current fold1 member.

## Decision

The next experiment targets the conflation inside the selector rather than
repeating a seed: Exp18 separates binary target presence (gate/count) from
relative peak severity (ranking/weighting) while keeping total selector-loss
scale and the rest of the recipe fixed.
