# Exp70 result: population trust-region residual train probe

Decision: `reject_before_any_calibration_or_outer`

MLflow run: `680c3bd78f4c4e2ca16d202e2802a3e7`

Git commit: `4359e832ec9044c2c2eb60a0bad2e703638049f1`

This was a train-only aggregate probe. No calibration, outer fold, test data,
row-level prediction, diagnostic checkpoint, promotion, or leaderboard run was
performed.

## Result

- 3,285 trainable parameters; the complete Exp61 incumbent stayed frozen.
- 96 targeted tests passed on the server before execution.
- Initial hard-mask identity and final foreground-support invariance were exact.
- 66 of 232 incumbent SAH-to-IPH errors were recovered: **28.448%**.
- Harm to correct IPH: **0.1578%** (gate at most 0.5%).
- Harm to correct IVH/SDH/EDH: **0.0554%** (gate at most 0.5%).
- Subtype changes on true-background incumbent foreground: **0.4929%**
  (gate at most 0.5%).
- Conditional accuracy: `0.992426 -> 0.991659`, delta **-0.07671pp**.
- Conditional macro recall: `0.995270 -> 0.995233`, delta **-0.00372pp**.
- Eight of ten preregistered gates passed. The non-decreasing accuracy and
  macro-recall gates failed and were not relaxed after observing the result.
- One epoch / 303 optimizer steps took 85.70 seconds; peak VRAM was 1.185 GiB.

## Interpretation

The shared population denominator and low-capacity residual removed the Exp69
collapse: correct-other harm fell from 24.235% to 0.0554%, background subtype
drift from 14.746% to 0.4929%, and macro-recall loss from 10.638 percentage
points to 0.00372 points. The causal redesign therefore worked as a trust
region.

It did not produce sufficient selectivity for improvement. The adapter
recovered SAH and a small number of IVH decisions, but it changed more correct
IPH/SDH pixels than errors it corrected, leaving a small negative accuracy
delta. Since the train baseline macro recall is already 99.527%, this probe
cannot justify calibration merely because the miss is numerically small.

A lower learning rate or smaller correction coefficient would mostly shrink
both benefit and harm; it does not address the observed error-to-correction
ratio. The next experiment must change the decision mechanism or the target
representation, and must be preregistered before any further calibration or
outer use.

Aggregate JSON SHA256:
`beabdd91b475cb70a9f359d3b8d8bf9cba0816b27b10e46710b32d4e15c61921`
