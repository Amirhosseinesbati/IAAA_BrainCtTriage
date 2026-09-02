# Exp71 result: selective gated residual train probe

Decision: `reject_before_any_calibration_or_outer`

MLflow run: `a878b2122aa14e2a95f3f8c919c3a2d9`

Git commit: `31d7181030b53b7999a2a1b81d195e0b73fc73ec`

This was an aggregate train-only diagnostic. No calibration, outer fold, test
data, row-level prediction, checkpoint, promotion, or leaderboard inference was
performed.

## Result

- 3,302 trainable parameters; the Exp61 incumbent remained fully frozen.
- 101 targeted tests passed on the server.
- Initial hard-mask identity and final foreground-support lock were exact.
- Error prevalence inside incumbent foreground was **0.5832%**.
- The gate activated on **10.9336%** of incumbent foreground.
- Gate error precision: **2.1753%**.
- Gate error recall: **40.7800%**.
- The gate provided only about 3.73x precision enrichment over prevalence,
  below the preregistered 10% precision requirement.
- SAH-to-IPH recovery: **0 of 232 (0%)**.
- Correct IPH harm: **0%**.
- Correct IVH/SDH/EDH harm: **0.2551%**.
- True-background subtype drift: **0.5392%**.
- Conditional accuracy delta: **-0.01533pp**.
- Conditional macro-recall delta: **-0.11039pp**.
- Six of thirteen preregistered gates failed.
- One epoch / 303 optimizer steps took 133.41 seconds; peak VRAM was 1.201 GiB.

## Interpretation

The supervised gate learned a non-random error signal and reached useful error
recall, but its precision was far too low for safe correction. It activated on
roughly one in nine incumbent-foreground pixels to find errors present in only
one in 171 pixels. The hard gate also routed the residual toward IVH/IPH/SDH
confusions rather than the target SAH-to-IPH errors.

Post-hoc threshold tuning is not authorized: the 0.5 threshold and all gates
were locked before execution, and using this train set to choose a new threshold
would optimize the same evidence twice. More importantly, threshold movement
would trade already-insufficient recall against already-insufficient precision;
it would not create missing separability.

Together, Exp67 through Exp71 show that small adapters over frozen incumbent
decoder features can be safe but do not provide enough transferable subtype
selectivity or volume correction. The next model experiment should change the
learned representation or supervision rather than add another frozen-feature
router/residual variant.

Aggregate JSON SHA256:
`e7ca2c749089271a8c1a9d12df071d1dc3975904d1f0208f30205e561462aba2`
