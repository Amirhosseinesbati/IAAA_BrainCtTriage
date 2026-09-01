# Exp66 preregistration — true-positive SAH recovery objective

## Question

Does replacing exp65's row-positive Tversky with a genuinely positive-pixel
SAH objective generalize the adapter's selective train recovery to calibration
without worsening normal false positives, volume error, or any non-target
subtype?

## Evidence fixed before calibration

- Exp65 made no hard change. A one-epoch update probe proved that its optimizer
  moved 12.56%, but the learned residual was negative on every eligible true-SAH
  probe pixel.
- The old Tversky includes false-positive mass from millions of background
  pixels even on positive rows; its gradient aligned with the main loss at
  median cosine +0.9979 and learned global SAH suppression.
- The new positive-pixel NLL has median cosine -0.9919 against the main loss and
  contains no background false-positive term.
- With weight 0.03 and cap 12, a train-only one-epoch probe recovered 12.89% of
  eligible true-SAH pixels while converting only 0.00389% of true-background
  pixels and six other-hemorrhage pixels. No calibration or outer data informed
  this recipe.
- The earlier calibration margin audit found cap 8 below the true-SAH q25 and
  cap 12 theoretically reaches 82.15% of eligible missed SAH, while cap 16 is
  unsafe and excluded.

## Locked recipe

- Same audited exp61 warm start and patient split: outer=2, calibration=1.
- Frozen base model; 3,217-parameter, zero-initialized background-to-SAH head.
- hidden=16 and residual=`12*tanh`, applied only where incumbent argmax is
  background. Existing SAH and all IVH/IPH/SDH/EDH predictions remain immutable.
- epochs=6, patience=3, batch_size=16, lr=5e-4, weight_decay=1e-4.
- Main pixel-weighted Dice/Focal and hard-empty loss unchanged.
- SAH positive-pixel NLL weight=0.03; SAH Tversky=0; diffuse Tversky=0;
  classification loss=0; sampler study-balance power=0.
- Four-step smoke first, then exactly one calibration-only run. No weight, cap,
  threshold, sampler, or epoch sweep is authorized from calibration feedback.
- Outer evaluation is disabled.

## Promotion gate

All conditions must pass simultaneously:

1. Exact checkpoint/manifest provenance, exact locked recipe, best epoch >= 1,
   and no outer evaluation.
2. SAH Dice gain >= 0.01 absolute and SAH MAE improvement >= 0.10 mL.
3. FPR/volume-penalized checkpoint-score gain >= 0.001.
4. Normal FPR must not increase and presence F1 must not decrease.
5. Total-volume MAE and absolute bias must not worsen.
6. EDH/IPH/IVH/SDH Dice, AUC, MAE and bias must remain exact to 1e-10.
7. Any-ICH and macro-subtype AUC must remain exact to 1e-10.

Passing authorizes a locked patient-disjoint five-fold OOF run; it is not final
promotion or leaderboard evidence. Any failed condition means
`reject_before_outer`, no OOF, and no copy into `checkpoint/ich`.
