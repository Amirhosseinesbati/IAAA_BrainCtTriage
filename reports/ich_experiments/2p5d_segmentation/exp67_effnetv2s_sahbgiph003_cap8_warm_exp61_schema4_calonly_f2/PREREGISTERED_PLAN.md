# Exp67 selective background/IPH→SAH calibration screen

## Authorization evidence

The train-only probe at MLflow run `f9eda9a494724ea39af284da483ec1f1`
passed all four preregistered selectivity gates: 11.25% recovery of eligible
true-SAH pixels predicted as IPH, 0/43,307 correctly predicted true-IPH pixels
converted, 50% IPH-support conversion precision, and 0.00637% background
conversion. Precision passed exactly at the boundary and only 80 positive
IPH-confused SAH pixels were eligible, so this is treated as fragile evidence
that authorizes one calibration screen—not as model success.

## Locked recipe

- Same audited exp61 checkpoint, schema4 manifest and patient-grouped split.
- Train folds 0/3/4; calibration fold 1; outer fold 2 remains unread.
- Frozen exp61 base; zero-initialized 3,217-parameter residual head.
- Support: incumbent argmax background or IPH; only the SAH logit can change.
- Cap=8, hidden=16, AdamW lr=5e-4, weight decay=1e-4.
- Six epochs maximum, patience=3, batch=16, seed=42.
- Main segmentation objective plus true-SAH-pixel NLL weight 0.03; legacy
  positive-row SAH Tversky=0; sampler study-balance=0.
- Checkpoint selected only by the preregistered fpr-volume-penalized score.
- No cap, loss-weight, seed, support or checkpoint sweep is allowed.

## Promotion gate

All provenance and recipe checks must pass. Quality requires:

- SAH Dice gain >=0.01, SAH MAE improvement >=0.10mL and checkpoint-score
  gain >=0.001.
- IPH Dice loss <=0.005, IPH MAE worsening <=0.10mL, absolute IPH-bias
  worsening <=0.10mL and IPH AUC loss <=0.005.
- No worsening of SAH/Any/macro AUC, normal FPR, presence F1, total-volume MAE
  or absolute total-volume bias.
- EDH, IVH and SDH Dice/AUC/MAE/bias remain exact to 1e-10.

Failure rejects the candidate before outer and closes the frozen relabel branch.
Passing only authorizes the locked recipe for patient-disjoint OOF; it is not
checkpoint promotion or a leaderboard claim.
