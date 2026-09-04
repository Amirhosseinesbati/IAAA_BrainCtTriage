# Geometry extreme: provenance resolved, anatomical correctness not established

Previous goal turn was progress: a completed pose diagnostic changed the next action. This turn verified all 1360 positive fold0-training CSV coordinate triples against original JSON keypoints on the server. No image reading, model execution, held-out statistics, data changes or new training occurred.

Script SHA256: `89e7521f4d1677a48dfa344038be13d678e22d2eb65a7a4a3072616e211c4c58`. Local and remote script hashes agreed; final process exited 0. Dataset, folds and raw metadata were checked against their existing pins. Raw JSON contents were hashed in deterministic CSV order; digest is in `TRAINING_GEOMETRY_PROVENANCE_20260904.json`. That digest records the input snapshot, not independent certification of correct annotations.

## Findings

Aggregate JSON copied locally; local and remote SHA256 both `4e8334d0cb3173f97f6489ca64696d04fbefc30306ee680693a750e3f37d968e`. MLflow archival of this and the preceding pose diagnostic is still pending; no private rows were transferred.

- All 1360 positive coordinate triples match original JSON within 1e-6 pixel. No coordinate is a zero placeholder.
- Exactly one positive slice exceeds 30 mm; it is also the sole slice whose third-point projection lies outside the reference segment.
- Its study has an official maximum over 30 mm. Nine positive slices belong to such studies; this does not mean nine extreme slices or nine studies.
- Across all 138 positive training studies, the largest absolute difference between annotation-derived maximum MLS and official study maximum is 0.00000197 mm.
- Source inspection shows the positive point triplets receive present masks and satisfy the unaugmented training eligibility rule. No anatomical exclusion exists in that path. This is not a measurement of actual stochastic sampled exposure or gradients.

Therefore the extreme is not explained by damaged transfer, CSV-vs-JSON coordinate mismatch or zero sentinel. Agreement with official maxima is not independent anatomical validation: maxima may derive from the same labels. Do not delete, relabel or clip this example solely because its geometry is unusual. No claim is made that it caused the failed A2-A7 experiments.

## Next decision toward the actual triage goal

Do not spend another full training run on the untested assumption that pose rectification fixes the error. The minimal discriminating GPU diagnostic is to decompose errors on the already frozen training-only sample into perpendicular/parallel third-point error and reference-line error, with fixed pose strata and sensitivity decomposition. Use baseline and A7 consistency at most, identical images, unchanged eval-mode weights, no held-out tuning and no optimization. A large third-point Euclidean error parallel to the line is not enough to motivate a new MLS architecture. Oracle reference-line substitutions, if measured internally, are diagnostic upper-bound comparisons, never deployable predictions or evaluation candidates.

This should choose between predicted-reference-conditioned refinement and a different route, not an open-ended diagnostic campaign. Any candidate remains subject to fixed resource gates, independent replication and final frozen-Champion triage gates. No new best model, promotion or submission ZIP is established.
