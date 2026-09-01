# MLS end-to-end error decomposition

- Studies: `67`; slices: `1466`
- Slice selector AUC (annotated targets vs other slices): `0.9322`
- Target retrieval recall: top-1 `1.000`, top-3 `1.000`, top-5 `1.000`, top-10 `1.000`
- Mean annotated target slices inside selected top-3/top-5: `1.52` / `2.48`
- Selector gate misses at threshold 0.5: `3`

## MAE decomposition

| profile | MAE mm |
|---|---:|
| Oracle annotated targets, median | 2.0849 |
| Oracle annotated targets, p90 | 1.3658 |
| Predicted selector top-3, median | 2.2936 |
| Predicted selector top-3, p90 | 2.4971 |
| Predicted selector top-5, median | 2.2506 |

The oracle-target median is the approximate keypoint/measurement floor.  The
difference to predicted top-3 median estimates the added selector/pooling cost.
This is diagnostic on fold 0 and must not be treated as an independent test.
