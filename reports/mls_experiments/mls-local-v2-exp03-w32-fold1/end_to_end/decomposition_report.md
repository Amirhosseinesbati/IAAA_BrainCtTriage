# MLS end-to-end error decomposition

- Studies: `67`; slices: `1346`
- Slice selector AUC (annotated targets vs other slices): `0.9212`
- Target retrieval recall: top-1 `0.971`, top-3 `1.000`, top-5 `1.000`, top-10 `1.000`
- Mean annotated target slices inside selected top-3/top-5: `1.46` / `2.33`
- Selector gate misses at threshold 0.5: `11`

## MAE decomposition

| profile | MAE mm |
|---|---:|
| Oracle annotated targets, median | 1.5252 |
| Oracle annotated targets, p90 | 0.6041 |
| Predicted selector top-3, median | 2.1999 |
| Predicted selector top-3, p90 | 2.2063 |
| Predicted selector top-5, median | 2.1324 |

The oracle-target median is the approximate keypoint/measurement floor.  The
difference to predicted top-3 median estimates the added selector/pooling cost.
This is diagnostic on fold 0 and must not be treated as an independent test.
