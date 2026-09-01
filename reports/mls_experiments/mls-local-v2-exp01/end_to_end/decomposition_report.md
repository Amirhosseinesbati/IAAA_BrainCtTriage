# MLS end-to-end error decomposition

- Studies: `70`; slices: `1723`
- Slice selector AUC (annotated targets vs other slices): `0.9099`
- Target retrieval recall: top-1 `1.000`, top-3 `1.000`, top-5 `1.000`, top-10 `1.000`
- Mean annotated target slices inside selected top-3/top-5: `1.54` / `2.54`
- Selector gate misses at threshold 0.5: `12`

## MAE decomposition

| profile | MAE mm |
|---|---:|
| Oracle annotated targets, median | 1.7356 |
| Oracle annotated targets, p90 | 1.3022 |
| Predicted selector top-3, median | 2.0078 |
| Predicted selector top-3, p90 | 2.1349 |
| Predicted selector top-5, median | 2.1418 |

The oracle-target median is the approximate keypoint/measurement floor.  The
difference to predicted top-3 median estimates the added selector/pooling cost.
This is diagnostic on fold 0 and must not be treated as an independent test.
