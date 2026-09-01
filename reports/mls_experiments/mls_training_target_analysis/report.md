# MLS training-target audit

This audit uses only label CSVs and geometry; no image or model was loaded.

## Dataset contract

- Rows: `3484`; studies: `338`.
- Target rows: `1781` across `177` studies.
- Negative rows: `1703` across `338` studies.
- Target annotations/study range from `5` to `27` (median `10.0`), so uniform row sampling gives highly unequal study weight.
- Spearman correlation between annotation count and study MLS: `0.292`.

## What the slice annotations represent

The study label is compared with statistics of the exact MLS reconstructed from every annotated slice's three keypoints and DICOM pixel spacing.

| Slice-geometry pooling | Study MAE (mm) | Study bias (mm) |
|---|---:|---:|
| min | 7.1027 | -7.1027 |
| median | 3.6393 | -3.6393 |
| p75 | 2.3585 | -2.3585 |
| p90 | 1.2876 | -1.2876 |
| max | 0.0000 | -0.0000 |

Best pure label-geometry statistic: `max`. This establishes whether study-level high-quantile pooling is intrinsic to the annotation contract rather than a post-processing accident.

## Training implications

1. Sampling should balance studies within target/non-target classes; balancing rows alone overweights heavily annotated studies.
2. The heatmap head should continue learning local three-point geometry. The final study target must be formed by a robust high quantile across target slices.
3. Checkpoint selection must include a study-level aggregation metric; slice MAE or selector AUC alone is insufficient.
4. The documented extreme target must be handled explicitly rather than silently dominating the regression loss.

## Extreme studies

| Study | GT MLS | Target slices | median geometry | p90 geometry | max geometry |
|---|---:|---:|---:|---:|---:|
| 271528 | 103.347 | 9 | 11.316 | 31.005 | 103.347 |
| 1062 | 23.386 | 9 | 19.886 | 22.628 | 23.386 |
| 1080 | 22.851 | 15 | 11.643 | 17.177 | 22.851 |
| 271003 | 22.294 | 11 | 16.554 | 19.343 | 22.294 |
| 271032 | 21.169 | 9 | 18.838 | 21.025 | 21.169 |
| 272190 | 20.586 | 12 | 14.113 | 20.313 | 20.586 |
| 270967 | 20.408 | 27 | 14.530 | 18.865 | 20.408 |
| 9248 | 20.405 | 11 | 18.112 | 20.065 | 20.405 |
