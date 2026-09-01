# MLS Vast experiment: mls-vast-exp14r1-w32-fold2-hybridsoft-repro

- Status: `failed_before_model_construction`
- Finished UTC: `2026-09-01T16:21:02.822564+00:00`
- Exit code: `1`
- Server commit: `a3cfa292`
- Vast instance: `49527185`
- MLflow run: `f2cc70364b4147c9a5a1cee533bbf7e4`
- Completed epochs: `0`
- GPU model compute: `none`
- Auto-destroy: `false`

## Failure boundary

The self-contained preprocessing completed, but it exposed an unversioned
dataset-contract change before training. The current base builder uses
`negative_ratio=3.0`; it produced 4744 source rows instead of the historical
2135-row source used by Exp10. The final generated dataset therefore contained:

| source | Exp14r1 rows | historical rows |
|---|---:|---:|
| target | 1781 | 1781 |
| legacy within-study negative | 2963 | 354 |
| hard within-study negative | 32 | 705 |
| clean-study negative | 644 | 644 |
| total | 5420 | 3484 |

The positive and clean-study counts agree. The 1936-row total difference is a
preprocessing-version difference, not transfer corruption. The expanded base
negative pool also left only 32 unused hard-within-study slices for the second
builder stage.

After preprocessing, MLflow created the run listed above, but dataset
initialization stopped because `Data/metadata/training_df.csv` was absent on
the clean server. No model was constructed, no forward/backward pass ran, no
epoch metric or checkpoint was written, and this attempt is not model-quality
evidence.

## Corrective action and integrity evidence

Commit `500dfaa` makes historical absolute Windows image paths portable and
falls back to DVC-tracked `Data/raw/training_df.pkl` for spacing and official
study truth. The historical processed folders were then transferred and
verified byte-for-byte:

- `mls_dataset`: 2136 files, 329207562 bytes, manifest
  `27651a89166c4508a7cc15cdb0185c89a7d08e2b0c1d011a4d8c13813781d353`
- `mls_multitask_v2`: 1351 files, 200975316 bytes, manifest
  `d7e791c58c638697bc9f266fee85bf8fbca7bfd849eaa9fafd48413636ca53a5`
- transfer archive SHA-256:
  `f4acbbe51e7958b53d2e932caafdb9c5a423f2668ef2dd2cd951ae857f4e9ece`

The 5420-row output remains preserved for audit at
`/workspace/iaaa_artifacts/failed_preprocess_exp14r1/Data_processed_5420`.
The immutable retry is `mls-vast-exp14r2-w32-fold2-hybridsoft-repro`; its
training fields remain exactly equal to Exp10.
