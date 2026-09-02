# Conservative five-checkpoint package integration plan

## Frozen package

The package contains three deployable MLS members but five exact checkpoints:

- fold0 member: Exp16 best-selector-AUC epoch16 baseline plus 10% Exp19
  epoch21 slice `mls_mm`;
- fold1 member: Exp09 epoch15 baseline plus 10% Exp18 epoch21 slice `mls_mm`;
- fold2 member: Exp15r epoch17 unchanged;
- final MLS output: median of the three locked member aggregations.

For fold0/fold1, selector presence, ranking probability and heatmap peak are
copied exactly from the baseline checkpoint. The dual-selector checkpoint may
alter only slice-level `mls_mm`. Exp20 is excluded. No weight, checkpoint,
threshold or pooling adjustment is permitted after build.

## Build gates

1. Every checkpoint must match its frozen byte size and SHA-256.
2. The verified extracted Exp16/Exp15r package is read-only; the shared
   submission source is not mutated.
3. The deterministic ZIP uses `ZIP_STORED`, stays below one GiB, contains no
   data/report/checkpoint source paths, has no duplicate members, and passes
   `ZipFile.testzip()`.
4. The archive manifest covers every payload member and the model manifest
   names the five-checkpoint recipe and 204-study OOF evidence.

## CUDA integration gates

1. The extracted package must import and load exactly five MLS checkpoints on
   the RTX3060; all neural-network forward tensors and outputs remain CUDA.
2. A full `model.py` smoke must return the complete finite competition schema
   without OOM and record runtime plus peak VRAM.
3. The packaged MLS runtime must be evaluated on all 204 disjoint held-out
   fold0/fold1/fold2 studies. Raw baseline and dual-selector slice predictions
   are compared with their independent CUDA audit caches; member-level hybrid
   aggregation is compared with the frozen OOF evaluator.
4. Allowed numerical residuals are: zero slice-index mismatch, selector
   `<=2e-6`, heatmap peak `<=2e-8`, slice MLS `<=2e-5mm`, and aggregated member
   `<=2e-5mm`. OOF MAE, Boundary-F1 and objective must reproduce within
   `1e-6`.
5. Raw per-study integration outputs remain on Vast. Only aggregate status,
   parity, runtime and memory reports may be copied locally or logged to
   MLflow.

Passing these gates authorizes copying the package candidate to the local
workspace. It does not authorize claiming leaderboard superiority; the next
gate is an official limited submission.
