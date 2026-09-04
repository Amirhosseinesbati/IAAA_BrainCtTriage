# A6 canonical geometry/selector factor-swap diagnostic

Frozen before mixed-factor results, after the same-pooling correction. This is
retrospective sensitivity analysis on already-used fold0, not model selection,
causal attribution, an independent validation result or permission to promote.

Use only the baseline and A6 epoch15/seed42 CSVs pinned by
POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json (SHA256
15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee).
Require the corrected aggregate JSON SHA256
0dd707cb86b39ef888ce56ee9fb29f455f6b6cfd4b5b870a32f6eeaa7dcf1b70.
Use its unchanged canonical pooling and clipping. Require exact study/truth and
ordered slice-index agreement, all70 fold0 studies and native metric reproduction.

Four fixed combinations only: baseline geometry+baseline selector (native),
A6 geometry+baseline selector, baseline geometry+A6 selector, A6+A6 (native).
Geometry supplies decoded scalar MLS and heatmap peak; selector supplies both
presence and peak probabilities (therefore gating, component selection and
probability weights). Heatmap guard remains0, so the peak scalar cannot influence
selection. Do not alter landmarks, thresholds, radius, quantiles or clipping.

Output aggregate MAE/RMSE/bias, threshold F1/confusions at3 and5mm, existing truth
strata, and counts of per-study threshold decisions changed versus baseline.
Native combinations must match the corrected report within1e-7 before mixed
results can be interpreted. Export no identifiers, individual values or slices.

The caches identify slices by series and sorted index, not SOPInstanceUID.
Index/count checks and prior native baseline reproduction support alignment but
do not independently prove physical slice identity. Any signal worth advancing
must be reproduced with current-reader CUDA inference and explicit slice identity
checks before a hybrid experiment or release claim. This diagnostic alone cannot
establish a selector training intervention or generalization benefit.

No training, model forward, fresh inference or scheduled monitoring is needed.
All lightweight cache/statistical operations run on the server, not locally.
