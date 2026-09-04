# Training-boundary exposure audit (before any new intervention)

Fixed scope: fold0 TRAINING metadata only, labels/folds/source/manifest SHA-pinned.
Run on the target server. No image, checkpoint, model inference, training,
held-out label statistics, prediction inspection, weight grid or threshold search.
Use the actual slice-class-balanced weight builder; compute expected draws under
replacement sampling with baseline batch5/drop_last. Check true patient and
study disjointness against the immutable fold manifest.

Questions fixed before results:

1. How many training studies/positive slices and expected draws fall around
   3mm versus5mm (fixed +/-0.5mm bands and explicit fixed strata)?
2. How often is a locally low-MLS slice from a study whose official maximum
   crosses a boundary? This is cross-level target disagreement, NOT automatically
   mislabeled data: per-slice geometry and study maximum are different targets.
3. Does each above-threshold positive study have an annotated slice reaching
   that boundary? Compare the max annotated geometry and official maximum.
4. How concentrated is positive exposure across studies, and what range do
   peak-aware selector targets cover under the existing0.75+0.25*relative rule?

Baseline and A6 both use slice-class-balanced sampling and threshold BCE at
1/3/5mm. Do not call either an absent feature. Full study balancing (Exp12) and
hybrid balancing (Exp13) were previously rejected on fold2. A1 independently
added an ordinal head; A3 added a study-bag objective; A4/A5 ranking objectives
were tried; dual selector Exp18/19/20 has prior full-study evidence. A6 changed
training geometry. A new intervention needs a distinction from these attempts.

Limitations: metadata exposure is not a measured training gradient or realized
epoch sample histogram. Rigid augmentation can affect coordinate masks/visibility;
the present audit is unaugmented. Hard threshold loss already encourages margins;
cross-level disagreement alone cannot justify replacing local targets with study
labels. Do not launch another training solely because a bin is small.

Acceptance: unit tests pass on target server, pinned input hashes match,
finite aggregate JSON only, local transfer SHA verified. This diagnostic cannot
promote a model or reopen a failed experiment. No scheduler reactivation.

Implementation correction before first result: the slice CSV has no
`study_mls_mm`; the initial metadata audit stopped with AttributeError before
producing statistics. Use the existing loader's `_attach_spacing` on training
rows, with its actual `training_df.pkl` pinned to SHA256
`0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d`.
The helper reads shared raw metadata to resolve official maxima/spacing but
returns only requested training rows. No held-out statistics are exposed or
used for interpretation. No outcome-driven change to the fixed questions/bins.
