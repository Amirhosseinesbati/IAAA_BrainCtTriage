# Executable comparator repair and baseline reproduction protocol

Use `scripts/evaluate_mls_canonical_resource_cuda.py` for future fixed fold0,
seed42, epoch15 resource screens. Old A2--A6 gate launchers are historical only
and must not be recycled. This new entry point performs no training, automatic
replication, tuning, promotion or ZIP creation.

Before consuming results it requires explicit checkpoint inference fields to
match the corrected baseline: all10 pooling fields, image512/channels3/single
selector, three imaging windows, spatial-softmax/DARK, float32/no autocast,
inference batch6, clamp[0,30]. The comparison includes pinned inference source
and runtime versions. Changed/missing values fail before CUDA model loading.
Changes to architecture/preprocessing require a separately justified protocol;
they cannot be silently admitted as the same inference path.

First perform a full70-study baseline self-test with immutable Exp16 epoch15
SHA256`c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062`.
Every study must reproduce its immutable seed42 prediction within1e-5mm and
every aggregate metric within1e-6. These tolerances are fixed before outcomes,
not tuned in response to failure. Patient-grouped folds, full truth coverage,
checkpoint SHA/fold/seed/epoch, reference hashes and implementation hashes
must all match. Preserve raw-file-content and sorted SOP-UID fingerprints for
each study privately; the aggregate binds that private file's hash.

Candidate screens then require the verified baseline result SHA and unchanged
evaluator source. Their raw content and ordered SOP identities must match the
verified baseline before each study's inference. No grids or missing-study
fallbacks exist. CUDA guard plus exclusive campaign GPU lock apply. Unexpected
overlap is refused; never delete another process's lock. Do not print private
IDs/values, copy them locally or upload them to MLflow.

Unit tests cover each pooling/preprocessing field, missing fields, legacy profile,
clipping, source/runtime mismatches, canonical aggregation parity, finite values,
coverage and input identity/content changes. Tests run on the target server only;
pure contract/metadata checks are not CPU model inference. A full CUDA baseline
self-test is still required; unit success alone is insufficient.

Outputs use a fresh directory; partial private results/status are retained on
failure. No implicit resume or overwrite is implemented. The immutable resource
bounds and1e-8 rounding tolerance remain exactly those in the corrected protocol.
Baseline self-test cannot count as an upgrade. Even candidate resource success
cannot authorize deployment: three seeds, cross-fold frozen-Champion/oracle
triage checks and all five-fold final hard gates remain mandatory.

No15-minute scheduler is activated. Observe this one bounded verification by
its actual process handle; do not poll epoch logs or launch another training.

Pre-CUDA compatibility amendment: all21 initial contract tests passed, but the
historical baseline raw configuration lacks exactly `selector_head_mode`.
Its strict loader has always reconstructed a single-head model; provide a
named schema migration to `single` ONLY for the exact baseline SHA above.
Do not fill any other missing field, alter an explicit value, or extend this
exception to a different checkpoint. New candidate fields remain mandatory.
The result records whether this migration was applied. This amendment was made
before observing any fresh CUDA predictions, not to relax a numerical gate.
