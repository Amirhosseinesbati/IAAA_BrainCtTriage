# A6 training-only geometry postmortem

Frozen after the A6 fixed resource rejection, before running this diagnostic.
Purpose: discriminate training-localization degradation from a discrepancy that
appears mainly at study deployment. It cannot by itself identify a unique causal
mechanism or establish generalization, and cannot reverse A6's rejection.

Use exactly the baseline diagnostic's128 hash-selected positive training slices
(89 studies), no augmentation, processed PNG, fold0/seed42. Reuse its selection
function and assert identical sample digest, labels and fold hashes. No held-out
image is allowed. Do not pick cases by error, confidence or A6 output.

Compare two immutable epoch15 checkpoints: baseline Exp16 (SHA256
`c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062`)
and A6 (`88b094341b260d48c90a2a1e12772c5bd5d82ac898e509db7ed7762d0b44aec6`).
Use the same batch8 CUDA inference, loading one model at a time under the global
GPU lock. No optimization, no checkpoint writes, no CPU model fallback.

For each model summarize three fixed decoders: legacy global softargmax,
localsoftargmax(radius6, temperature1) and historical DARK. Report per-landmark
mean/median error, slice MLS MAE/bias/F1 at3 and5mm, and global/local-versus-DARK
MLS discrepancy. Also report per-landmark fraction of ground-truth heatmap
centers outside the argmax-centered radius6 square. This last statistic measures
whether the selected training geometry window includes the target, not whether
that exclusion uniquely caused the final error.

Retain aggregate data only. Reproduce the previous baseline DARK/global MAEs
within1e-4 before interpreting differences, otherwise flag the diagnostic invalid.
Positive-only training slices do not evaluate false-positive selector behavior,
unannotated slices, DICOM preprocessing differences or study-level pooling.
If training DARK localization already deteriorates, inspect optimization before
adding selector complexity. If it does not, the next targeted diagnostic should
separate selector/pooling/preprocessing; do not assume which is responsible.
No new model training or parameter search is authorized by this diagnostic alone.
