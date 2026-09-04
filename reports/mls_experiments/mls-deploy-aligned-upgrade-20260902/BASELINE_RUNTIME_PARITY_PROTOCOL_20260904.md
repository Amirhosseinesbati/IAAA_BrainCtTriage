# Fixed numerical runtime investigation, not checkpoint selection

The canonical fresh baseline audit completed all70 studies but failed the
unchanged reproduction tolerance:42 studies exceed1e-5mm, mean difference
0.006000233mm, max0.316395283mm. All boundary F1s match; MAE differs0.004110375mm.
Do not treat this as a model upgrade or declare the new evaluator validated.

Test the SAME baseline checkpoint and70 studies under exactly three conditions:

- Repeat batch6, cuDNN TF32 enabled (current default); tests repeatability.
- Batch16, cuDNN TF32 enabled; changes only inference batching.
- Batch6, cuDNN TF32 disabled; changes only convolution precision allowance.

All modes keep matmul TF32 off, cuDNN benchmark/deterministic false, no autocast,
unchanged shared reader/decoder/pooling/clipping. Verify full raw-file bytes
and ordered SOP fingerprints against the fresh baseline before each study.
No labels are used to pick a favorable numerical mode: compare reproducibility
to the fixed original and fresh references; metric changes are diagnostics.
No seed, checkpoint, threshold, pooling or blend search; no model training.

Rationale: [PyTorch2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
documents lack of bitwise equality across hardware/batched operations and the
precision implications of TF32 convolution. This supports testing, not assuming,
the mechanism. Maximum0.316mm is too large to dismiss without investigation.

Script `scripts/diagnose_mls_runtime_parity_cuda.py` pins references/source,
uses the campaign GPU lock and CUDA-only model execution, and emits aggregates
only. It suppresses the display of deidentified-UID format warnings, which
contain identifiers; actual missing/duplicate UID and reader errors still fail.
The result cannot satisfy baseline self-test, authorize training or release a
model. No monitoring automation is activated.
