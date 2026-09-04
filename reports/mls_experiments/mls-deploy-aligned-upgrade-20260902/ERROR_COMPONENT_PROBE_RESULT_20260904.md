# Error-component GPU diagnostic: prioritize outer-point refinement, not pose rectification

Previous turn advanced evidence by establishing raw annotation agreement without modifying labels. This turn completed the fixed GPU diagnostic (session 60402 exited 0). Both checkpoint states, including buffers, remained unchanged. The same 128 training samples and exact image/truth exposure digest were reproduced. No validation image or model update was used. Parallel/perpendicular synthetic geometry self-tests passed on the server; no local Python/model execution occurred.

Aggregate result local/remote SHA256: `a9f7a644b3fc1b8762f45452abc71ac28ee7428c6a4b21ebbd332186be188650`.

MLflow archival completed (session 93569 exited 0): pose, provenance and error-component aggregate JSONs were uploaded under `reports/post_a7_diagnostics` in run `5746a793c5d04da994935651fbaae5d4`. All three were downloaded again and matched source SHA256. No private rows were uploaded. This supersedes the pending-archive notes in the preceding two reports; receipt is `POST_A7_DIAGNOSTICS_ARCHIVE_RECEIPT_20260904.json`.

## Findings (training diagnostic, mm)

| Quantity | Baseline | A7 consistency |
|---|---:|---:|
| Full MLS MAE | 1.455074293 | 1.158305193 |
| True reference endpoints + predicted outer point MAE | 1.387634161 | 1.177020716 |
| Predicted endpoints + true outer point MAE | 0.728647503 | 0.596659121 |
| Outer-point mean absolute parallel error | 5.931536414 | 5.434726265 |
| Outer-point mean absolute perpendicular error | 1.515457775 | 1.282884422 |

Full MAEs reproduce the prior A7 translation diagnostic exactly. Correcting reference endpoints alone changes mean error only modestly for baseline and slightly worsens A7 mean error, consistent with cancellation between components. Correcting the outer point has the larger average diagnostic effect for both models. These are hypothetical substitutions, not deployable systems, additive causal attributions, or rigorous bounds on what a learned refinement can achieve.

The large outer-point Euclidean error is substantially parallel to the true reference line. Reducing that component alone need not improve MLS. Angle strata do not support the simple story that tilt is the dominant error: baseline MAE is 1.5393 for <=10 degrees (95 slices) versus 1.2126 for >10 degrees (33 slices); A7 is 1.2022 versus 1.0320. These groups differ in other characteristics and are training slices, so this is neither causal evidence nor a generalization test.

## Bounded next implementation decision

Deprioritize a stand-alone rigid pose rectifier. Investigate a reference-conditioned outer-point refinement head: retain full-image features and predict the reference endpoints, then condition refinement on those predicted endpoints to focus on the anatomically meaningful perpendicular displacement. This is a feature/representation change, not a renamed signed-offset loss (A2 already tried that).

Before any training, inspect existing head/feature interfaces and implement a minimal prototype with invariance/coordinate round-trip tests and one CUDA forward/backward memory check. Predicted geometry must be used in both training and inference; never use GT endpoints to choose deployment crops. Retain a fallback for invalid/short predicted reference lines and avoid tight crops or label exclusions derived from the single anomalous annotation. Do not constrain the outer point to the observed training parallel range. Keep original loss and selector initially to isolate the representation change; compare to an update-matched control rather than changing batch, supervision and architecture together.

Training launch still needs a fixed protocol, source/data pins, measured GPU feasibility and existing resource gates. A7 remains rejected on held-out gates. This diagnostic establishes neither a new champion nor final triage improvement; no submission ZIP is justified.
