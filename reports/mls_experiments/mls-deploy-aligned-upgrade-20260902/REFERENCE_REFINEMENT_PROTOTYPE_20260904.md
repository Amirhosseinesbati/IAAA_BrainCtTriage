# Predicted-reference-conditioned outer heatmap prototype

Previous goal turn was progress: fixed CUDA error decomposition favored outer-point investigation over a stand-alone pose rectifier. This implementation is an opt-in prototype, not a released model or training launch.

## What changes

A full-field residual convolutional head receives existing HRNet features, detached bounded coarse heatmap context, and two coordinate fields parallel/perpendicular to the line between predicted endpoint peaks. It changes only outer-point logits. No image crop, warping, GT reference points, dataset-derived anatomical bounds, new loss, or selector branch is introduced. Full-image features preserve context.

Predicted discrete peaks are computed identically in train/eval. Fields use integer heatmap pixel centres, normalized by image diagonal. Very short (<4 heatmap pixels) or nonfinite reference predictions yield a zero residual. Endpoint logits remain unchanged by the residual head; shared features can still change them during later joint optimization. Nonfinite coarse outputs must still be rejected upstream, not treated as valid fallback predictions.

The final residual conv starts at zero, preserving base predictions exactly. Initial gradients reach its final layer; earlier residual layers receive gradients after that layer moves. Detaching geometry avoids extra gradients through endpoint selection but makes conditioning discontinuous; this is a deliberate tradeoff, not a differentiable rectifier. Coarse logits remain directly supervised. Parameter overhead and CUDA memory are measured by preflight.

## Test scope

Synthetic CUDA-only checks cover coordinate projection, translation consistency, 90-degree coordinate roundtrip, finite degenerate fields, identity initialization, unchanged endpoint outputs, fallback, active residual, feature/head gradients, and full HRNet-W32 batch5 512x512 forward/backward. Full-model initial eval parity checks heatmaps and selector against the same base instance. No patient data, pretrained checkpoint or optimizer step is used.

Geometric field equivariance does NOT prove whole CNN rotation/translation equivariance. One random batch does not establish long-run memory safety, performance or convergence. Release/training loaders are untouched; prototype wrapper state-dict names are not compatible with old checkpoints without explicit integration. No training may start until serialization, real training-loss compatibility, runtime consistency, fallback frequency and matched-control protocol are verified. Historical A2 signed-offset loss is not repeated here.

## Next required work

### Completed preflight

Session 97499 exited 0. All 16 CUDA checks passed on RTX3090. Full HRNet batch5/512 forward-backward peak allocated memory was 4.213667 GiB (no optimizer states; this is not total training VRAM). Extra parameters: 47,617. Initial heatmaps and selector were exactly equal to the same base model in eval mode. Feature and final-refiner gradients were finite/nonzero. No patient images, pretrained weights or optimizer steps were used. Results are in `REFERENCE_REFINEMENT_PREFLIGHT_20260904.json`; MLflow archival of this new preflight is pending.

Source hashes: module `f215d6ec8a73e4308d31366b0eeb97a28996f81c8af1226a73d2af001bf50cce`; test script `a2a2bdafefbb2705e28adcda3f65207b3ed0ed71424ffd286030d3da513335ca`. Local/server source hashes matched. No older model/config/loader file was changed.
Result JSON local/server SHA256 verified equal: `78aa095ec8c1dfd6ac31f03854f02e52a0b7a4a10c71c619afdfe8381f31f281`.

If preflight passes, integrate as an explicit opt-in configuration with old paths bitwise unchanged; add checkpoint roundtrip and both ordinary/extended forward contract tests. Preserve baseline loss, selector, batch5/update budget and final fixed epoch for an interpretable comparison. Decide and preregister initialization/control pairing before any candidate training. A7 is not a validated replacement initialization. No held-out threshold/crop tuning, automatic promotion or ZIP follows from this prototype.
