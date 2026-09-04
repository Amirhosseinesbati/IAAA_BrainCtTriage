# Training-only translation evidence and next model experiment

## Outcome

Completed on RTX3090 in7.75s; peak allocated VRAM0.584GiB. No model updates,
no validation images, no CPU model execution. Four geometry tests plus seven
runtime-reference tests passed on the server. These are tests of diagnostic
implementation, not evidence of improved medical predictions.

Exactly the frozen128 positive training slices from89 studies were used.
Baseline Exp16 epoch15 remains unchanged. No private rows were persisted by
this probe or transferred. Both independent runtime controls and the negative
control have also been preserved locally and download-hash-verified in MLflow.

| Measurement | Horizontal +8px | Vertical +8px |
|---|---:|---:|
| Mean absolute MLS change,mm |0.785377|0.356826|
| Median absolute MLS change,mm |0.598351|0.226641|
| p90 absolute MLS change,mm |1.623546|0.717063|
| Maximum absolute MLS change,mm |2.605244|3.490469|
| Prediction crossings at3mm /128 |13|5|
| Prediction crossings at5mm /128 |7|1|
| Mean corrected landmark displacement,mm |1.055510|0.983860|
| Mean selector probability change |0.024280|0.010165|
| Mean overlap heatmap JS,nats |0.001877|0.001627|
| Baseline annotation MAE,mm |1.455074|1.455074|
| Translated annotation MAE,mm |1.347055|1.479588|
| True landmarks cropped |0|0|

The preregistered mechanism flag passed for both shifts (mean change>0.1mm or
p90>0.5mm). This means a potentially important invariance violation exists;
it does NOT pass any model-release or held-out improvement gate.

## Critical interpretation

- Ordinary translation augmentation is already implemented. Do not claim it
  is missing. Explicit coupled-output consistency is the new hypothesis.
- Horizontal translation improved sample MAE while vertical translation worsened
  it slightly. Therefore the evidence is instability, not universal degradation.
  Do not select +8px as favorable test-time preprocessing from this probe.
- Prediction crossings are local slice threshold changes, not counts of newly
  wrong study triage decisions or sensitivity/specificity estimates. Positives
  only; no inference about selector specificity on negative anatomy.
- Mean removed input energy was0.0344% horizontal and0.0544% vertical;
  maxima0.6303% and2.1750%. No labeled landmark was clipped. This mitigates but
  does not eliminate padding/cropping confounding; intensity mass does not
  directly measure anatomical importance.
- Small distribution JS coexists with larger decoded-coordinate changes, with
  individual landmark displacement up to24.17mm. DARK peak sensitivity and
  genuinely changed localization may both contribute. The aggregate cannot
  attribute the effect uniquely or prove that generic JS alone will fix it.
- The sample is known training data, not an independent generalization estimate.
  The baseline MAE differs from the older decoder probe because this probe uses
  explicit IEEE inference; do not count runtime drift as a model improvement.

## Next bounded experiment, not yet launched

This evidence justifies implementing a matched paired-view consistency trial,
not another architecture/pooling sweep. Keep current HRNet, DARK decoding,
selector and canonical aggregation. The existing timm features-only backbone
already traverses all HRNet multiresolution stages; its misleading stage1
comment is not evidence of a shallow backbone bug.

Before any full training: test coordinate/target transformation and overlap
masking, finite CUDA gradients to both views, and exact zero-weight behavior.
Use only training-fold images. Do not introduce elastic deformation or naive
heatmap flips. Preserve model/checkpoint provenance for any training-only
parameters without silently changing inference schema.

A valid causal comparison needs a paired-view supervised control and the same
views plus consistency, with equal initialization, sample exposure, optimizer
steps, BN updates/dropout RNG handling and numerical policy. Select one fixed
regularizer/weight prospectively based on scale/gradient checks using training
data only, not held-out tuning. Retain fixed epoch15; do not rename a fine-tuned
epoch20 checkpoint to15. Two matched runs must run sequentially under GPU lock.
Record both irrespective of outcome. Do not start seed/fold replication until
the corrected canonical resource gates pass; no automatic promotion.

No training was launched by this diagnostic. No better model, local release
checkpoint or submission ZIP is claimed. Goal remains the original final
triage improvement, with same-runtime multi-seed/cross-fold controls and all
frozen-Champion final gates still required. Cancelled15-minute monitor stays off.

## Artifact provenance

Remote:/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/train_translation_probe_20260904.json

Aggregate SHA256:
05f6e98700d5ae64211632ddf42e1e9f2802c377d37ca8aa3097bd59d8562b87

Probe source SHA256:
02171bdfe04a5e64740d4349749739151cff03399844671c09bba5acdf6cac8f

Protocol SHA256:
3bd6f5901e44bd41b57479a588ec92d115e11d1d355a0877c3dcafa550470cec

Qualification MLflow run:8478b358f7b84f47b41f3b0ca882152d;
artifacts:reports/same_runtime_qualification. Four aggregate artifact downloads
and four metrics verified. Receipt SHA256:
8b8d6d6f1ce8ae79603d8af8e7741878f2579f2afc0ed7ad9236e507939d9d48.

Scientific motivation and limitations are in TRANSLATION_PROBE_PROTOCOL_20260904.md;
the cited chest-radiograph segmentation paper is not evidence of MLS benefit.
