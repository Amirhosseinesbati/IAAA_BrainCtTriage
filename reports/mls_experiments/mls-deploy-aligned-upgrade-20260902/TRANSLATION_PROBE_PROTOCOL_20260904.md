# Fixed training-only translation mechanism probe

Pre-execution hypothesis: an MLS estimate should be invariant under a small
rigid image translation; its landmarks should translate together. This is not
a claim that the current pipeline lacks augmentation: it already implements
rotation, translation and intensity jitter. Explicit output consistency would
be a distinct regularizer, if a measurable weakness warrants a matched trial.

Reuse the previously frozen 128 annotated fold0 training slices (89 studies),
sample SHA b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1.
Checkpoint: immutable Exp16 fold0 seed42 epoch15, SHA
c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062.
No held-out images, new sample selection, optimization, or TTA search.

Evaluate original, +8 horizontal and +8 vertical image pixels, batch8,
float32/no autocast, explicit IEEE convolution/matmul on3090. Eval-mode weights
and normalization frozen. Zero-pad, never wrap. An8-pixel input translation is
exactly2 heatmap pixels under the existing x_heatmap=x_image*128/512 convention.
Tests verify overlap alignment, zero-fill, identity and MLS translation
invariance. Do not naively horizontally flip heatmaps: the current convention
does not equate image reflection511-x to heatmap reflection127-x/4.

Record aggregate translation-corrected DARK landmark error, absolute MLS and
selector changes, prediction crossings at3/5mm, overlap-renormalized heatmap
Jensen-Shannon divergence, and baseline/transformed annotation MAE. Count
clipped true landmarks and removed input energy to expose boundary confounding.
No patient/slice rows leave memory; output only aggregates.

Before observing results, flag a potentially material mechanism if either
shift has mean absolute MLS change>0.1mm or p90>0.5mm. This flag does not prove
benefit or authorize automatic training. Any clipped landmarks or appreciable
cropped anatomy require interpretation. A subsequent trial, if justified, needs
paired-view supervised control vs the same views plus consistency: same
initialization, samples, optimizer steps, BN/dropout behavior and runtime.
No changes to canonical inference, pooling, comparator gates or Champion.

Scientific motivation: Bortsova et al.,2019,
[Semi-Supervised Medical Image Segmentation via Learning Consistency under Transformations](https://arxiv.org/html/1911.01218v1).
They tested transformation consistency beyond ordinary augmentation in chest
radiograph segmentation, including a labeled-only variant; not CT MLS evidence.
Our rigid-translation hypothesis is an extrapolation. Elastic transformations
need not preserve MLS and are not adopted. Their transductive SemiTC+ use of
validation/test images is incompatible with our leak-free protocol and excluded.
No benefit is inferred until matched MLS experiments and final triage gates pass.

One CUDA job under existing campaign lock. Never train/infer on CPU or locally.
No15-minute automation. Preserve aggregate and checksums locally and in MLflow.
