# Fracture sequence/MIL research decision — 2026-08-31

## Question and evidence boundary

SciSpace was queried for data-efficient skull-fracture CT detection and for
study-level head-CT methods that model ordered slices. The search was followed
by inspection of the primary SA-DMIL paper. This note records transferable
ideas, not a claim that results from other datasets will reproduce here.

Our internal evidence remains primary: simple neighbor-channel 2.5D stacking
and a nested logistic study aggregator did not improve the fracture OOF
baseline. Any sequence model must therefore beat the fixed-pooling OOF baseline
under nested, patient-disjoint validation; a literature result alone is not an
acceptance criterion.

## Most relevant findings

1. **Smooth Attention Deep MIL (SA-DMIL)** directly matches the ordered head-CT
   setting. A CNN extracts a feature vector per slice, attention pools the slice
   vectors, and the attention logits are regularized on the slice-neighbor graph.
   The first-order penalty is the graph-Laplacian quadratic
   `f.T @ L @ f`; the second-order alternative is `f.T @ L @ L @ f`. Training
   uses `(1-alpha) * BCE + alpha * smoothness`. On RSNA ICH, the paper reports
   that smooth attention improves scan-level AUC by about five percentage points
   over non-smooth attention; the best reported first-order setting used
   `alpha=0.5` and reached scan AUC 0.879. The experiment used 1,000 development
   scans, 150 held-out scans, five independent runs, three CT windows, and a
   single GPU. This is relevant evidence for the inductive bias, but its sample
   size and positive prevalence are much larger than ours.

2. A large skull-fracture study compared YOLOv3 object detection with a modified
   attention U-Net and reported higher external-test sensitivity and specificity
   for segmentation. The study used thousands of fracture and normal patients
   with expert annotations, so it supports a segmentation/ROI hypothesis but
   does not justify training a large 3D segmenter on our sparse boxes.

3. Skull R-CNN argues for two anatomy-specific changes: proposals constrained
   toward skull morphology and full-resolution features for thin fracture lines.
   In our setting, the low-risk analogue is a bone/skull ROI or loss weighting,
   not an immediate custom detector rewrite.

4. RibFrac challenge evidence favors segmentation/detection systems, large-scale
   pretraining, and anatomy segmentation as useful priors. Transfer to skull CT
   is indirect and must be validated internally.

## Recommended experiment after the YOLO26 replication gate

Run a **small frozen-feature SA-MIL experiment**, not an end-to-end 3D model:

- Freeze a selected detector and extract ordered per-slice embeddings plus its
  fracture confidence.
- Use a small gated-attention head with a first-order neighbor Laplacian penalty.
- Tune `alpha` only inside each training partition; include `alpha=0` as the
  non-smooth control.
- Preserve patient-disjoint five-fold OOF evaluation and compare against the
  predeclared fixed-pooling profile with paired bootstrap.
- Report macro AUC, worst-fold AUC, fold-wise deltas, and uncertainty. Do not
  select the architecture from pooled OOF alone.
- Reject the path if gains depend on a single fold or if nested selection does
  not beat fixed pooling.

This staged design is compatible with a 12 GB RTX 3060 because the detector can
be frozen and embeddings cached. It also tests the paper's central inductive
bias without spending resources on a large 3D network.

## Current decision

Do not interrupt the controlled YOLO26s replication. If replication is positive
or complementary, complete its OOF evidence first. If it is negative or mixed,
SA-MIL is the next principled sequence-context experiment; naive 2.5D stacking
should not be repeated.

## Primary sources

- Wu et al., [Smooth Attention for Deep Multiple Instance Learning: Application
  to CT Intracranial Hemorrhage Detection](https://arxiv.org/abs/2307.09457),
  2023.
- Shan et al., [Automated Identification of Skull Fractures With Deep Learning:
  A Comparison Between Object Detection and Segmentation
  Approach](https://doi.org/10.3389/FNEUR.2021.687931), 2021.
- Kuang et al., [Skull R-CNN: A CNN-based network for the skull fracture
  detection](https://proceedings.mlr.press/v121/kuang20a.html), 2020.
- Yang et al., [Deep Rib Fracture Instance Segmentation and Classification from
  CT on the RibFrac Challenge](https://doi.org/10.1109/TMI.2025.3565514), 2025.
