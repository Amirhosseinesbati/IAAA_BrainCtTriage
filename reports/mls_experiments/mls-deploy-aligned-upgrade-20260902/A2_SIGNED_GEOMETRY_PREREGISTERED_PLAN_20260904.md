# A2 signed-geometry: pre-registered resource gate

## Hypothesis

MLS is the perpendicular distance from the outermost falx point to the line
joining its anterior and posterior attachments. The current differentiable MLS
loss uses the absolute distance, so a mirrored outer point receives the same
MLS loss. In the 1,781 labelled target slices, annotation-defined signed
offset is nearly balanced (894 positive, 885 negative, 2 zero). A low-weight
signed-offset loss should disambiguate this geometry without altering the
production keypoint decoder, study pooling, or triage rules.

This is consistent with structural-midline work that constrains the geometry of
midline localization rather than treating MLS as unconstrained scalar
regression. It is intentionally narrower than adding another classifier head
or changing post-processing.

## Fixed resource-screen experiment

- Candidate: `mls-vast-deploy-aligned-a2-signed-geometry`, fold 0, seed 42.
- Single change: `signed_offset_loss_weight=0.10`; all other training and
  inference settings are copied from the strict deploy-aligned baseline.
- Epoch: 15 is the only CUDA audit checkpoint. Training runs through epoch 23
  solely to preserve the fixed schedule and recovery artifact.
- Held-out evaluation: all 70 fold-0 studies, CUDA-only, one fixed model
  member, zero inference failures.
- Reference: locked baseline fold-0 seed 42 from the same three-seed audit.
- No checkpoint, alpha, pooling, threshold, or loss-weight grid is allowed.

## Resource gates

The A2 seed-42 resource screen advances to seeds 2026 and 3407 only if all of
the following hold against the fixed seed-42 baseline:

| Metric | Required A2 result |
|---|---:|
| MLS MAE | <= 1.4709586392 mm |
| F1@3 mm | >= 0.8196721311 |
| F1@5 mm | >= 0.7368421053 |
| Boundary-F1 | >= 0.7782571182 |
| selection objective | <= 1.9044444028 |

Passing this screen is not a release claim. It only authorizes exactly two
additional fixed-epoch seeds on fold 0. Their median must then beat the locked
three-seed fold-0 baseline before A2 may expand to any new fold. Final triage
promotion still requires the frozen-Champion, deploy-aligned multi-fold gate.

## Hardware policy

The first A2 run keeps batch size 5 and FP32 to isolate the scientific change.
The RTX 3090 is used for faster CUDA execution, not as an unrecorded optimizer
change. A separate CUDA memory preflight may establish a larger safe batch for
later replications, but cannot retroactively change or rescue this resource
screen.
