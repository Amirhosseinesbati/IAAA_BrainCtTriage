# A4: sparse same-study selector-ranking screen

- Status: preregistered before any A4 CUDA outcome.
- Scope: one RTX 3090 run, fold 0 / seed 42, fixed epoch 15 only.
- Compute: CUDA-only; exactly one GPU process; no CPU model fallback.

## Evidence and hypothesis

The completed A3 study-bag attention auxiliary was rejected on all five fixed
resource gates (MAE `2.221690`, F1@3 `0.739726`, F1@5 `0.681818`, boundary-F1
`0.710772`, objective `2.800145`). Its aggregate diagnostic found both low-MLS
overestimation and severe-MLS underestimation, so neither a global calibration
shift nor a second attention-pooling attempt is justified.

The frozen deployment path nevertheless still uses the selector ordering to
choose an active local slice component before aggregating MLS. Existing
`peak_aware_soft` BCE supplies a per-slice soft target but does not explicitly
penalize an inversion between two annotated slices of one study. A4 tests only
that local ordering: for a same-study pair with absolute annotated local-MLS
difference >= `1 mm`, the peak selector logit of the higher-MLS slice must be
larger. Geometry prediction, study pooling, sampling and inference stay fixed.

This is a deliberately smaller intervention than A3. Attention MIL is a valid
general bag-level mechanism, but it is not automatically appropriate for a
deployment rule based on local ranking; the rejected A3 result is direct
evidence against repeating it here. The pairwise logistic form follows the
standard RankNet formulation, while the MRI/CT MIL literature supports treating
slices as related instances without assuming that an untested bag surrogate is
optimal.

Primary references: [RankNet/listwise survey](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf),
[attention MIL](https://proceedings.mlr.press/v80/ilse18a.html), and
[CT volume-as-bag MIL](https://pubmed.ncbi.nlm.nih.gov/34040275/).

## Fixed implementation

`MLSPositiveStudyPairDataset` enumerates only two target slices from one
training study. Every fourth ordinary batch, the existing model processes that
pair, and `within_study_pair_rank_loss` applies a binary logistic loss to the
peak-logit difference. The auxiliary has weight `0.10`, no gradient through a
study-level predicted MLS, and a zero contribution for ties/near-ties. A3's
`study_bag_loss_weight` is explicitly zero.

The ordinary loss, HRNet-W32, image size, strict determinism, fixed fold/seed,
23 epochs, and fixed epoch-15 audit are held constant. Batch size is raised
from 5 to 10 only after a separate CUDA forward/backward preflight on the
actual RTX 3090; an OOM or non-finite result fails closed and the batch is
reduced before training, with the final value recorded.

## Resource gate and stop rule

The exact unchanged resource screen used by A2/A3 is reused on the fixed 70
fold-0 studies. To authorize *only* the remaining two fold-0 seeds, A4 must
satisfy all five predeclared limits:

| Metric | Required |
|---|---:|
| study MAE | <= 1.470959 mm |
| F1 at 3 mm | >= 0.819672 |
| F1 at 5 mm | >= 0.736842 |
| boundary F1 | >= 0.778257 |
| objective | <= 1.904444 |

Any failed limit means `rejected_stop_a4_expansion`: no A4 seed replication,
no cross-fold training, no checkpoint promotion, and no submission ZIP. A pass
is only a resource screen; it still requires the frozen-Champion triage screen
and all-fold final promotion gate before a release can exist.

## Tracking and privacy

MLflow receives normal training aggregates and fixed-audit aggregate metrics.
Private per-study predictions remain only in the server artifact root and are
not copied locally, tracked, or logged to MLflow. The decision JSON, audit
status/metrics and their SHA-256 values are copied only after terminal status.
