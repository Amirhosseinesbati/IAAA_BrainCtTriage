# Training boundary exposure: scarcity and missing-peak hypotheses not supported

## Outcome

The fixed fold0 training-only audit completed on the target server. Four unit
tests passed, all six pinned inputs matched, and study/patient separation was
verified. No images, checkpoints, model inference or training were used; no
held-out statistic or private prediction was returned. Source commit028d557.

| Fixed +/-0.5mm band | Near3mm | Near5mm |
|---|---:|---:|
| Positive training slices | 129 | 91 |
| Represented training studies | 75 | 56 |
| Expected positive draws per epoch | 128.289 | 90.498 |
| Fraction of positive sampler mass | 9.485% | 6.691% |
| Studies with official maximum in band | 21 | 13 |

Training comprises2706 rows/268 studies, including1360 positive rows from138
studies. Batch5/drop_last gives541 optimizer steps and2705 consumed draws per
epoch. Expected positives are1352.5; a batch has no positive with probability
3.125% under the replacement sampler. These are mathematical expectations,
not observations from a sampled epoch or measured gradient contributions.

## What this rules out, and what it does not

1. **No evidence of a relative lack of3mm examples versus5mm.** The former has
   more nearby slices and represented studies. This does not establish adequate
   statistical power or representativeness, but defeats that simple explanation
   for A6 improving5mm while regressing3mm.
2. **The annotated maximum is present in training.** For all138 positive
   studies, max MLS reconstructed from annotated geometry matches the official
   maximum within approximately0.000002mm. No above3mm or above5mm study lacks
   an annotated slice crossing its boundary. Missing training peak labels are
   therefore not supported as the explanation. This is not a full data/reader
   audit or a guarantee that the model selects the peak at deployment.
3. **Cross-level labels differ as expected.** At3mm,426 positive slices lie
   below3mm despite their study maximum being above it; at5mm the count is379.
   They account for31.324% and27.868% of positive exposure, respectively. These
   are legitimate local-versus-maximum differences, NOT evidence of bad labels.
   Replacing local geometry targets with study maxima would corrupt supervision.
4. **Study exposure is moderately, not catastrophically, concentrated.** The
   positive-study effective count from sampler masses is126.149 out of138;
   positive rows per study range5--23, median9. This metric alone says nothing
   about independence of slices or optimal sampler choice.
5. **The selector is not an explicit boundary classifier.** Its positive target
   quantiles(min/p10/median/p90/max) are0.75/0.775815/0.890796/1.0/1.0. It encodes
   targetness plus relative severity, while current local MLS loss already
   includes1/3/5mm boundary BCE. Neither mechanism is absent from the code.

## Reconciliation with earlier experiments

- Exp12 full study-balanced and Exp13 hybrid study-balanced samplers already
  failed their fixed fold2 production tests. Do not repeat them as a new idea.
- A1's independent ordinal head failed its three-seed fold0 test; that is not
  evidence that boundary loss was missing beforehand.
- A3 already connected positive slice bags to the official study maximum. Its
  corrected same-pooling result did not pass. A4/A5 tested selector ranking.
- Exp18/19/20 already tested dual selection and regression complements. Exp19's
  fixed fold0 complement passed its historical component gate; Exp20's fixed
  fold2 complement failed. The conservative historical recipe and standalone
  replacement are distinct claims. Do not summarize all dual-head work as
  uniformly useless or uniformly successful.
- A6 has a real5mm gain and3mm loss under the corrected canonical comparison.
  Neither simple geometry/selector swap fixes both. The present training
  metadata does not causally explain those held-out errors.

## Decision and next required work

No new training or model release is justified by this diagnostic alone. In
particular, do not launch boundary oversampling simply because the3mm gate
failed. Preserve the current Champion and all final triage gates.

Before another experiment, replace the legacy resource-comparison entry point
with a fail-fast canonical-inference contract (pooling AND clipping), tested
against the immutable baseline. Old A2--A6 launchers still invoke the legacy
candidate profile and must not be recycled. Then choose one scientifically
distinct training intervention with matched optimization, using targeted
research when needed. Neither this exposure audit nor repeated fold0 selection
establishes final triage improvement. No submission ZIP is authorized.

## Reproducibility and limits

Script `scripts/audit_mls_training_boundary_exposure.py` SHA256:
`63cdd28d6ba17461de73830c619ad2114a826d23256364e910d7e56db86eff05`.
Protocol SHA256:
`e757ca06a68434ba8b04ba73480d99e5358121a87d8bff1df73c68834609ac12`.
The initial attempt failed before producing statistics because the slice CSV
has no official study maximum. The corrected script uses the existing loader's
metadata attachment with pinned raw metadata; no training code was changed.
All statistics are unaugmented: positive coordinates are initially in bounds,
but this does not test post-augmentation visibility. The current heatmap helper
sets mask1 for each provided coordinate; it does not reject out-of-frame points.
No claim about actual augmentation failures is made here.

The15-minute monitor remains paused. Only aggregate JSONs and receipts are
transferred locally; data and private predictions remain on the server.

Aggregate `train_boundary_exposure_20260904.json` SHA256:
`67d5cc2624f383c8a9b5fa394e8ce489bfdf8d0a974a4f8143c6db3f63ce43b3`.
Receipt `train_boundary_exposure_mlflow_receipt_20260904.json` SHA256:
`11aeb19ff2c290a19d0dfdf8492dd9cc95a51a528726529e57dd9498e78fcb44`.
Remote originals: `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/`.
MLflow run `9b8e9fc5996a42549e3aca5aa40763d7` remains FINISHED; seven aggregate
metrics matched independent readback. The script/protocol/result are logged
under `reports/training_boundary_exposure`.
