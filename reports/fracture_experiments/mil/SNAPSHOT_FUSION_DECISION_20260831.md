# Fracture Snapshot Fusion Decision — 2026-08-31

## Decision

Keep the existing fixed-0.45 YOLOv8s + SA-MIL package as the incumbent and promote
the epoch-10/15 snapshot model as a separate A/B leaderboard candidate. Do not
replace the incumbent before a real leaderboard comparison because the paired AUC
confidence interval still crosses zero.

## Leakage-controlled evidence

- Incumbent deployable five-fold macro AUC: `0.907808`
- Plain snapshot fusion macro AUC: `0.916479`
- Decision-preserving cross-fit macro AUC: `0.917025`
- Cross-fit improvement over incumbent: `+0.009217`
- Decision-preserving deployment-score diagnostic AUC: `0.919338`
- Decision-preserving worst-fold AUC: `0.864169`
- Cross-fit F1: `0.548387` (identical to the incumbent by construction)
- Cross-fit precision / recall: `0.500000 / 0.607143`
- Paired stratified 50,000-bootstrap AUC difference 95% interval:
  `[-0.013024, 0.009198, 0.031567]`
- Bootstrap probability that the candidate is not better: `0.20136`

The first direct threshold calibration of the plain fusion was rejected: its
leave-one-fold-out F1 fell to `0.394366`. The accepted mapping therefore preserves
the incumbent positive/negative decision and uses snapshot fusion only to improve
ranking within each side of the boundary. This retains the incumbent classification
behavior while allowing a higher AUC.

## Package gate

- Candidate: `yolov8s-epoch10-15-5fold-sa-mil-snapshot-fixed040-v1`
- Contents: 10 optimizer-stripped YOLOv8s detectors and 15 SA-MIL heads
- Model artifact count: `25`
- Model artifact bytes: `227,559,592`
- Assigned-fold parity: passed for detector, MIL, snapshot pooling, empirical CDF,
  fusion and decision mapping
- DICOM-to-image parity: exact on three representative studies
  (`max_abs_difference=0`)
- Peak GPU allocation: about `1.91 GB`
- Mean end-to-end model runtime: about `5.51 s/study`
- Projected mean runtime for 68 studies: about `374 s` (`6.2 min`)

## Interpretation

The candidate is technically deployable and materially promising, but the sample
contains only 28 positive studies and Fold 4 still loses `0.00937` AUC versus the
incumbent. Its result must therefore be treated as an A/B candidate, not proof of a
leaderboard gain. The next decisive evidence is a controlled real-leaderboard
comparison with identical non-fracture components.

MLflow aggregate evaluation run:
`26204a8dfee44841915e7336ba05f1e7`. Per-study predictions and calibration arrays
remain private on the Vast workspace and were not uploaded.
