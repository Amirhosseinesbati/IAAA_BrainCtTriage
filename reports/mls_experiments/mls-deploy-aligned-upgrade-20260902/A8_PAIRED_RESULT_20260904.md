# A8 reference-conditioned refinement — completed, rejected

## Decision

A8 completed both fixed arms and the single preregistered canonical evaluation. The reference-conditioned refinement improved its matched newly-trained control on MAE, RMSE and the selection objective, but reduced all three threshold/boundary F1 measures. More importantly, it remained materially worse than the qualified existing baseline and failed the resource gates. It is not replication-, promotion- or submission-eligible. No epoch, pooling rule or threshold was selected after seeing validation results.

| Metric | Qualified existing baseline | A8 matched control | A8 refinement | Refinement minus control |
|---|---:|---:|---:|---:|
| Study MAE mm (lower) | 1.4705651353 | 1.9812330001 | 1.7534302088 | -0.2278027913 |
| RMSE mm (lower) | 2.4404310564 | 3.4077445932 | 2.8434135059 | -0.5643310873 |
| F1 at 1 mm (higher) | 0.8205128205 | 0.8250000000 | 0.7500000000 | -0.0750000000 |
| F1 at 3 mm (higher) | 0.8196721311 | 0.7692307692 | 0.7368421053 | -0.0323886640 |
| F1 at 5 mm (higher) | 0.7368421053 | 0.7000000000 | 0.6842105263 | -0.0157894737 |
| Boundary F1 (higher) | 0.7782571182 | 0.7346153846 | 0.7105263158 | -0.0240890688 |
| Selection objective (lower) | 1.9140508989 | 2.5120022308 | 2.3323775772 | -0.1796246536 |

Relative to the existing baseline, refinement increased MAE by 0.2828650735 mm, decreased 3-mm F1 by 0.0828300259, decreased Boundary F1 by 0.0677308024, and worsened the objective by 0.4183266783. The candidate's improvement over its weak matched control therefore does not establish an MLS upgrade.

## Validity and scope

- Both arms completed exactly 15 epochs / 8115 optimizer updates from the same immutable initialization.
- Per-epoch augmented input exposure hashes matched exactly between arms.
- Validation images used for training or checkpoint selection: zero.
- Evaluation scope: fixed fold0, seed42, epoch15, 70 studies, CUDA, same qualified runtime and unchanged pooling/gates.
- The control and candidate configurations differed only by `use_reference_refinement=false/true`.
- The comparison tests the complete refinement package; it does not isolate coordinate conditioning from added parameter capacity.
- The 70-study resource screen is not final triage validation. Since the candidate failed here, no downstream triage claim is made.

## Failure interpretation

The refiner learned a useful continuous-error correction relative to its matched control, as shown by lower MAE/RMSE, but that correction did not preserve clinically relevant threshold behavior. The negative shift in F1 at 1/3/5 mm and Boundary F1 is consistent with a smoother or biased correction that improves some large errors while moving more cases across decision boundaries. Its bias also shifted from -0.0508 mm in control to -0.6968 mm in refinement. This is evidence against promoting the current formulation, not evidence that reference conditioning is impossible in principle.

The unusually weak A8 control is also important: it means the newly-trained pair cannot be judged only by candidate-minus-control. The authoritative comparator remains the qualified existing baseline. Re-running the same experiment, selecting a different epoch, or tuning aggregation on the evaluated 70 studies would create adaptive leakage and is prohibited.

## Artifacts and checksums

- Pair aggregate: `A8_PAIR_AGGREGATE_20260904.json`; SHA256 `1d139bca0d9cccfe46cb73ba40937ffe74c862e96697e15a64e774349a773e80`.
- Control audit aggregate: `A8_CONTROL_AUDIT_AGGREGATE_20260904.json`; SHA256 `33f4615a62a8cf02662f974f7e9bc1e653c38d772e7228dd8b2c22f38603263d`.
- Refinement audit aggregate: `A8_REFINEMENT_AUDIT_AGGREGATE_20260904.json`; SHA256 `43c6425a6bf62949858fe7890c7d5f7951f9cb30751b82707d44989477d10083`.
- Control training summary: `A8_CONTROL_TRAINING_SUMMARY_20260904.json`; SHA256 `273cbded7c2d225af50ee0b0f317306cb907d5f324bcd9d3655e6adc3c8961cc`.
- Refinement training summary: `A8_REFINEMENT_TRAINING_SUMMARY_20260904.json`; SHA256 `34d7238116485940cfd7387d112546d78cf50d6dcff3296cbee21197c5c6d9fb`.
- Pair completion receipt: `A8_PAIR_COMPLETION_20260904.json`; SHA256 `e6342cd08ce19cee9d310d1750e974bf9616131b11a43c31d3f5e67881f36584`.
- Final MLflow archive receipt: `A8_FINAL_ARCHIVE_RECEIPT_20260904.json`; local/server SHA256 `9dfd5e130ba7cb6df648af5a8f03c2b31bcc2b1b2ec12496d54c4e173c43d509`.
- Control checkpoint SHA256: `e470b2b5bdbb79bf34fd8d2a4e30bc1ed4078b1686770f1ddff4d75680741b3e`; MLflow run `451470b102064180add9d0d21f2e45fe`.
- Refinement checkpoint SHA256: `a72ee65f00732b1cf1867e5576aa31e3368b988c5bddfa16a1b5e3e1cdd35c68`; MLflow run `02f60665f699444cb0b9500c6a2eaf9f`.

Both MLflow checkpoints and training summaries, plus both arm audit summaries and the pair summary, were independently downloaded and checksum-matched. Private per-study rows were neither copied locally nor uploaded as report artifacts. The rejected checkpoints remain recoverable from MLflow/server but were deliberately not copied into the local best-model directory.

## Next decision boundary

Do not replicate A8 and do not build a ZIP. A subsequent experiment needs a new training-only-supported hypothesis that directly protects threshold calibration/Boundary F1 and is compared against the qualified existing baseline under the same immutable screen. The final objective remains improvement of frozen-Champion triage Macro-F1 and Urgent F1; MLS-only resource metrics are screening evidence, not the endpoint.
