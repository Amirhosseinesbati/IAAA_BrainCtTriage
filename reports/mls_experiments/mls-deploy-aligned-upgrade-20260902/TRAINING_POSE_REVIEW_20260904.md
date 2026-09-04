# Training-only pose diagnostic: result and next decision

The preceding user turn cancelled a schedule; it did not advance model evidence. This continuation executed a new metadata-only server diagnostic, with pinned labels, raw metadata and fold manifest. No model calls, images, training, held-out geometry analysis, promotion or schedule activation occurred.

## Reproducible result

Script: `scripts/audit_mls_training_pose.py`, SHA256 `c708ea172a5fdb72e5c8b0318a270bb4e22520130da39b8e6c02ace1e70bcb9c`.
Result: `TRAINING_POSE_AUDIT_20260904.json`; remote original at `/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/training_pose_audit_20260904.json`.
Local and remote SHA256 verified equal: `866ab69796a51606071c3af4ce18e97de2b4ef39c6f1d687cdaf5a407b011995`. This diagnostic has not yet been archived to MLflow.

Population: 2706 training rows / 268 studies; geometry on 1360 positive rows / 138 studies. Study identifiers are unique in the fold manifest and each patient belongs to only one fold. Coordinates are finite and inside the 512-pixel image. Segment lengths are nonzero. Analytical vertical-segment self-test passed.

The initial attempt deliberately stopped on blank CSV spacing fields. Aggregate inspection established both spacing columns were missing in all 1360 positives; coordinates had zero nonfinite values. The existing loader already recovers scalar x spacing from raw metadata. The corrected diagnostic uses the checksum-pinned raw PixelSpacing1 median per training study, for both axes, explicitly matching that convention. This does not prove physical pixel isotropy. No labels or loader were changed. The original diagnostic script remains backed up on the server as `audit_mls_training_pose.py.before_spacing_fix`. An inline inspection printed valid aggregate counts but subsequently exited with a PowerShell heredoc terminator error; it made no writes. The final script exited successfully.

Slice-weighted findings:

- Absolute line angle: median 5.60 degrees, p90 15.30, maximum 35.38. 362/1360 exceed 10 degrees; 38 exceed 20.
- Segment length: median 145.72 mm, p10 120.09, p90 160.90; minimum 35.49.
- Midpoint horizontal offset: p10 -19.46 mm, median -2.20, p90 14.90.
- Third-point parallel position / segment length: median 0.386, p10 0.307, p90 0.472; one projection outside the segment.
- Absolute perpendicular displacement: median 3.32 mm; maximum 103.35 mm. This extreme requires investigation before deriving anatomical bounds or another training intervention.

## Decision

Pose variation exists, but the result does not establish that rectification improves held-out MLS or final triage. Do not crop targets to the observed parallel range, remove the extreme, or call it annotation corruption without checking loader eligibility, missing-landmark conventions, raw annotation agreement and whether the same row drives both extreme measures. The next bounded action is a training-only aggregate provenance/eligibility audit of these unusual geometries. Raw rows remain server-side. If necessary, use a narrowly scoped GPU diagnostic after verifying the label semantics; no CPU model inference.

A7 remains rejected by the resource gates. This metadata result is not a new best model, not a leaderboard prediction, and not permission to relax gates or start an unconstrained hyperparameter sweep.
