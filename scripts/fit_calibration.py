"""Nested-OOF evaluation and final fit of a lightweight calibration bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.calibration import (
    TriageCalibrator, assess_calibration_candidate, cross_validate_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_predictions", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "calibration" / "triage_calibration.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "calibration")
    parser.add_argument("--force", action="store_true", help="Save a rejected candidate explicitly")
    args = parser.parse_args()
    frame = pd.read_csv(args.oof_predictions)
    calibrated, _ = cross_validate_calibration(frame)
    assessment = assess_calibration_candidate(frame, calibrated)
    final = TriageCalibrator.fit(frame)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.report_dir / "nested_oof_calibrated.csv", index=False)
    (args.report_dir / "candidate_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8",
    )
    candidate_path = args.report_dir / "triage_calibration.candidate.json"
    final.save(candidate_path)
    if assessment["accepted"] or args.force:
        final.save(args.output)
        assessment["saved_to_submission_path"] = True
    else:
        assessment["saved_to_submission_path"] = False
    print(json.dumps(assessment, indent=2))
    if assessment["saved_to_submission_path"]:
        print(f"Saved accepted calibration bundle to {args.output}")
    else:
        print(f"Rejected candidate kept for audit at {candidate_path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
