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

from src.evaluation.calibration import TriageCalibrator, cross_validate_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_predictions", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "calibration" / "triage_calibration.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "calibration")
    args = parser.parse_args()
    frame = pd.read_csv(args.oof_predictions)
    calibrated, metrics = cross_validate_calibration(frame)
    final = TriageCalibrator.fit(frame)
    final.save(args.output)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.report_dir / "nested_oof_calibrated.csv", index=False)
    (args.report_dir / "nested_oof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved final calibration bundle to {args.output}")


if __name__ == "__main__":
    main()
