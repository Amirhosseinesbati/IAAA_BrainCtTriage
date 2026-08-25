"""Evaluate an OOF intermediate CSV with official triage and Macro-F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.metrics import compute_competition_metrics
from src.evaluation.report import build_error_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "oof")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    errors = build_error_table(frame)
    metrics = compute_competition_metrics(
        errors["triage_class"], errors["pred_triage"],
        patient_ids=errors["patient_id"] if "patient_id" in errors else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.output_dir / "oof_error_table.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
