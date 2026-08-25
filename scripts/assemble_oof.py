"""Assemble and validate ICH/fracture/MLS OOF predictions."""

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
from src.evaluation.oof import assemble_oof_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ich", type=Path, required=True)
    parser.add_argument("--fracture", type=Path, required=True)
    parser.add_argument("--mls", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "oof_predictions.csv")
    args = parser.parse_args()
    assembled = assemble_oof_predictions(
        pd.read_csv(args.ich), pd.read_csv(args.fracture), pd.read_csv(args.mls),
    )
    metrics = compute_competition_metrics(
        assembled["triage_class"], assembled["pred_triage"],
        patient_ids=assembled["patient_id"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assembled.to_csv(args.output, index=False)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Validated {len(assembled)} complete OOF studies -> {args.output}")


if __name__ == "__main__":
    main()
