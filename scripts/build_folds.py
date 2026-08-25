"""Create the immutable patient-grouped fold manifest used by all tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, TRAINING_CSV_PATH
from src.evaluation.folds import aggregate_studies, create_fold_manifest, fold_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=TRAINING_CSV_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "config" / "folds.csv")
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()
    manifest = create_fold_manifest(aggregate_studies(args.csv), n_folds=args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Saved {len(manifest)} studies / {manifest.patient_id.nunique()} patients to {args.output}")
    print(fold_summary(manifest).to_string())


if __name__ == "__main__":
    main()
