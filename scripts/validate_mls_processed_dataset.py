"""Validate a portable processed MLS dataset without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.mls_heatmap.dataset import (
    MLSHeatmapDataset,
    resolve_mls_image_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2",
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-positive-rows", type=int)
    parser.add_argument("--expected-negative-rows", type=int)
    parser.add_argument("--expected-studies", type=int)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    csv_path = root / "mls_labels_multitask.csv"
    image_dir = root / "images"
    frame = pd.read_csv(csv_path)
    frame["is_target"] = pd.to_numeric(frame["is_target"], errors="raise").astype(int)
    if not frame["is_target"].isin([0, 1]).all():
        raise ValueError("is_target must contain only 0 and 1")

    counts = {
        "rows": int(len(frame)),
        "positive_rows": int(frame["is_target"].eq(1).sum()),
        "negative_rows": int(frame["is_target"].eq(0).sum()),
        "studies": int(frame["patient_id"].astype(str).nunique()),
    }
    expected = {
        "rows": args.expected_rows,
        "positive_rows": args.expected_positive_rows,
        "negative_rows": args.expected_negative_rows,
        "studies": args.expected_studies,
    }
    mismatches = {
        name: {"observed": counts[name], "expected": value}
        for name, value in expected.items()
        if value is not None and counts[name] != value
    }
    if mismatches:
        raise ValueError(f"MLS dataset count mismatch: {mismatches}")

    resolved_paths = [
        resolve_mls_image_path(
            row.get("image_path", ""),
            row["image_name"],
            image_dir,
            project_root=PROJECT_ROOT,
        )
        for row in frame.to_dict(orient="records")
    ]
    if len({path.resolve() for path in resolved_paths}) != len(resolved_paths):
        raise ValueError("MLS dataset resolves multiple rows to the same image")

    dataset = MLSHeatmapDataset(
        csv_path=str(csv_path),
        img_dir=str(image_dir),
        augment=False,
        include_negatives=True,
        return_selector=True,
    )
    spacing = pd.to_numeric(dataset.data["spacing_x"], errors="coerce")
    truth = pd.to_numeric(dataset.data["study_mls_mm"], errors="coerce")
    result = {
        "schema_version": 1,
        "dataset_root": str(root),
        **counts,
        "resolved_image_paths": len(resolved_paths),
        "spacing_complete": bool(spacing.notna().all() and spacing.gt(0).all()),
        "study_truth_complete": bool(truth.notna().all() and truth.ge(0).all()),
        "model_loaded": False,
        "model_compute": "none",
    }
    if not result["spacing_complete"] or not result["study_truth_complete"]:
        raise ValueError(f"MLS metadata validation failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
