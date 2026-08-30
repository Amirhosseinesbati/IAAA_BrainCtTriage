"""Compare positive validation-study difficulty across fracture folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _box_areas(dataset: Path, labels: pd.Series) -> list[float]:
    areas: list[float] = []
    for relative in labels:
        path = dataset / str(relative)
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                areas.append(float(parts[3]) * float(parts[4]))
    return areas


def analyze(dataset: Path) -> dict[str, object]:
    studies = pd.read_csv(dataset / "studies.csv")
    manifest = pd.read_csv(dataset / "manifest.csv")
    positive = studies[(studies["split"] == "val") & (studies["fracture"] == 1)]
    rows: list[dict[str, object]] = []
    all_areas: list[float] = []
    for study in positive.itertuples(index=False):
        slices = manifest[
            (manifest["split"] == "val")
            & (manifest["study_id"] == study.study_id)
            & (manifest["slice_fracture"] == 1)
        ]
        areas = _box_areas(dataset, slices["label"])
        all_areas.extend(areas)
        indices = slices["slice_index"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "study_id": int(study.study_id),
                "available_slices": int(study.available_slices),
                "positive_slices": int(study.positive_slices),
                "positive_span": int(indices.max() - indices.min() + 1),
                "n_boxes": len(areas),
                "median_box_area_fraction": float(np.median(areas)),
                "min_box_area_fraction": float(np.min(areas)),
                "max_box_area_fraction": float(np.max(areas)),
            }
        )
    return {
        "dataset": str(dataset),
        "n_positive_studies": len(rows),
        "positive_slices_median": float(np.median([row["positive_slices"] for row in rows])),
        "boxes_per_study_median": float(np.median([row["n_boxes"] for row in rows])),
        "box_area_fraction_median": float(np.median(all_areas)),
        "box_area_fraction_q25": float(np.quantile(all_areas, 0.25)),
        "studies": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {dataset.name: analyze(dataset) for dataset in args.datasets}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
