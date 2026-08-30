"""Benchmark packaged fracture MIL inference and verify OOF numerical parity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from submission.fracture_mil import FractureMILPredictor


def _select_studies(catalog: pd.DataFrame) -> list[str]:
    studies = (
        catalog.groupby("study_id", sort=True)
        .agg(n_slices=("slice_index", "size"), truth=("study_fracture", "first"))
        .reset_index()
    )
    longest = str(studies.sort_values("n_slices", ascending=False).iloc[0]["study_id"])
    positive = str(
        studies.loc[studies["truth"].eq(1)]
        .sort_values("n_slices", ascending=False)
        .iloc[0]["study_id"]
    )
    negatives = studies.loc[studies["truth"].eq(0)].sort_values("n_slices")
    median_negative = str(negatives.iloc[len(negatives) // 2]["study_id"])
    return list(dict.fromkeys([longest, positive, median_negative]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = pd.read_csv(
        args.catalog, dtype={"study_id": str, "patient_id": str}
    )
    oof = pd.read_csv(args.oof_predictions, dtype={"study_id": str})
    selected = _select_studies(catalog)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictor = FractureMILPredictor(args.package, device=args.device)
    load_seconds = time.perf_counter() - started
    load_peak = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )

    rows: list[dict[str, object]] = []
    for study_id in selected:
        study = catalog.loc[catalog["study_id"].eq(study_id)].sort_values(
            "slice_index", kind="stable"
        )
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in study["image_path"]]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Failed to read cached image for study {study_id}")
        assigned_fold = int(study["outer_fold"].iloc[0])
        expected = pd.read_csv(
            args.model_root / f"fold_{assigned_fold}_v2" / "study_predictions.csv",
            dtype={"study_id": str},
        ).set_index("study_id").loc[study_id]
        expected_blend = oof.set_index("study_id").loc[study_id]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actual = predictor.predict_images(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual_fold = actual["folds"][assigned_fold]
        adjacent_error = abs(
            float(actual_fold["adjacent_pair"]) - float(expected["prob_adjacent_pair"])
        )
        mil_error = abs(float(actual_fold["mil_score"]) - float(expected["mil_score"]))
        blend_error = abs(
            float(actual_fold["blend_score"])
            - float(expected_blend["deployable_blend_score"])
        )
        if adjacent_error > 5e-4 or mil_error > 5e-4 or blend_error > 5e-3:
            raise RuntimeError(
                f"Packaged parity failed for {study_id}: "
                f"adjacent={adjacent_error}, mil={mil_error}, blend={blend_error}"
            )
        rows.append(
            {
                "study_id": study_id,
                "outer_fold": assigned_fold,
                "truth": int(study["study_fracture"].iloc[0]),
                "n_slices": len(images),
                "seconds": elapsed,
                "peak_gpu_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0
                ),
                "fracture_prob": float(actual["fracture_prob"]),
                "ensemble_score": float(actual["ensemble_score"]),
                "assigned_fold_adjacent_error": adjacent_error,
                "assigned_fold_mil_error": mil_error,
                "assigned_fold_blend_error": blend_error,
            }
        )

    seconds = np.asarray([float(row["seconds"]) for row in rows])
    peak = max(int(row["peak_gpu_bytes"]) for row in rows)
    report = {
        "package": str(args.package),
        "device": args.device,
        "load_seconds": load_seconds,
        "load_peak_gpu_bytes": load_peak,
        "studies": rows,
        "runtime": {
            "mean_seconds": float(seconds.mean()),
            "worst_seconds": float(seconds.max()),
            "projected_68_studies_mean_seconds": float(seconds.mean() * 68),
            "projected_68_studies_worst_seconds": float(seconds.max() * 68),
            "peak_gpu_bytes": peak,
        },
        "parity": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
