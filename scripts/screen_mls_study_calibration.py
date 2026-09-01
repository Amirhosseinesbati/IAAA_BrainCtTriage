"""Nested screening of a one-parameter MLS study-level calibration.

The script uses saved CUDA predictions only. For each predeclared prediction
source, an additive offset is applied only to studies already predicted as
active, preserving the explicit negative sentinel. The offset is selected on
two folds and evaluated unchanged on the third.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen_mls_snapshot_blends import _blend_items, _load
from search_mls_crossfold_pooling import (
    NEGATIVE_MLS_MM,
    PoolingProfile,
    _decode,
    _metrics,
    _predict,
)


EPOCHS = (13, 15, 17)
SOURCES = {
    "single_e15": {"epochs": (15,), "mode": "mean"},
    "mean_e13_e15_e17": {"epochs": (13, 15, 17), "mode": "mean"},
    "median_e13_e15_e17": {"epochs": (13, 15, 17), "mode": "median"},
}
PROFILE = PoolingProfile("severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.0)
OFFSETS_MM = np.round(np.arange(-0.50, 0.801, 0.05), 2)


def _path(root: Path, epoch: int) -> Path:
    path = root / f"epoch{epoch:03d}" / "study_slice_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _calibrate(prediction: np.ndarray, offset: float) -> np.ndarray:
    calibrated = prediction.copy()
    active = calibrated > NEGATIVE_MLS_MM + 1e-6
    calibrated[active] = np.clip(calibrated[active] + offset, NEGATIVE_MLS_MM, 30.0)
    return calibrated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0-root", type=Path, required=True)
    parser.add_argument("--fold1-root", type=Path, required=True)
    parser.add_argument("--fold2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roots = {"fold0": args.fold0_root, "fold1": args.fold1_root, "fold2": args.fold2_root}
    frames = {
        fold: {epoch: _load(_path(root, epoch)) for epoch in EPOCHS}
        for fold, root in roots.items()
    }
    decoded = {
        fold: {epoch: _decode(frame) for epoch, frame in fold_frames.items()}
        for fold, fold_frames in frames.items()
    }
    truths = {
        fold: fold_frames[15]["gt_MLS_mm"].to_numpy(float)
        for fold, fold_frames in frames.items()
    }
    for fold, fold_frames in frames.items():
        reference = fold_frames[15]
        for epoch, frame in fold_frames.items():
            if frame["study_id"].tolist() != reference["study_id"].tolist():
                raise ValueError(f"Study IDs differ in {fold}/epoch{epoch}")
            if not np.allclose(frame["gt_MLS_mm"], reference["gt_MLS_mm"]):
                raise ValueError(f"Ground truth differs in {fold}/epoch{epoch}")

    source_predictions: dict[str, dict[str, np.ndarray]] = {}
    for source, spec in SOURCES.items():
        epochs = spec["epochs"]
        weights = np.ones(len(epochs), dtype=float)
        source_predictions[source] = {}
        for fold in roots:
            study_items = []
            for index in range(len(decoded[fold][15])):
                study_items.append(_blend_items(
                    [decoded[fold][epoch][index] for epoch in epochs],
                    weights,
                    str(spec["mode"]),
                ))
            source_predictions[source][fold] = np.asarray([
                _predict(items, PROFILE) for items in study_items
            ])

    rows = []
    for source, fold_predictions in source_predictions.items():
        for offset in OFFSETS_MM:
            row = {"source": source, "active_study_offset_mm": float(offset)}
            for fold in roots:
                prediction = _calibrate(fold_predictions[fold], float(offset))
                metrics = _metrics(truths[fold], prediction)
                boundary = float(np.mean([metrics["f1_3mm"], metrics["f1_5mm"]]))
                metrics["boundary_f1"] = boundary
                metrics["selection_objective"] = metrics["mae_mm"] + 2 * (1 - boundary)
                for key, value in metrics.items():
                    row[f"{fold}_{key}"] = value
            row["mean_mae_mm"] = float(np.mean([
                row[f"{fold}_mae_mm"] for fold in roots
            ]))
            row["worst_mae_mm"] = float(np.max([
                row[f"{fold}_mae_mm"] for fold in roots
            ]))
            row["mean_boundary_f1"] = float(np.mean([
                row[f"{fold}_boundary_f1"] for fold in roots
            ]))
            row["mean_selection_objective"] = float(np.mean([
                row[f"{fold}_selection_objective"] for fold in roots
            ]))
            rows.append(row)

    results = pd.DataFrame(rows)
    summaries = {}
    for source in SOURCES:
        source_rows = results.loc[results["source"] == source]
        loo = {}
        for held_out in roots:
            selection_folds = [fold for fold in roots if fold != held_out]
            train_mae = source_rows[
                [f"{fold}_mae_mm" for fold in selection_folds]
            ].mean(axis=1)
            train_boundary = source_rows[
                [f"{fold}_boundary_f1" for fold in selection_folds]
            ].mean(axis=1)
            train_objective = source_rows[
                [f"{fold}_selection_objective" for fold in selection_folds]
            ].mean(axis=1)
            ranked = source_rows.assign(
                _train_objective=train_objective,
                _train_mae=train_mae,
                _train_boundary=train_boundary,
                _abs_offset=source_rows["active_study_offset_mm"].abs(),
            ).sort_values(["_train_objective", "_train_mae", "_abs_offset"])
            index = ranked.index[0]
            selected = source_rows.loc[index]
            loo[held_out] = {
                "selected_using": selection_folds,
                "offset_mm": float(selected["active_study_offset_mm"]),
                "selection_mean_objective": float(train_objective.loc[index]),
                "held_out_mae_mm": float(selected[f"{held_out}_mae_mm"]),
                "held_out_boundary_f1": float(selected[f"{held_out}_boundary_f1"]),
                "held_out_objective": float(selected[f"{held_out}_selection_objective"]),
            }
        entries = list(loo.values())
        no_offset = source_rows.loc[source_rows["active_study_offset_mm"] == 0].iloc[0]
        diagnostic = source_rows.sort_values(
            ["mean_selection_objective", "mean_mae_mm", "active_study_offset_mm"]
        ).iloc[0]
        summaries[source] = {
            "uncalibrated": no_offset.to_dict(),
            "nested_loo": loo,
            "nested_aggregate": {
                "mean_held_out_mae_mm": float(np.mean([
                    entry["held_out_mae_mm"] for entry in entries
                ])),
                "worst_held_out_mae_mm": float(np.max([
                    entry["held_out_mae_mm"] for entry in entries
                ])),
                "mean_held_out_boundary_f1": float(np.mean([
                    entry["held_out_boundary_f1"] for entry in entries
                ])),
                "mean_held_out_objective": float(np.mean([
                    entry["held_out_objective"] for entry in entries
                ])),
            },
            "best_all_fold_diagnostic": diagnostic.to_dict(),
        }

    payload = {
        "warning": (
            "Offsets are selected with nested LOO. Snapshot blend sources were "
            "defined after prior analysis and remain screening candidates."
        ),
        "profile": PROFILE.__dict__,
        "offset_grid_mm": OFFSETS_MM.tolist(),
        "sources": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "study_calibration_grid.csv", index=False)
    (args.output_dir / "study_calibration_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        source: summary["nested_aggregate"] for source, summary in summaries.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
