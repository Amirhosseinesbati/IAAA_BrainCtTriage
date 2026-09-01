"""Strict cross-fold selection from precomputed MLS pooling grids.

This script performs no model/image inference and does not recompute pooling
predictions. It joins the identical profile rows already evaluated for each
fold, then selects both an aligned snapshot epoch and a pooling profile using
only the non-held-out folds.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EPOCHS = (13, 15, 17, 19, 21, 23)
PROFILE_COLUMNS = [
    "family",
    "size",
    "component_ratio",
    "selector_gate",
    "min_active_slices",
    "quantile",
    "probability_weighted",
    "heatmap_guard_ratio",
]
METRIC_COLUMNS = [
    "mae_mm",
    "rmse_mm",
    "bias_mm",
    "f1_3mm",
    "f1_5mm",
    "boundary_f1",
    "selection_objective",
]


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_dict(row: pd.Series) -> dict:
    return {key: _python_scalar(value) for key, value in row.items()}


def _epoch_from_candidate(value: str) -> int | None:
    match = re.search(r"epoch0*(\d+)(?:\D|$)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _load_fold_grid(path: Path, fold: str, epochs: tuple[int, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing precomputed pooling grid: {path}")

    required = ["candidate", *PROFILE_COLUMNS, *METRIC_COLUMNS]
    frame = pd.read_csv(path, usecols=required)
    frame["snapshot_epoch"] = frame["candidate"].map(_epoch_from_candidate)
    frame = frame.loc[frame["snapshot_epoch"].isin(epochs)].copy()
    frame["snapshot_epoch"] = frame["snapshot_epoch"].astype(int)

    weighted = frame["probability_weighted"].astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not weighted.isin(mapping).all():
        bad = sorted(weighted.loc[~weighted.isin(mapping)].unique())
        raise ValueError(f"Unexpected probability_weighted values in {path}: {bad}")
    frame["probability_weighted"] = weighted.map(mapping)

    key_columns = ["snapshot_epoch", *PROFILE_COLUMNS]
    duplicate_count = int(frame.duplicated(key_columns).sum())
    if duplicate_count:
        raise ValueError(f"{path} has {duplicate_count} duplicate epoch/profile rows")

    counts = frame.groupby("snapshot_epoch").size().to_dict()
    missing_epochs = [epoch for epoch in epochs if epoch not in counts]
    if missing_epochs:
        raise ValueError(f"{path} is missing requested epochs: {missing_epochs}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Inconsistent profile counts by epoch in {path}: {counts}")

    keep = [*key_columns, *METRIC_COLUMNS]
    renamed = {metric: f"{fold}_{metric}" for metric in METRIC_COLUMNS}
    return frame[keep].rename(columns=renamed)


def _selection_payload(
    row: pd.Series,
    held_out: str,
    selection_folds: list[str],
    criterion: str,
    selection_value: float,
) -> dict:
    return {
        "snapshot_epoch": int(row["snapshot_epoch"]),
        "profile": {key: _python_scalar(row[key]) for key in PROFILE_COLUMNS},
        "selected_using": selection_folds,
        "selection_criterion": criterion,
        "selection_value": float(selection_value),
        "held_out_mae_mm": float(row[f"{held_out}_mae_mm"]),
        "held_out_rmse_mm": float(row[f"{held_out}_rmse_mm"]),
        "held_out_boundary_f1": float(row[f"{held_out}_boundary_f1"]),
        "held_out_selection_objective": float(
            row[f"{held_out}_selection_objective"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0-grid", type=Path, required=True)
    parser.add_argument("--fold1-grid", type=Path, required=True)
    parser.add_argument("--fold2-grid", type=Path, required=True)
    parser.add_argument(
        "--epoch",
        action="append",
        type=int,
        dest="epochs",
        help="Aligned snapshot epoch; repeat as needed (default: 13/15/17/19/21/23).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    epochs = tuple(args.epochs or DEFAULT_EPOCHS)
    if len(epochs) != len(set(epochs)):
        raise ValueError("Snapshot epochs must be unique")

    grid_paths = {
        "fold0": args.fold0_grid,
        "fold1": args.fold1_grid,
        "fold2": args.fold2_grid,
    }
    fold_names = list(grid_paths)
    key_columns = ["snapshot_epoch", *PROFILE_COLUMNS]
    fold_frames = {
        fold: _load_fold_grid(path, fold, epochs)
        for fold, path in grid_paths.items()
    }

    results = fold_frames["fold0"]
    for fold in fold_names[1:]:
        before = len(results)
        results = results.merge(
            fold_frames[fold], on=key_columns, how="inner", validate="one_to_one"
        )
        if len(results) != before:
            raise ValueError(
                f"Profile-key mismatch while joining {fold}: {before} -> {len(results)}"
            )

    for metric in ("mae_mm", "rmse_mm", "boundary_f1", "selection_objective"):
        columns = [f"{fold}_{metric}" for fold in fold_names]
        results[f"mean_{metric}"] = results[columns].mean(axis=1)
    results["worst_mae_mm"] = results[
        [f"{fold}_mae_mm" for fold in fold_names]
    ].max(axis=1)

    robust = results.sort_values(
        ["mean_mae_mm", "worst_mae_mm", "mean_rmse_mm"],
        ascending=[True, True, True],
    )
    boundary = results.sort_values(
        ["mean_boundary_f1", "mean_mae_mm", "worst_mae_mm"],
        ascending=[False, True, True],
    )
    balanced = results.sort_values(
        ["mean_selection_objective", "mean_mae_mm", "worst_mae_mm"],
        ascending=[True, True, True],
    )

    def compute_loo(frame: pd.DataFrame) -> dict[str, dict[str, dict]]:
        leave_one_fold_out: dict[str, dict[str, dict]] = {}
        for held_out in fold_names:
            selection_folds = [fold for fold in fold_names if fold != held_out]
            train_mae = frame[
                [f"{fold}_mae_mm" for fold in selection_folds]
            ].mean(axis=1)
            train_worst_mae = frame[
                [f"{fold}_mae_mm" for fold in selection_folds]
            ].max(axis=1)
            train_rmse = frame[
                [f"{fold}_rmse_mm" for fold in selection_folds]
            ].mean(axis=1)
            train_boundary = frame[
                [f"{fold}_boundary_f1" for fold in selection_folds]
            ].mean(axis=1)
            train_balanced = frame[
                [f"{fold}_selection_objective" for fold in selection_folds]
            ].mean(axis=1)

            ranked = frame.assign(
                _train_mae=train_mae,
                _train_worst_mae=train_worst_mae,
                _train_rmse=train_rmse,
                _train_boundary=train_boundary,
                _train_balanced=train_balanced,
            )
            mae_index = ranked.sort_values(
                ["_train_mae", "_train_worst_mae", "_train_rmse"]
            ).index[0]
            boundary_index = ranked.sort_values(
                ["_train_boundary", "_train_mae", "_train_worst_mae"],
                ascending=[False, True, True],
            ).index[0]
            balanced_index = ranked.sort_values(
                ["_train_balanced", "_train_mae", "_train_worst_mae"]
            ).index[0]

            leave_one_fold_out[held_out] = {
                "mae_selection": _selection_payload(
                    frame.loc[mae_index], held_out, selection_folds,
                    "mean_mae_mm", train_mae.loc[mae_index]
                ),
                "boundary_selection": _selection_payload(
                    frame.loc[boundary_index], held_out, selection_folds,
                    "mean_boundary_f1", train_boundary.loc[boundary_index]
                ),
                "balanced_selection": _selection_payload(
                    frame.loc[balanced_index], held_out, selection_folds,
                    "mean_selection_objective", train_balanced.loc[balanced_index]
                ),
            }
        return leave_one_fold_out

    def aggregate_nested(
        leave_one_fold_out: dict[str, dict[str, dict]], kind: str
    ) -> dict:
        entries = [leave_one_fold_out[fold][kind] for fold in fold_names]
        return {
            "mean_held_out_mae_mm": float(np.mean([
                entry["held_out_mae_mm"] for entry in entries
            ])),
            "worst_held_out_mae_mm": float(np.max([
                entry["held_out_mae_mm"] for entry in entries
            ])),
            "mean_held_out_boundary_f1": float(np.mean([
                entry["held_out_boundary_f1"] for entry in entries
            ])),
            "mean_held_out_selection_objective": float(np.mean([
                entry["held_out_selection_objective"] for entry in entries
            ])),
        }

    selection_kinds = ("mae_selection", "boundary_selection", "balanced_selection")
    leave_one_fold_out = compute_loo(results)
    per_epoch_nested = {}
    for epoch in epochs:
        epoch_loo = compute_loo(results.loc[results["snapshot_epoch"] == epoch])
        per_epoch_nested[str(epoch)] = {
            "leave_one_fold_out_pooling": epoch_loo,
            "nested_aggregate": {
                kind: aggregate_nested(epoch_loo, kind) for kind in selection_kinds
            },
        }

    counts_by_fold = {
        fold: {
            str(epoch): int((frame["snapshot_epoch"] == epoch).sum())
            for epoch in epochs
        }
        for fold, frame in fold_frames.items()
    }
    payload = {
        "warning": (
            "No model/image inference or pooling recomputation was run. "
            "All-fold rankings are diagnostic; nested leave-one-fold-out "
            "selection is the strict robustness estimate."
        ),
        "source_grids": {fold: str(path) for fold, path in grid_paths.items()},
        "fold_names": fold_names,
        "snapshot_epochs": list(epochs),
        "profile_counts_by_fold_and_epoch": counts_by_fold,
        "n_joint_candidates": int(len(results)),
        "leave_one_fold_out_snapshot_and_pooling": leave_one_fold_out,
        "nested_aggregate": {
            kind: aggregate_nested(leave_one_fold_out, kind)
            for kind in selection_kinds
        },
        "per_epoch_nested_pooling": per_epoch_nested,
        "best_all_fold_mae": _row_dict(robust.iloc[0]),
        "best_all_fold_boundary": _row_dict(boundary.iloc[0]),
        "best_all_fold_balanced": _row_dict(balanced.iloc[0]),
        "top_all_fold_balanced": [
            _row_dict(row) for _, row in balanced.head(20).iterrows()
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "crossfold_snapshot_pooling_grid.csv", index=False)
    summary_path = args.output_dir / "crossfold_snapshot_pooling_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
