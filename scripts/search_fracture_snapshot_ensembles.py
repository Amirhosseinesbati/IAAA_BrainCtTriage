"""Leakage-aware cross-fold search for fracture snapshot ensembles.

Detector slice scores are averaged before study pooling, matching the intended
submission inference order. Snapshot/pooling selection is evaluated with
leave-one-fold-out transfer across the immutable patient-grouped folds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fracture.pooling import aggregate_study_scores


EPOCH_PATTERN = re.compile(r"epoch(\d+)$")
POOLING_METHODS = (
    "max",
    "top2_mean",
    "top3_mean",
    "top5_mean",
    "adjacent_pair",
    "window3_mean",
    "noisy_or",
)


def _parse_fold_dir(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected FOLD=SCREENING_DIR")
    fold, directory = value.split("=", 1)
    return int(fold), Path(directory)


def _available_epochs(directory: Path) -> set[int]:
    result: set[int] = set()
    for child in directory.iterdir():
        match = EPOCH_PATTERN.fullmatch(child.name)
        if match and (child / "slice_predictions.csv").is_file():
            result.add(int(match.group(1)))
    return result


def _load_fold(directory: Path, epochs: tuple[int, ...]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    reference: pd.DataFrame | None = None
    scores: dict[int, np.ndarray] = {}
    identity_columns = ["study_id", "slice_index", "study_fracture"]
    for epoch in epochs:
        path = directory / f"epoch{epoch}" / "slice_predictions.csv"
        frame = pd.read_csv(path, dtype={"study_id": str}).sort_values(
            ["study_id", "slice_index"]
        ).reset_index(drop=True)
        missing = set(identity_columns + ["slice_score"]).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if reference is None:
            reference = frame[identity_columns].copy()
        elif not reference.equals(frame[identity_columns]):
            raise ValueError(f"Slice identity/truth mismatch at {path}")
        values = frame["slice_score"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite slice scores at {path}")
        scores[epoch] = values
    if reference is None:
        raise ValueError(f"No epoch predictions loaded from {directory}")
    return reference, scores


def _study_predictions(reference: pd.DataFrame, slice_scores: np.ndarray) -> pd.DataFrame:
    frame = reference.copy()
    frame["slice_score"] = slice_scores
    rows: list[dict[str, float | int | str]] = []
    for study_id, group in frame.groupby("study_id", sort=True):
        ordered = group.sort_values("slice_index")
        pooled = aggregate_study_scores(ordered["slice_score"].to_numpy(float))
        rows.append({
            "study_id": str(study_id),
            "truth": int(ordered["study_fracture"].max()),
            **{f"prob_{name}": value for name, value in pooled.items()},
        })
    return pd.DataFrame.from_records(rows)


def _select_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Prioritize worst-fold AUC, macro AUC, then cheaper snapshot count."""
    return max(
        rows,
        key=lambda row: (
            float(row["worst_fold_auc"]),
            float(row["macro_auc"]),
            -int(row["n_snapshots"]),
            str(row["candidate"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-screening", action="append", type=_parse_fold_dir, required=True)
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fold_dirs = dict(args.fold_screening)
    if len(fold_dirs) < 3 or len(fold_dirs) != len(args.fold_screening):
        raise ValueError("At least three unique folds are required")
    common_epochs = set.intersection(*(
        _available_epochs(directory) for directory in fold_dirs.values()
    ))
    if args.epochs is not None:
        requested = set(args.epochs)
        missing = requested.difference(common_epochs)
        if missing:
            raise ValueError(f"Requested epochs unavailable in every fold: {sorted(missing)}")
        common_epochs = requested
    epochs = tuple(sorted(common_epochs))
    if not epochs:
        raise ValueError("No common periodic epochs")
    max_snapshots = min(args.max_snapshots, len(epochs))
    if max_snapshots < 1:
        raise ValueError("--max-snapshots must be positive")

    raw = {
        fold: _load_fold(directory, epochs)
        for fold, directory in sorted(fold_dirs.items())
    }
    combinations = [
        combo
        for size in range(1, max_snapshots + 1)
        for combo in itertools.combinations(epochs, size)
    ]
    prediction_cache: dict[tuple[tuple[int, ...], int], pd.DataFrame] = {}
    grid: list[dict[str, object]] = []
    folds = sorted(raw)
    for combo in combinations:
        for fold in folds:
            reference, epoch_scores = raw[fold]
            mean_score = np.mean([epoch_scores[epoch] for epoch in combo], axis=0)
            prediction_cache[(combo, fold)] = _study_predictions(reference, mean_score)
        for pooling in POOLING_METHODS:
            per_fold = {
                fold: float(roc_auc_score(
                    prediction_cache[(combo, fold)]["truth"],
                    prediction_cache[(combo, fold)][f"prob_{pooling}"],
                ))
                for fold in folds
            }
            values = list(per_fold.values())
            candidate = f"epochs_{'-'.join(map(str, combo))}__{pooling}"
            grid.append({
                "candidate": candidate,
                "epochs": ",".join(map(str, combo)),
                "pooling": pooling,
                "n_snapshots": len(combo),
                "macro_auc": float(np.mean(values)),
                "worst_fold_auc": float(np.min(values)),
                **{f"fold_{fold}_auc": value for fold, value in per_fold.items()},
            })

    final = _select_candidate(grid)
    transfer_rows: list[pd.DataFrame] = []
    transfers: dict[str, object] = {}
    for held_out in folds:
        selection_rows: list[dict[str, object]] = []
        selection_folds = [fold for fold in folds if fold != held_out]
        for row in grid:
            values = [float(row[f"fold_{fold}_auc"]) for fold in selection_folds]
            selected_row = dict(row)
            selected_row["macro_auc"] = float(np.mean(values))
            selected_row["worst_fold_auc"] = float(np.min(values))
            selection_rows.append(selected_row)
        selected = _select_candidate(selection_rows)
        combo = tuple(int(value) for value in str(selected["epochs"]).split(","))
        pooling = str(selected["pooling"])
        held_predictions = prediction_cache[(combo, held_out)][
            ["study_id", "truth", f"prob_{pooling}"]
        ].copy()
        held_predictions = held_predictions.rename(
            columns={f"prob_{pooling}": "prob_lofo_snapshot_ensemble"}
        )
        held_predictions["fold"] = held_out
        held_predictions["selected_candidate"] = selected["candidate"]
        transfer_rows.append(held_predictions)
        transfers[str(held_out)] = {
            "selected_candidate": selected["candidate"],
            "selected_using_folds": selection_folds,
            "selection_macro_auc": selected["macro_auc"],
            "selection_worst_fold_auc": selected["worst_fold_auc"],
            "held_out_auc": float(selected[f"fold_{held_out}_auc"]),
        }

    transfer_auc = [float(value["held_out_auc"]) for value in transfers.values()]
    payload = {
        "evaluation_contract": (
            "slice-score snapshot averaging; pooling/snapshots selected on other folds only"
        ),
        "common_epochs": list(epochs),
        "n_candidates": len(grid),
        "final_robust_candidate": final,
        "leave_one_fold_out_transfer": transfers,
        "lofo_macro_auc": float(np.mean(transfer_auc)),
        "lofo_worst_fold_auc": float(np.min(transfer_auc)),
        "warning": (
            "The all-fold final candidate is diagnostic until submitted unchanged; "
            "LOFO transfer is the primary internal selection evidence."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).sort_values(
        ["worst_fold_auc", "macro_auc", "n_snapshots"],
        ascending=[False, False, True],
    ).to_csv(args.output / "snapshot_ensemble_grid.csv", index=False)
    pd.concat(transfer_rows, ignore_index=True).to_csv(
        args.output / "lofo_predictions.csv", index=False
    )
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
