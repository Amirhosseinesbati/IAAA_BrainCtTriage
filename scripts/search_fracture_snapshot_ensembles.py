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
from scripts.compare_fracture_study_predictions import _sampled_auc


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


def _macro_paired_bootstrap(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    fold_differences: list[np.ndarray] = []
    observed_by_fold: dict[str, float] = {}
    for fold, (reference, candidate) in sorted(frames.items()):
        if not reference[["study_id", "truth"]].equals(candidate[["study_id", "truth"]]):
            raise ValueError(f"Paired bootstrap identity mismatch in fold {fold}")
        truth = reference["truth"].to_numpy(dtype=np.int64)
        reference_score = reference["probability"].to_numpy(dtype=np.float64)
        candidate_score = candidate["probability"].to_numpy(dtype=np.float64)
        positive = np.flatnonzero(truth == 1)
        negative = np.flatnonzero(truth == 0)
        reference_auc = np.empty(iterations, dtype=np.float64)
        candidate_auc = np.empty(iterations, dtype=np.float64)
        for start in range(0, iterations, 2_000):
            stop = min(start + 2_000, iterations)
            size = stop - start
            sampled_positive = rng.choice(positive, (size, positive.size), replace=True)
            sampled_negative = rng.choice(negative, (size, negative.size), replace=True)
            reference_auc[start:stop] = _sampled_auc(
                reference_score, sampled_positive, sampled_negative
            )
            candidate_auc[start:stop] = _sampled_auc(
                candidate_score, sampled_positive, sampled_negative
            )
        difference = candidate_auc - reference_auc
        fold_differences.append(difference)
        observed_by_fold[str(fold)] = float(
            roc_auc_score(truth, candidate_score) - roc_auc_score(truth, reference_score)
        )
    macro_difference = np.mean(fold_differences, axis=0)
    return {
        "iterations": iterations,
        "seed": seed,
        "observed_difference_by_fold": observed_by_fold,
        "observed_macro_difference": float(np.mean(list(observed_by_fold.values()))),
        "macro_difference_bootstrap_95": [
            float(value) for value in np.quantile(macro_difference, [0.025, 0.5, 0.975])
        ],
        "probability_candidate_not_better": float(np.mean(macro_difference <= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-screening", action="append", type=_parse_fold_dir, required=True)
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--reference-epoch", type=int, default=10)
    parser.add_argument("--reference-pooling", choices=POOLING_METHODS, default="adjacent_pair")
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
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
    final_combo = tuple(int(value) for value in str(final["epochs"]).split(","))
    final_pooling = str(final["pooling"])
    reference_combo = (args.reference_epoch,)
    if reference_combo not in combinations:
        raise ValueError("Reference epoch is not in the searched snapshot combinations")
    final_rows: list[pd.DataFrame] = []
    paired_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in folds:
        reference_frame = prediction_cache[(reference_combo, fold)][
            ["study_id", "truth", f"prob_{args.reference_pooling}"]
        ].rename(columns={f"prob_{args.reference_pooling}": "probability"})
        candidate_frame = prediction_cache[(final_combo, fold)][
            ["study_id", "truth", f"prob_{final_pooling}"]
        ].rename(columns={f"prob_{final_pooling}": "probability"})
        paired_frames[fold] = (reference_frame, candidate_frame)
        output_frame = candidate_frame.rename(
            columns={"probability": "prob_final_snapshot_ensemble"}
        ).copy()
        output_frame["fold"] = fold
        final_rows.append(output_frame)
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
        "paired_vs_reference": {
            "reference": f"epoch{args.reference_epoch}__{args.reference_pooling}",
            **_macro_paired_bootstrap(
                paired_frames,
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            ),
        },
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
    pd.concat(final_rows, ignore_index=True).to_csv(
        args.output / "final_candidate_predictions.csv", index=False
    )
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
