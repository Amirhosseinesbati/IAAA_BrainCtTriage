"""Select fracture study pooling with leave-one-fold-out transfer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


POOLING_COLUMNS = (
    "prob_max",
    "prob_top2_mean",
    "prob_top3_mean",
    "prob_top5_mean",
    "prob_adjacent_pair",
    "prob_window3_mean",
    "prob_noisy_or",
)


def _parse_fold_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected FOLD=PATH")
    fold, path = value.split("=", 1)
    return int(fold), Path(path)


def _load(items: list[tuple[int, Path]]) -> pd.DataFrame:
    if len(items) < 3:
        raise ValueError("At least three folds are required")
    frames: list[pd.DataFrame] = []
    seen_folds: set[int] = set()
    seen_studies: set[str] = set()
    required = {"study_id", "truth", *POOLING_COLUMNS}
    for fold, path in items:
        if fold in seen_folds:
            raise ValueError(f"Duplicate fold: {fold}")
        frame = pd.read_csv(path, dtype={"study_id": str})
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        overlap = seen_studies.intersection(frame["study_id"])
        if overlap:
            raise ValueError(f"Studies occur in multiple folds: {sorted(overlap)[:3]}")
        frame = frame.copy()
        frame["fold"] = fold
        frames.append(frame)
        seen_folds.add(fold)
        seen_studies.update(frame["study_id"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["fold", "study_id"]
    ).reset_index(drop=True)


def _auc_table(frame: pd.DataFrame, folds: list[int]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for column in POOLING_COLUMNS:
        per_fold = {
            str(fold): float(roc_auc_score(
                frame.loc[frame["fold"] == fold, "truth"],
                frame.loc[frame["fold"] == fold, column],
            ))
            for fold in folds
        }
        values = list(per_fold.values())
        result[column] = {
            "macro_auc": float(np.mean(values)),
            "worst_fold_auc": float(np.min(values)),
            **{f"fold_{fold}_auc": value for fold, value in per_fold.items()},
        }
    return result


def _best_profile(table: dict[str, dict[str, float]]) -> str:
    # Robustness is primary because a single shifted acquisition phenotype can
    # dominate hidden-test errors; macro AUC breaks ties.
    return max(
        POOLING_COLUMNS,
        key=lambda column: (
            table[column]["worst_fold_auc"],
            table[column]["macro_auc"],
            -POOLING_COLUMNS.index(column),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-prediction", action="append", type=_parse_fold_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = _load(args.fold_prediction)
    folds = sorted(int(value) for value in frame["fold"].unique())
    full_table = _auc_table(frame, folds)
    final_profile = _best_profile(full_table)

    transfer_prediction = np.full(len(frame), np.nan, dtype=np.float64)
    transfers: dict[str, object] = {}
    for held_out in folds:
        selection_folds = [fold for fold in folds if fold != held_out]
        selection_table = _auc_table(frame, selection_folds)
        selected = _best_profile(selection_table)
        mask = frame["fold"].to_numpy() == held_out
        transfer_prediction[mask] = frame.loc[mask, selected]
        transfers[str(held_out)] = {
            "selected_profile": selected,
            "selected_using_folds": selection_folds,
            "held_out_auc": float(roc_auc_score(
                frame.loc[mask, "truth"], transfer_prediction[mask]
            )),
            "selection_macro_auc": selection_table[selected]["macro_auc"],
            "selection_worst_fold_auc": selection_table[selected]["worst_fold_auc"],
        }
    if not np.isfinite(transfer_prediction).all():
        raise RuntimeError("Leave-one-fold-out transfer predictions are incomplete")
    frame["prob_lofo_selected_pooling"] = transfer_prediction
    held_out_aucs = [
        float(transfers[str(fold)]["held_out_auc"]) for fold in folds
    ]
    payload = {
        "evaluation_contract": "fixed detector epoch; pooling selected on other folds only",
        "n_studies": int(len(frame)),
        "n_positive": int(frame["truth"].sum()),
        "pooling_metrics": full_table,
        "final_robust_profile": final_profile,
        "leave_one_fold_out_transfer": transfers,
        "lofo_selected_macro_auc": float(np.mean(held_out_aucs)),
        "lofo_selected_worst_fold_auc": float(np.min(held_out_aucs)),
        "lofo_selected_pooled_auc": float(roc_auc_score(
            frame["truth"], transfer_prediction
        )),
        "warning": (
            "The final profile uses all available OOF folds and is diagnostic until "
            "validated unchanged on the real leaderboard. Macro/worst-fold transfer "
            "metrics are primary; pooled AUC can be distorted by fold/model score-scale "
            "differences."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "lofo_pooling_predictions.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
