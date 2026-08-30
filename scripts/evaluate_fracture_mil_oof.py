"""Aggregate fracture MIL outer-fold predictions with paired macro bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.compare_fracture_study_predictions import _sampled_auc


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def _macro_paired_bootstrap(
    predictions: pd.DataFrame,
    reference_score: str,
    candidate_score: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    reference_macro = np.zeros(iterations, dtype=np.float64)
    candidate_macro = np.zeros(iterations, dtype=np.float64)
    grouped = list(predictions.groupby("outer_fold", sort=True))
    if not grouped:
        raise ValueError("No outer folds found")
    for _, fold in grouped:
        truth = fold["truth"].to_numpy(dtype=np.int64)
        reference = fold[reference_score].to_numpy(dtype=np.float64)
        candidate = fold[candidate_score].to_numpy(dtype=np.float64)
        positive = np.flatnonzero(truth == 1)
        negative = np.flatnonzero(truth == 0)
        if positive.size == 0 or negative.size == 0:
            raise ValueError("Every outer fold must contain both classes")
        fold_reference = np.empty(iterations, dtype=np.float64)
        fold_candidate = np.empty(iterations, dtype=np.float64)
        for start in range(0, iterations, 2_000):
            stop = min(start + 2_000, iterations)
            size = stop - start
            sampled_positive = rng.choice(
                positive, size=(size, positive.size), replace=True
            )
            sampled_negative = rng.choice(
                negative, size=(size, negative.size), replace=True
            )
            fold_reference[start:stop] = _sampled_auc(
                reference, sampled_positive, sampled_negative
            )
            fold_candidate[start:stop] = _sampled_auc(
                candidate, sampled_positive, sampled_negative
            )
        reference_macro += fold_reference
        candidate_macro += fold_candidate
    return reference_macro / len(grouped), candidate_macro / len(grouped)


def _fold_percentile_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("outer_fold")[column].rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold-pattern", default="fold_{fold}_v2")
    parser.add_argument("--reference-score", default="prob_adjacent_pair")
    parser.add_argument("--candidate-score", default="mil_score")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    fold_metrics: dict[int, dict[str, object]] = {}
    for fold in range(5):
        directory = args.root / args.fold_pattern.format(fold=fold)
        prediction_path = directory / "study_predictions.csv"
        metrics_path = directory / "metrics.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete outer fold directory: {directory}")
        frame = pd.read_csv(
            prediction_path, dtype={"study_id": str, "patient_id": str}
        )
        required = {
            "study_id",
            "patient_id",
            "truth",
            "outer_fold",
            args.reference_score,
            args.candidate_score,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
        if not frame["outer_fold"].eq(fold).all():
            raise ValueError(f"Prediction rows in {directory} do not belong to fold {fold}")
        frames.append(frame)
        fold_metrics[fold] = json.loads(metrics_path.read_text(encoding="utf-8"))

    predictions = pd.concat(frames, ignore_index=True)
    if predictions["study_id"].duplicated().any():
        raise ValueError("A study appears in multiple outer-fold prediction files")
    patient_fold_counts = predictions.groupby("patient_id")["outer_fold"].nunique()
    if not patient_fold_counts.eq(1).all():
        raise ValueError("A patient appears in multiple outer folds")
    if not np.isfinite(
        predictions[[args.reference_score, args.candidate_score]].to_numpy()
    ).all():
        raise ValueError("OOF scores contain non-finite values")

    per_fold: list[dict[str, object]] = []
    for fold, frame in predictions.groupby("outer_fold", sort=True):
        truth = frame["truth"].to_numpy(dtype=np.int64)
        reference_auc = float(roc_auc_score(truth, frame[args.reference_score]))
        candidate_auc = float(roc_auc_score(truth, frame[args.candidate_score]))
        per_fold.append(
            {
                "fold": int(fold),
                "n_studies": len(frame),
                "n_positive": int(truth.sum()),
                "reference_auc": reference_auc,
                "candidate_auc": candidate_auc,
                "difference": candidate_auc - reference_auc,
                "selected_alpha": fold_metrics[int(fold)]["selected_alpha"],
                "final_epochs": fold_metrics[int(fold)]["final_epochs"],
            }
        )
    reference_values = np.asarray(
        [float(row["reference_auc"]) for row in per_fold], dtype=np.float64
    )
    candidate_values = np.asarray(
        [float(row["candidate_auc"]) for row in per_fold], dtype=np.float64
    )
    reference_bootstrap, candidate_bootstrap = _macro_paired_bootstrap(
        predictions,
        args.reference_score,
        args.candidate_score,
        iterations=args.iterations,
        seed=args.seed,
    )
    difference = candidate_bootstrap - reference_bootstrap
    predictions["reference_fold_rank"] = _fold_percentile_rank(
        predictions, args.reference_score
    )
    predictions["candidate_fold_rank"] = _fold_percentile_rank(
        predictions, args.candidate_score
    )
    truth = predictions["truth"].to_numpy(dtype=np.int64)
    payload = {
        "reference_score": args.reference_score,
        "candidate_score": args.candidate_score,
        "n_studies": len(predictions),
        "n_patients": int(predictions["patient_id"].nunique()),
        "n_positive": int(truth.sum()),
        "per_fold": per_fold,
        "reference_macro_auc": float(reference_values.mean()),
        "candidate_macro_auc": float(candidate_values.mean()),
        "macro_difference": float(candidate_values.mean() - reference_values.mean()),
        "reference_worst_fold_auc": float(reference_values.min()),
        "candidate_worst_fold_auc": float(candidate_values.min()),
        "worst_fold_difference": float(candidate_values.min() - reference_values.min()),
        "reference_fold_rank_pooled_auc": float(
            roc_auc_score(truth, predictions["reference_fold_rank"])
        ),
        "candidate_fold_rank_pooled_auc": float(
            roc_auc_score(truth, predictions["candidate_fold_rank"])
        ),
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "reference_macro_auc_95": _interval(reference_bootstrap),
            "candidate_macro_auc_95": _interval(candidate_bootstrap),
            "difference_95": _interval(difference),
            "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(["outer_fold", "study_id"]).to_csv(
        args.output / "oof_predictions.csv", index=False
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
