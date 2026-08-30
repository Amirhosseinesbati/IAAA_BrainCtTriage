"""Cross-fit the fracture decision threshold required by the official 0.5 rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def _select_f1_threshold(truth: np.ndarray, score: np.ndarray) -> float:
    """Select deterministically by F1, then precision, specificity, threshold."""
    truth = np.asarray(truth, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Threshold development data must contain both classes")
    if score.ndim != 1 or score.shape != truth.shape or not np.isfinite(score).all():
        raise ValueError("Scores must be one finite value per truth row")
    unique = np.unique(score)
    candidates = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    best_key: tuple[float, float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        metrics = _binary_metrics(truth, score >= threshold)
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["specificity"]),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _select_weight_and_threshold(
    truth: np.ndarray,
    reference_cdf: np.ndarray,
    candidate_cdf: np.ndarray,
    weights: list[float],
) -> tuple[float, float, dict[str, float | int]]:
    """Select on development data only, preferring the simpler lower MIL weight."""
    best_key: tuple[float, float, float, float] | None = None
    best: tuple[float, float, dict[str, float | int]] | None = None
    for weight in weights:
        score = (1.0 - weight) * reference_cdf + weight * candidate_cdf
        threshold = _select_f1_threshold(truth, score)
        metrics = _binary_metrics(truth, score >= threshold)
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["specificity"]),
            -float(weight),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (float(weight), threshold, metrics)
    if best is None:
        raise RuntimeError("No candidate weight was evaluated")
    return best


def _map_threshold_to_half(score: np.ndarray, threshold: float) -> np.ndarray:
    """Monotonic [0,1] mapping whose official 0.5 cutoff equals threshold."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("Deployment threshold must be strictly inside (0, 1)")
    score = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    return np.where(
        score < threshold,
        0.5 * score / threshold,
        0.5 + 0.5 * (score - threshold) / (1.0 - threshold),
    )


def _paired_bootstrap_f1(
    predictions: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    difference = np.empty(iterations, dtype=np.float64)
    groups = list(predictions.groupby("outer_fold", sort=True))
    for iteration in range(iterations):
        truth_parts: list[np.ndarray] = []
        reference_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        for _, fold in groups:
            truth = fold["truth"].to_numpy(dtype=np.int64)
            positive = np.flatnonzero(truth == 1)
            negative = np.flatnonzero(truth == 0)
            indices = np.concatenate(
                (
                    rng.choice(positive, size=positive.size, replace=True),
                    rng.choice(negative, size=negative.size, replace=True),
                )
            )
            truth_parts.append(truth[indices])
            reference_parts.append(fold[reference_column].to_numpy(dtype=np.int64)[indices])
            candidate_parts.append(fold[candidate_column].to_numpy(dtype=np.int64)[indices])
        sampled_truth = np.concatenate(truth_parts)
        reference_f1 = f1_score(
            sampled_truth, np.concatenate(reference_parts), zero_division=0
        )
        candidate_f1 = f1_score(
            sampled_truth, np.concatenate(candidate_parts), zero_division=0
        )
        difference[iteration] = candidate_f1 - reference_f1
    return difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--weights",
        default="0.45",
        help="Comma-separated MIL weights selected within each development split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = [float(value) for value in args.weights.split(",")]
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("--weights must be a non-empty list within [0, 1]")
    weights = sorted(set(weights))

    frame = pd.read_csv(
        args.predictions, dtype={"study_id": str, "patient_id": str}
    )
    required = {
        "study_id",
        "patient_id",
        "truth",
        "outer_fold",
        "prob_adjacent_pair",
        "reference_train_cdf",
        "candidate_train_cdf",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame["study_id"].duplicated().any():
        raise ValueError("A study appears more than once")
    if not np.isfinite(
        frame[
            ["prob_adjacent_pair", "reference_train_cdf", "candidate_train_cdf"]
        ].to_numpy()
    ).all():
        raise ValueError("Non-finite scores found")

    selections: list[dict[str, float | int]] = []
    rows: list[pd.DataFrame] = []
    for fold in sorted(frame["outer_fold"].unique()):
        development = frame["outer_fold"] != fold
        heldout = frame["outer_fold"] == fold
        truth_development = frame.loc[development, "truth"].to_numpy(dtype=np.int64)
        reference_threshold = _select_f1_threshold(
            truth_development,
            frame.loc[development, "prob_adjacent_pair"].to_numpy(dtype=np.float64),
        )
        selected_weight, candidate_threshold, development_metrics = (
            _select_weight_and_threshold(
                truth_development,
                frame.loc[development, "reference_train_cdf"].to_numpy(
                    dtype=np.float64
                ),
                frame.loc[development, "candidate_train_cdf"].to_numpy(
                    dtype=np.float64
                ),
                weights,
            )
        )
        selected = frame.loc[heldout].copy()
        selected["selected_candidate_weight"] = selected_weight
        selected["selected_blend_score"] = (
            (1.0 - selected_weight) * selected["reference_train_cdf"]
            + selected_weight * selected["candidate_train_cdf"]
        )
        selected["reference_threshold"] = reference_threshold
        selected["candidate_threshold"] = candidate_threshold
        selected["reference_binary"] = (
            selected["prob_adjacent_pair"] >= reference_threshold
        ).astype(np.int64)
        selected["candidate_binary"] = (
            selected["selected_blend_score"] >= candidate_threshold
        ).astype(np.int64)
        selected["fracture_prob"] = _map_threshold_to_half(
            selected["selected_blend_score"].to_numpy(), candidate_threshold
        )
        rows.append(selected)
        selections.append(
            {
                "held_out_fold": int(fold),
                "n_development": int(development.sum()),
                "reference_threshold": reference_threshold,
                "selected_candidate_weight": selected_weight,
                "candidate_threshold": candidate_threshold,
                "development_f1": float(development_metrics["f1"]),
            }
        )

    predictions = pd.concat(rows, ignore_index=True).sort_values("study_id")
    truth = predictions["truth"].to_numpy(dtype=np.int64)
    reference_metrics = _binary_metrics(truth, predictions["reference_binary"])
    candidate_metrics = _binary_metrics(truth, predictions["candidate_binary"])
    final_weight, final_threshold, final_apparent = _select_weight_and_threshold(
        truth,
        predictions["reference_train_cdf"].to_numpy(dtype=np.float64),
        predictions["candidate_train_cdf"].to_numpy(dtype=np.float64),
        weights,
    )
    difference = _paired_bootstrap_f1(
        predictions,
        "reference_binary",
        "candidate_binary",
        iterations=args.iterations,
        seed=args.seed,
    )
    payload = {
        "protocol": "leave_one_outer_fold_out_f1_threshold_selection",
        "candidate_weight_grid": weights,
        "selections": selections,
        "reference_crossfit": reference_metrics,
        "candidate_crossfit": candidate_metrics,
        "crossfit_f1_difference": float(candidate_metrics["f1"] - reference_metrics["f1"]),
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": np.quantile(difference, [0.025, 0.5, 0.975]).tolist(),
            "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
        },
        "deployment": {
            "score": "mean_of_fold_empirical_cdf_blends",
            "candidate_weight": final_weight,
            "selected_threshold_all_oof": final_threshold,
            "threshold_mapping": "piecewise_linear_threshold_to_probability_0.5",
            "apparent_all_oof_metrics": final_apparent,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "crossfit_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    (args.output / "deployment_calibration.json").write_text(
        json.dumps(payload["deployment"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(rendered)


if __name__ == "__main__":
    main()
