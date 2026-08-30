"""Fit and honestly evaluate a regularized study-level fracture aggregator.

Inputs must be out-of-fold predictions from one fixed detector-training epoch.
Hyperparameter selection is nested by the immutable patient-grouped fold, so no
study is scored by an aggregator that used its label during fitting or tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_fracture_study_predictions import _sampled_auc


FEATURES = (
    "max",
    "top2_mean",
    "top3_mean",
    "top5_mean",
    "adjacent_pair",
    "window3_mean",
    "prob_noisy_or",
    "count_ge_025",
    "count_ge_050",
    "fraction_ge_025",
    "n_slices",
)
BASELINES = tuple(f"prob_{name}" for name in (
    "max", "top2_mean", "top3_mean", "top5_mean", "adjacent_pair",
    "window3_mean", "noisy_or",
))


def _parse_fold_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected FOLD=PATH")
    fold, path = value.split("=", 1)
    return int(fold), Path(path)


def load_predictions(items: list[tuple[int, Path]]) -> pd.DataFrame:
    if len(items) < 3:
        raise ValueError("At least three folds are required for nested evaluation")
    frames: list[pd.DataFrame] = []
    seen_folds: set[int] = set()
    seen_studies: set[str] = set()
    required = {"study_id", "truth", *FEATURES, *BASELINES}
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
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[list(FEATURES)].to_numpy(float)).all():
        raise ValueError("Non-finite aggregator features")
    return result.sort_values(["fold", "study_id"]).reset_index(drop=True)


def _model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
            solver="liblinear",
        ),
    )


def _select_c(frame: pd.DataFrame, c_values: tuple[float, ...]) -> tuple[float, dict[str, float]]:
    groups = sorted(frame["fold"].unique())
    scores: dict[str, float] = {}
    for c_value in c_values:
        prediction = np.full(len(frame), np.nan, dtype=np.float64)
        for held_out in groups:
            train = frame["fold"].to_numpy() != held_out
            validation = ~train
            fitted = _model(c_value).fit(
                frame.loc[train, list(FEATURES)], frame.loc[train, "truth"].astype(int)
            )
            prediction[validation] = fitted.predict_proba(
                frame.loc[validation, list(FEATURES)]
            )[:, 1]
        scores[str(c_value)] = float(roc_auc_score(frame["truth"], prediction))
    # Prefer stronger regularization when inner scores tie.
    selected = max(c_values, key=lambda value: (scores[str(value)], -value))
    return selected, scores


def nested_group_oof(
    frame: pd.DataFrame, c_values: tuple[float, ...]
) -> tuple[np.ndarray, dict[str, object]]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    selections: dict[str, object] = {}
    for held_out in sorted(frame["fold"].unique()):
        train = frame.loc[frame["fold"] != held_out].copy()
        validation = frame["fold"].to_numpy() == held_out
        selected_c, inner_scores = _select_c(train, c_values)
        fitted = _model(selected_c).fit(
            train.loc[:, list(FEATURES)], train["truth"].astype(int)
        )
        prediction[validation] = fitted.predict_proba(
            frame.loc[validation, list(FEATURES)]
        )[:, 1]
        selections[str(held_out)] = {
            "selected_c": selected_c,
            "inner_group_oof_auc_by_c": inner_scores,
        }
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nested OOF predictions are incomplete")
    return prediction, selections


def _paired_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    rng = np.random.default_rng(seed)
    reference_auc = np.empty(iterations)
    candidate_auc = np.empty(iterations)
    for start in range(0, iterations, 2_000):
        stop = min(start + 2_000, iterations)
        size = stop - start
        sampled_positive = rng.choice(positive, (size, positive.size), replace=True)
        sampled_negative = rng.choice(negative, (size, negative.size), replace=True)
        reference_auc[start:stop] = _sampled_auc(reference, sampled_positive, sampled_negative)
        candidate_auc[start:stop] = _sampled_auc(candidate, sampled_positive, sampled_negative)
    difference = candidate_auc - reference_auc
    return {
        "observed_difference": float(
            roc_auc_score(truth, candidate) - roc_auc_score(truth, reference)
        ),
        "difference_bootstrap_95": [
            float(value) for value in np.quantile(difference, [0.025, 0.5, 0.975])
        ],
        "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
        "iterations": iterations,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-prediction", action="append", type=_parse_fold_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.01, 0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_predictions(args.fold_prediction)
    c_values = tuple(sorted(set(args.c_values)))
    nested_prediction, selections = nested_group_oof(frame, c_values)
    truth = frame["truth"].to_numpy(dtype=np.int64)
    frame["prob_nested_logistic"] = nested_prediction
    baseline_auc = {
        column: float(roc_auc_score(truth, frame[column])) for column in BASELINES
    }
    best_baseline = max(baseline_auc, key=baseline_auc.get)
    nested_auc = float(roc_auc_score(truth, nested_prediction))
    per_fold_auc = {
        str(fold): float(roc_auc_score(group["truth"], group["prob_nested_logistic"]))
        for fold, group in frame.groupby("fold", sort=True)
    }
    final_c, final_cv_scores = _select_c(frame, c_values)
    final_model = _model(final_c).fit(frame.loc[:, list(FEATURES)], truth)
    payload = {
        "evaluation_contract": "fixed-detector-epoch nested patient-grouped fold OOF",
        "n_studies": int(len(frame)),
        "n_positive": int(truth.sum()),
        "features": list(FEATURES),
        "nested_oof_auc": nested_auc,
        "nested_oof_auc_by_fold": per_fold_auc,
        "baseline_auc": baseline_auc,
        "best_baseline": best_baseline,
        "outer_fold_selections": selections,
        "final_model_c": final_c,
        "final_group_oof_auc_by_c": final_cv_scores,
        "paired_vs_best_baseline": _paired_bootstrap(
            truth,
            frame[best_baseline].to_numpy(float),
            nested_prediction,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "nested_oof_predictions.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "selected_c": final_c,
            "evaluation_contract": payload["evaluation_contract"],
        },
        args.output / "study_aggregator.joblib",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
