"""Aggregate five independently calibrated outer folds for the 2.5D ICH gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | list[list[int]]]:
    truth = truth.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.count_nonzero(truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    return {
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(sorted_reference, values, side="right") / len(sorted_reference)


def _diagnostic_threshold(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Select on all OOF rows; useful for design, not an unbiased final estimate."""
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores]))
    rows = []
    for threshold in candidates:
        metrics = _binary_metrics(truth, scores >= threshold)
        if float(metrics["sensitivity"]) >= 0.95:
            rows.append((
                float(metrics["f1"]),
                float(metrics["specificity"]),
                float(threshold),
                metrics,
            ))
    if not rows:
        raise ValueError("No OOF CDF threshold satisfies sensitivity constraint")
    _, _, threshold, metrics = max(rows, key=lambda row: row[:3])
    return {"threshold": threshold, **{key: value for key, value in metrics.items() if key != "confusion_matrix"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.run_dir) != 5:
        raise ValueError("Exactly five outer-fold run directories are required")

    fold_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    seen_folds: set[int] = set()
    for run_dir in args.run_dir:
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        rule = json.loads((run_dir / "presence_rule.json").read_text(encoding="utf-8"))
        outer_fold = int(config["outer_fold"])
        if outer_fold in seen_folds:
            raise ValueError(f"Duplicate outer fold: {outer_fold}")
        seen_folds.add(outer_fold)
        method = str(rule["pooling"])
        score_column = f"score_{method}"
        calibration = pd.read_csv(
            run_dir / "best_calibration_study_predictions.csv",
            dtype={"study_id": str},
        )
        outer = pd.read_csv(
            run_dir / "outer_study_predictions.csv",
            dtype={"study_id": str},
        )
        scores = outer[score_column].to_numpy(dtype=np.float64)
        truth = outer["truth_any_ich"].to_numpy(dtype=np.int64)
        predicted = scores >= float(rule["threshold"])
        outer = outer.assign(
            outer_fold=outer_fold,
            selected_pooling=method,
            selected_threshold=float(rule["threshold"]),
            selected_score=scores,
            calibrated_cdf_score=_empirical_cdf(
                calibration[score_column].to_numpy(dtype=np.float64), scores
            ),
            predicted_by_fold_rule=predicted.astype(np.uint8),
        )
        metrics = _binary_metrics(truth, predicted)
        fold_metrics.append({
            "outer_fold": outer_fold,
            "studies": int(len(outer)),
            "positives": int(truth.sum()),
            "pooling": method,
            "threshold": float(rule["threshold"]),
            "roc_auc": float(roc_auc_score(truth, scores)),
            **metrics,
        })
        fold_rows.append(outer)

    if seen_folds != set(range(5)):
        raise ValueError(f"Expected folds 0..4, got {sorted(seen_folds)}")
    oof = pd.concat(fold_rows, ignore_index=True).sort_values(
        ["outer_fold", "study_id"]
    ).reset_index(drop=True)
    if oof["study_id"].duplicated().any() or len(oof) != 338:
        raise ValueError("OOF must contain every study exactly once")
    truth = oof["truth_any_ich"].to_numpy(dtype=np.int64)
    fold_predicted = oof["predicted_by_fold_rule"].to_numpy(dtype=np.int64)
    cdf_scores = oof["calibrated_cdf_score"].to_numpy(dtype=np.float64)

    subtype_metrics: dict[str, dict[str, object]] = {}
    for label in OUTPUT_LABELS[1:]:
        per_fold: dict[str, float | None] = {}
        aucs = []
        for fold, group in oof.groupby("outer_fold"):
            subtype_truth = group[f"truth_{label}"].to_numpy(dtype=np.int64)
            if len(np.unique(subtype_truth)) < 2:
                per_fold[str(fold)] = None
                continue
            auc = float(roc_auc_score(subtype_truth, group[f"score_{label}"]))
            per_fold[str(fold)] = auc
            aucs.append(auc)
        subtype_metrics[label] = {
            "per_fold_auc": per_fold,
            "macro_auc_available_folds": float(np.mean(aucs)) if aucs else None,
            "worst_auc_available_folds": float(np.min(aucs)) if aucs else None,
            "positive_studies": int(oof[f"truth_{label}"].sum()),
        }

    summary = {
        "studies": len(oof),
        "positive_studies": int(truth.sum()),
        "folds": sorted(fold_metrics, key=lambda row: int(row["outer_fold"])),
        "macro_outer_auc": float(np.mean([row["roc_auc"] for row in fold_metrics])),
        "worst_outer_auc": float(np.min([row["roc_auc"] for row in fold_metrics])),
        "pooled_calibration_cdf_auc": float(roc_auc_score(truth, cdf_scores)),
        "unbiased_per_fold_rules": _binary_metrics(truth, fold_predicted),
        "diagnostic_all_oof_cdf_rule": _diagnostic_threshold(truth, cdf_scores),
        "subtypes": subtype_metrics,
        "warning": "The all-OOF CDF threshold is selected on the same OOF labels and is diagnostic; per-fold rules are the unbiased binary estimate.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.output_dir / "oof_study_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
