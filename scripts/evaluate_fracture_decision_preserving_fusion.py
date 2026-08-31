"""Evaluate snapshot ranking while exactly preserving incumbent binary decisions."""

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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from scripts.evaluate_fracture_mil_oof import _interval, _macro_paired_bootstrap


def _decision_preserving_score(
    incumbent: np.ndarray, ranking: np.ndarray, threshold: np.ndarray | float
) -> np.ndarray:
    incumbent = np.asarray(incumbent, dtype=np.float64)
    ranking = np.clip(np.asarray(ranking, dtype=np.float64), 0.0, 1.0)
    threshold = np.asarray(threshold, dtype=np.float64)
    if incumbent.shape != ranking.shape or not np.isfinite(incumbent).all():
        raise ValueError("Incumbent and ranking scores must be aligned and finite")
    return np.where(incumbent < threshold, 0.5 * ranking, 0.5 + 0.5 * ranking)


def _classification_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = score >= 0.5
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--incumbent-threshold-summary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(
        args.predictions, dtype={"study_id": str, "patient_id": str}
    )
    required = {
        "study_id",
        "patient_id",
        "truth",
        "outer_fold",
        "deployable_blend_score",
        "deployable_snapshot_fusion_score",
    }
    missing = required.difference(frame.columns)
    if missing or frame["study_id"].duplicated().any():
        raise ValueError(f"Invalid prediction table; missing={sorted(missing)}")
    threshold_payload = json.loads(
        args.incumbent_threshold_summary.read_text(encoding="utf-8")
    )
    thresholds = {
        int(row["held_out_fold"]): float(row["candidate_threshold"])
        for row in threshold_payload["selections"]
    }
    if set(thresholds) != set(range(5)):
        raise ValueError("Incumbent threshold summary must contain all five folds")
    deployment_threshold = float(
        threshold_payload["deployment"]["selected_threshold_all_oof"]
    )
    incumbent = frame["deployable_blend_score"].to_numpy(dtype=np.float64)
    ranking = frame["deployable_snapshot_fusion_score"].to_numpy(dtype=np.float64)
    crossfit_threshold = frame["outer_fold"].map(thresholds).to_numpy(dtype=np.float64)
    frame["decision_preserving_crossfit_score"] = _decision_preserving_score(
        incumbent, ranking, crossfit_threshold
    )
    frame["decision_preserving_deployment_score"] = _decision_preserving_score(
        incumbent, ranking, deployment_threshold
    )
    expected_binary = incumbent >= crossfit_threshold
    actual_binary = frame["decision_preserving_crossfit_score"].to_numpy() >= 0.5
    if not np.array_equal(expected_binary, actual_binary):
        raise RuntimeError("Decision-preserving mapping changed an incumbent decision")

    per_fold: list[dict[str, float | int]] = []
    for fold, heldout in frame.groupby("outer_fold", sort=True):
        reference_auc = float(
            roc_auc_score(heldout["truth"], heldout["deployable_blend_score"])
        )
        candidate_auc = float(
            roc_auc_score(
                heldout["truth"], heldout["decision_preserving_crossfit_score"]
            )
        )
        deployment_auc = float(
            roc_auc_score(
                heldout["truth"], heldout["decision_preserving_deployment_score"]
            )
        )
        per_fold.append(
            {
                "fold": int(fold),
                "reference_auc": reference_auc,
                "crossfit_candidate_auc": candidate_auc,
                "deployment_candidate_auc": deployment_auc,
                "crossfit_difference": candidate_auc - reference_auc,
            }
        )

    reference_bootstrap, candidate_bootstrap = _macro_paired_bootstrap(
        frame,
        "deployable_blend_score",
        "decision_preserving_crossfit_score",
        iterations=args.iterations,
        seed=args.seed,
    )
    difference = candidate_bootstrap - reference_bootstrap
    reference_auc = np.asarray([row["reference_auc"] for row in per_fold])
    candidate_auc = np.asarray([row["crossfit_candidate_auc"] for row in per_fold])
    deployment_auc = np.asarray(
        [row["deployment_candidate_auc"] for row in per_fold]
    )
    truth = frame["truth"].to_numpy(dtype=np.int64)
    payload = {
        "method": "incumbent_decision_preserving_snapshot_ranking",
        "protocol": "leave_one_outer_fold_out_incumbent_thresholds",
        "per_fold": per_fold,
        "reference_macro_auc": float(reference_auc.mean()),
        "crossfit_candidate_macro_auc": float(candidate_auc.mean()),
        "crossfit_macro_difference": float(candidate_auc.mean() - reference_auc.mean()),
        "crossfit_candidate_worst_fold_auc": float(candidate_auc.min()),
        "deployment_candidate_macro_auc": float(deployment_auc.mean()),
        "deployment_candidate_worst_fold_auc": float(deployment_auc.min()),
        "crossfit_classification": _classification_metrics(
            truth, frame["decision_preserving_crossfit_score"].to_numpy()
        ),
        "deployment_classification_apparent": _classification_metrics(
            truth, frame["decision_preserving_deployment_score"].to_numpy()
        ),
        "deployment_incumbent_threshold": deployment_threshold,
        "incumbent_decisions_preserved": True,
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": _interval(difference),
            "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "private_oof_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
