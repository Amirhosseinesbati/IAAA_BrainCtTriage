"""Cross-fit a detector/MIL rank blend without tuning on the held-out fold."""

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

from scripts.evaluate_fracture_mil_oof import _interval, _macro_paired_bootstrap


def _rank(values: pd.Series) -> np.ndarray:
    return values.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _blend(reference: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * reference + weight * candidate


def _select_weight(
    development: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for weight in weights:
        fold_aucs = []
        for _, fold in development.groupby("outer_fold", sort=True):
            score = _blend(
                fold["reference_rank"].to_numpy(dtype=np.float64),
                fold["candidate_rank"].to_numpy(dtype=np.float64),
                float(weight),
            )
            fold_aucs.append(float(roc_auc_score(fold["truth"], score)))
        rows.append(
            {
                "candidate_weight": float(weight),
                "macro_auc": float(np.mean(fold_aucs)),
                "worst_auc": float(np.min(fold_aucs)),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["macro_auc"],
            row["worst_auc"],
            -row["candidate_weight"],
        ),
    )
    return {"selected": selected, "curve": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-score", default="prob_adjacent_pair")
    parser.add_argument("--candidate-score", default="mil_score")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < args.weight_step <= 1.0:
        raise ValueError("weight-step must be in (0, 1]")
    predictions = pd.read_csv(
        args.predictions, dtype={"study_id": str, "patient_id": str}
    )
    required = {
        "study_id",
        "patient_id",
        "truth",
        "outer_fold",
        args.reference_score,
        args.candidate_score,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if sorted(predictions["outer_fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Exactly outer folds 0..4 are required")
    predictions["reference_rank"] = predictions.groupby("outer_fold")[
        args.reference_score
    ].rank(method="average", pct=True)
    predictions["candidate_rank"] = predictions.groupby("outer_fold")[
        args.candidate_score
    ].rank(method="average", pct=True)
    predictions["crossfit_blend_score"] = np.nan
    weights = np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step)
    weights = np.clip(weights, 0.0, 1.0)
    selections: list[dict[str, object]] = []
    for held_out_fold in range(5):
        development = predictions.loc[
            predictions["outer_fold"].ne(held_out_fold)
        ]
        selection = _select_weight(development, weights)
        selected_weight = float(selection["selected"]["candidate_weight"])
        held_out = predictions["outer_fold"].eq(held_out_fold)
        predictions.loc[held_out, "crossfit_blend_score"] = _blend(
            predictions.loc[held_out, "reference_rank"].to_numpy(dtype=np.float64),
            predictions.loc[held_out, "candidate_rank"].to_numpy(dtype=np.float64),
            selected_weight,
        )
        selections.append(
            {
                "held_out_fold": held_out_fold,
                "selected_candidate_weight": selected_weight,
                "development_selected_macro_auc": selection["selected"]["macro_auc"],
                "development_selected_worst_auc": selection["selected"]["worst_auc"],
                "curve": selection["curve"],
            }
        )
    if predictions["crossfit_blend_score"].isna().any():
        raise RuntimeError("Cross-fit blend left missing predictions")

    per_fold = []
    for fold, frame in predictions.groupby("outer_fold", sort=True):
        reference_auc = float(roc_auc_score(frame["truth"], frame["reference_rank"]))
        blend_auc = float(
            roc_auc_score(frame["truth"], frame["crossfit_blend_score"])
        )
        per_fold.append(
            {
                "fold": int(fold),
                "reference_auc": reference_auc,
                "blend_auc": blend_auc,
                "difference": blend_auc - reference_auc,
                "selected_candidate_weight": selections[int(fold)][
                    "selected_candidate_weight"
                ],
            }
        )
    reference_bootstrap, blend_bootstrap = _macro_paired_bootstrap(
        predictions,
        "reference_rank",
        "crossfit_blend_score",
        iterations=args.iterations,
        seed=args.seed,
    )
    difference = blend_bootstrap - reference_bootstrap
    reference_auc = np.asarray([row["reference_auc"] for row in per_fold])
    blend_auc = np.asarray([row["blend_auc"] for row in per_fold])
    payload = {
        "method": "leave_one_outer_fold_out_rank_blend_selection",
        "reference_score": args.reference_score,
        "candidate_score": args.candidate_score,
        "weight_step": args.weight_step,
        "selection_rule": "max_development_macro_then_worst_then_min_candidate_weight",
        "selections": selections,
        "per_fold": per_fold,
        "reference_macro_auc": float(reference_auc.mean()),
        "blend_macro_auc": float(blend_auc.mean()),
        "macro_difference": float(blend_auc.mean() - reference_auc.mean()),
        "reference_worst_fold_auc": float(reference_auc.min()),
        "blend_worst_fold_auc": float(blend_auc.min()),
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": _interval(difference),
            "probability_blend_not_better": float(np.mean(difference <= 0.0)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "crossfit_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
