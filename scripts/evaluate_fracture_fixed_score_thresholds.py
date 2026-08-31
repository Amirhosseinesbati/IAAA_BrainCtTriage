"""Cross-fit decision thresholds for two fixed fracture study scores."""

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

from scripts.evaluate_fracture_mil_thresholds import (
    _binary_metrics,
    _map_threshold_to_half,
    _paired_bootstrap_f1,
    _select_f1_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-column", required=True)
    parser.add_argument("--candidate-column", required=True)
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
        args.reference_column,
        args.candidate_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame["study_id"].duplicated().any():
        raise ValueError("A study appears more than once")
    score_columns = [args.reference_column, args.candidate_column]
    if not np.isfinite(frame[score_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite scores found")

    rows: list[pd.DataFrame] = []
    selections: list[dict[str, float | int]] = []
    for fold in sorted(frame["outer_fold"].unique()):
        development = frame["outer_fold"].ne(fold)
        heldout = frame["outer_fold"].eq(fold)
        truth = frame.loc[development, "truth"].to_numpy(dtype=np.int64)
        reference_threshold = _select_f1_threshold(
            truth,
            frame.loc[development, args.reference_column].to_numpy(dtype=np.float64),
        )
        candidate_threshold = _select_f1_threshold(
            truth,
            frame.loc[development, args.candidate_column].to_numpy(dtype=np.float64),
        )
        selected = frame.loc[heldout].copy()
        selected["reference_threshold"] = reference_threshold
        selected["candidate_threshold"] = candidate_threshold
        selected["reference_binary"] = (
            selected[args.reference_column] >= reference_threshold
        ).astype(np.int64)
        selected["candidate_binary"] = (
            selected[args.candidate_column] >= candidate_threshold
        ).astype(np.int64)
        selected["fracture_prob"] = _map_threshold_to_half(
            selected[args.candidate_column].to_numpy(dtype=np.float64),
            candidate_threshold,
        )
        rows.append(selected)
        selections.append(
            {
                "held_out_fold": int(fold),
                "n_development": int(development.sum()),
                "reference_threshold": reference_threshold,
                "candidate_threshold": candidate_threshold,
            }
        )

    predictions = pd.concat(rows, ignore_index=True).sort_values("study_id")
    truth = predictions["truth"].to_numpy(dtype=np.int64)
    reference_metrics = _binary_metrics(truth, predictions["reference_binary"])
    candidate_metrics = _binary_metrics(truth, predictions["candidate_binary"])
    final_threshold = _select_f1_threshold(
        truth, predictions[args.candidate_column].to_numpy(dtype=np.float64)
    )
    final_apparent = _binary_metrics(
        truth,
        predictions[args.candidate_column].to_numpy(dtype=np.float64)
        >= final_threshold,
    )
    difference = _paired_bootstrap_f1(
        predictions,
        "reference_binary",
        "candidate_binary",
        iterations=args.iterations,
        seed=args.seed,
    )
    payload = {
        "protocol": "leave_one_outer_fold_out_fixed_score_f1_threshold_selection",
        "reference_score": args.reference_column,
        "candidate_score": args.candidate_column,
        "selections": selections,
        "reference_crossfit": reference_metrics,
        "candidate_crossfit": candidate_metrics,
        "crossfit_f1_difference": float(
            candidate_metrics["f1"] - reference_metrics["f1"]
        ),
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": np.quantile(
                difference, [0.025, 0.5, 0.975]
            ).tolist(),
            "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
        },
        "deployment": {
            "score": args.candidate_column,
            "selected_threshold_all_oof": final_threshold,
            "threshold_mapping": "piecewise_linear_threshold_to_probability_0.5",
            "apparent_all_oof_metrics": final_apparent,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "private_crossfit_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    (args.output / "deployment_calibration.json").write_text(
        json.dumps(payload["deployment"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(rendered)


if __name__ == "__main__":
    main()
