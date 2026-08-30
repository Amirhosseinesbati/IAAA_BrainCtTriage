"""Characterize fracture MIL OOF errors and emit leakage-safe mining evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _outcome(truth: int, prediction: int) -> str:
    return { (0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP" }[
        (int(truth), int(prediction))
    ]


def _driver(reference_cdf: float, candidate_cdf: float, tolerance: float = 0.10) -> str:
    difference = float(reference_cdf) - float(candidate_cdf)
    if difference > tolerance:
        return "detector_dominant"
    if difference < -tolerance:
        return "mil_dominant"
    return "joint"


def _summary_table(frame: pd.DataFrame, group: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for value, rows in frame.groupby(group, sort=True):
        result[str(value)] = {
            "n": int(len(rows)),
            "median_n_slices": float(rows["n_slices"].median()),
            "median_adjacent_pair": float(rows["prob_adjacent_pair"].median()),
            "median_mil_score": float(rows["mil_score"].median()),
            "median_reference_cdf": float(rows["reference_train_cdf"].median()),
            "median_candidate_cdf": float(rows["candidate_train_cdf"].median()),
            "median_blend_score": float(rows["deployable_blend_score"].median()),
            "median_decision_margin": float(rows["decision_margin"].median()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions, dtype={"study_id": str, "patient_id": str}
    )
    required = {
        "study_id",
        "patient_id",
        "truth",
        "outer_fold",
        "prob_adjacent_pair",
        "mil_score",
        "reference_train_cdf",
        "candidate_train_cdf",
        "deployable_blend_score",
        "candidate_threshold",
        "candidate_binary",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing: {sorted(missing)}")
    if predictions["study_id"].duplicated().any():
        raise ValueError("Each study must appear exactly once")

    caches: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    for fold in range(5):
        directory = args.cache_root / f"fold_{fold}"
        slices = pd.read_csv(directory / "slices.csv", dtype={"study_id": str})
        scores = np.load(directory / "slice_scores.npy", allow_pickle=False)
        if len(slices) != len(scores):
            raise RuntimeError(f"Fold {fold} cache rows/scores differ")
        caches[fold] = (slices, scores)

    details: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        fold = int(row.outer_fold)
        slices, scores = caches[fold]
        selected = slices.loc[slices["study_id"].eq(str(row.study_id))].sort_values(
            "slice_index", kind="stable"
        )
        if selected.empty:
            raise RuntimeError(f"Study {row.study_id} is absent from fold {fold} cache")
        study_scores = scores[selected.index.to_numpy(dtype=np.int64)]
        max_position = int(np.argmax(study_scores))
        if study_scores.size >= 2:
            adjacent_values = np.sqrt(study_scores[:-1] * study_scores[1:])
            adjacent_position = int(np.argmax(adjacent_values))
        else:
            adjacent_position = 0
        details.append(
            {
                **row._asdict(),
                "outcome": _outcome(row.truth, row.candidate_binary),
                "driver": _driver(row.reference_train_cdf, row.candidate_train_cdf),
                "decision_margin": float(
                    row.deployable_blend_score - row.candidate_threshold
                ),
                "n_slices": int(len(selected)),
                "max_slice_score": float(study_scores[max_position]),
                "max_slice_index": int(selected.iloc[max_position]["slice_index"]),
                "max_slice_image_path": str(selected.iloc[max_position]["image_path"]),
                "adjacent_start_slice_index": int(
                    selected.iloc[adjacent_position]["slice_index"]
                ),
                "adjacent_start_image_path": str(
                    selected.iloc[adjacent_position]["image_path"]
                ),
                "adjacent_next_image_path": str(
                    selected.iloc[min(adjacent_position + 1, len(selected) - 1)][
                        "image_path"
                    ]
                ),
                "slices_score_ge_010": int(np.sum(study_scores >= 0.10)),
                "slices_score_ge_025": int(np.sum(study_scores >= 0.25)),
                "slices_score_ge_050": int(np.sum(study_scores >= 0.50)),
            }
        )

    detail_frame = pd.DataFrame.from_records(details)
    errors = detail_frame.loc[detail_frame["outcome"].isin(["FP", "FN"])].copy()
    errors = errors.sort_values(
        ["outcome", "decision_margin"], ascending=[True, False], kind="stable"
    )
    per_fold = (
        detail_frame.groupby(["outer_fold", "outcome"]).size().unstack(fill_value=0)
    )
    payload = {
        "n_studies": int(len(detail_frame)),
        "outcome_counts": {
            str(key): int(value)
            for key, value in detail_frame["outcome"].value_counts().sort_index().items()
        },
        "error_driver_counts": {
            outcome: {
                str(key): int(value)
                for key, value in group["driver"].value_counts().sort_index().items()
            }
            for outcome, group in errors.groupby("outcome", sort=True)
        },
        "per_fold_counts": {
            str(int(fold)): {str(key): int(value) for key, value in row.items()}
            for fold, row in per_fold.iterrows()
        },
        "outcome_feature_summary": _summary_table(detail_frame, "outcome"),
        "error_driver_feature_summary": _summary_table(errors, "driver"),
        "mining_policy": (
            "Use only each study's assigned outer-fold detector/cache. That detector "
            "never trained on the study, so ranked negative studies and slice paths "
            "are leakage-safe inputs for a different target fold's outer-train set."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    detail_frame.to_csv(args.output / "private_all_studies.csv", index=False)
    errors.to_csv(args.output / "private_ranked_errors.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "aggregate_summary.json").write_text(
        rendered + "\n", encoding="utf-8"
    )
    print(rendered)


if __name__ == "__main__":
    main()
