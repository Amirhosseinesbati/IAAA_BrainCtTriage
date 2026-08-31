#!/usr/bin/env python3
"""Paired patient-cluster bootstrap for legacy/current fracture predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--legacy-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-score-column", default="deployable_blend_score")
    parser.add_argument("--candidate-binary-column", default="candidate_binary")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _metrics(
    truth: np.ndarray,
    legacy_score: np.ndarray,
    candidate_score: np.ndarray,
    legacy_binary: np.ndarray,
    candidate_binary: np.ndarray,
) -> dict[str, float]:
    return {
        "legacy_auc": float(roc_auc_score(truth, legacy_score)),
        "candidate_auc": float(roc_auc_score(truth, candidate_score)),
        "auc_difference": float(
            roc_auc_score(truth, candidate_score)
            - roc_auc_score(truth, legacy_score)
        ),
        "legacy_f1": float(f1_score(truth, legacy_binary, zero_division=0)),
        "candidate_f1": float(f1_score(truth, candidate_binary, zero_division=0)),
        "f1_difference": float(
            f1_score(truth, candidate_binary, zero_division=0)
            - f1_score(truth, legacy_binary, zero_division=0)
        ),
    }


def _quantiles(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def main() -> None:
    args = _parse_args()
    legacy = pd.read_csv(args.legacy)
    candidate = pd.read_csv(args.candidate)

    legacy = legacy.loc[legacy["fold"].astype(int) == args.fold].copy()
    candidate = candidate.loc[candidate["outer_fold"].astype(int) == args.fold].copy()
    legacy = legacy.rename(
        columns={
            "gt_fracture_prob": "legacy_truth",
            "pred_fracture_prob": "legacy_score",
        }
    )
    candidate = candidate.rename(columns={"truth": "candidate_truth"})
    required_candidate = [
        "study_id",
        "patient_id",
        "candidate_truth",
        args.candidate_score_column,
        args.candidate_binary_column,
    ]
    merged = legacy[
        ["study_id", "patient_id", "legacy_truth", "legacy_score"]
    ].merge(
        candidate[required_candidate],
        on=["study_id", "patient_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(legacy) or len(merged) != len(candidate):
        raise ValueError(
            "Predictions do not have identical fold coverage: "
            f"legacy={len(legacy)}, candidate={len(candidate)}, merged={len(merged)}"
        )
    truth = merged["legacy_truth"].astype(int).to_numpy()
    candidate_truth = merged["candidate_truth"].astype(int).to_numpy()
    if not np.array_equal(truth, candidate_truth):
        raise ValueError("Legacy and candidate fracture truths disagree")

    merged["truth"] = truth
    merged["legacy_binary"] = (
        merged["legacy_score"].to_numpy(dtype=float) >= args.legacy_threshold
    ).astype(int)
    merged["candidate_binary"] = merged[args.candidate_binary_column].astype(int)

    point = _metrics(
        truth,
        merged["legacy_score"].to_numpy(dtype=float),
        merged[args.candidate_score_column].to_numpy(dtype=float),
        merged["legacy_binary"].to_numpy(dtype=int),
        merged["candidate_binary"].to_numpy(dtype=int),
    )

    patient_truth = merged.groupby("patient_id", sort=True)["truth"].max()
    positive_patients = patient_truth.index[patient_truth == 1].to_numpy()
    negative_patients = patient_truth.index[patient_truth == 0].to_numpy()
    patient_rows = {
        patient_id: group.index.to_numpy()
        for patient_id, group in merged.groupby("patient_id", sort=False)
    }
    rng = np.random.default_rng(args.seed)
    auc_differences: list[float] = []
    f1_differences: list[float] = []
    for _ in range(args.iterations):
        sampled = np.concatenate(
            [
                rng.choice(
                    positive_patients, size=len(positive_patients), replace=True
                ),
                rng.choice(
                    negative_patients, size=len(negative_patients), replace=True
                ),
            ]
        )
        indices = np.concatenate([patient_rows[patient_id] for patient_id in sampled])
        frame = merged.loc[indices]
        metrics = _metrics(
            frame["truth"].to_numpy(dtype=int),
            frame["legacy_score"].to_numpy(dtype=float),
            frame[args.candidate_score_column].to_numpy(dtype=float),
            frame["legacy_binary"].to_numpy(dtype=int),
            frame["candidate_binary"].to_numpy(dtype=int),
        )
        auc_differences.append(metrics["auc_difference"])
        f1_differences.append(metrics["f1_difference"])

    payload = {
        "protocol": "paired_stratified_patient_cluster_bootstrap",
        "fold": args.fold,
        "n_studies": int(len(merged)),
        "n_patients": int(merged["patient_id"].nunique()),
        "n_positive_studies": int(truth.sum()),
        "n_positive_patients": int(len(positive_patients)),
        "legacy_threshold": args.legacy_threshold,
        "candidate_score_column": args.candidate_score_column,
        "candidate_binary_column": args.candidate_binary_column,
        "point_estimates": point,
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "auc_difference_95": _quantiles(auc_differences),
            "f1_difference_95": _quantiles(f1_differences),
            "probability_candidate_auc_not_better": float(
                np.mean(np.asarray(auc_differences) <= 0.0)
            ),
            "probability_candidate_f1_not_better": float(
                np.mean(np.asarray(f1_differences) <= 0.0)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
