"""Evaluate a pre-specified rank complement on one held-out fracture fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--reference-column", required=True)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.candidate_weight <= 1.0:
        raise ValueError("candidate-weight must be within [0, 1]")

    reference = pd.read_csv(args.reference, dtype={"study_id": str})
    if "outer_fold" not in reference:
        raise ValueError("Reference predictions need outer_fold")
    reference = reference.loc[
        reference["outer_fold"].eq(args.fold),
        ["study_id", "truth", args.reference_column],
    ]
    candidate = pd.read_csv(args.candidate, dtype={"study_id": str})[
        ["study_id", "truth", args.candidate_column]
    ]
    frame = reference.merge(candidate, on=["study_id", "truth"], validate="one_to_one")
    if len(frame) != len(reference) or frame["study_id"].duplicated().any():
        raise RuntimeError("Reference and candidate study identities do not match")
    frame["reference_rank"] = frame[args.reference_column].rank(
        method="average", pct=True
    )
    frame["candidate_rank"] = frame[args.candidate_column].rank(
        method="average", pct=True
    )
    frame["fixed_rank_blend"] = (
        (1.0 - args.candidate_weight) * frame["reference_rank"]
        + args.candidate_weight * frame["candidate_rank"]
    )
    truth = frame["truth"].to_numpy(dtype=np.int64)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    if not positive.size or not negative.size:
        raise ValueError("Held-out fold must contain both classes")
    reference_score = frame["reference_rank"].to_numpy(dtype=np.float64)
    blend_score = frame["fixed_rank_blend"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    difference = np.empty(args.iterations, dtype=np.float64)
    for iteration in range(args.iterations):
        indices = np.concatenate(
            (
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            )
        )
        difference[iteration] = roc_auc_score(
            truth[indices], blend_score[indices]
        ) - roc_auc_score(truth[indices], reference_score[indices])

    reference_auc = float(roc_auc_score(truth, reference_score))
    candidate_auc = float(roc_auc_score(truth, frame["candidate_rank"]))
    blend_auc = float(roc_auc_score(truth, blend_score))
    payload = {
        "protocol": "pre_specified_single_fold_cohort_rank_replication_diagnostic",
        "deployable": False,
        "held_out_fold": args.fold,
        "n_studies": len(frame),
        "n_positive": int(positive.size),
        "reference_column": args.reference_column,
        "candidate_column": args.candidate_column,
        "candidate_weight": args.candidate_weight,
        "reference_auc": reference_auc,
        "candidate_auc": candidate_auc,
        "fixed_rank_blend_auc": blend_auc,
        "observed_difference": blend_auc - reference_auc,
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": np.quantile(
                difference, [0.025, 0.5, 0.975]
            ).tolist(),
            "probability_blend_not_better": float(np.mean(difference <= 0.0)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "private_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
