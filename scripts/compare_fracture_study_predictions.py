"""Paired stratified bootstrap comparison of study-level fracture scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _sampled_auc(
    scores: np.ndarray,
    sampled_positive: np.ndarray,
    sampled_negative: np.ndarray,
) -> np.ndarray:
    """Compute exact AUCs from stratified bootstrap indices, including ties."""
    positive_scores = scores[sampled_positive][:, :, None]
    negative_scores = scores[sampled_negative][:, None, :]
    greater = np.mean(positive_scores > negative_scores, axis=(1, 2))
    tied = np.mean(positive_scores == negative_scores, axis=(1, 2))
    return greater + 0.5 * tied


def _score_columns(
    shared: str,
    reference: str | None,
    candidate: str | None,
) -> tuple[str, str]:
    """Resolve backward-compatible, potentially asymmetric score columns."""
    return reference or shared, candidate or shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--score", default="prob_adjacent_pair")
    parser.add_argument(
        "--reference-score",
        help="Reference score column; defaults to --score.",
    )
    parser.add_argument(
        "--candidate-score",
        help="Candidate score column; defaults to --score.",
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_score_name, candidate_score_name = _score_columns(
        args.score,
        args.reference_score,
        args.candidate_score,
    )

    reference = pd.read_csv(args.reference).sort_values("study_id").reset_index(drop=True)
    candidate = pd.read_csv(args.candidate).sort_values("study_id").reset_index(drop=True)
    for name, frame, score_name in (
        ("reference", reference, reference_score_name),
        ("candidate", candidate, candidate_score_name),
    ):
        required = {"study_id", "truth", score_name}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not reference["study_id"].equals(candidate["study_id"]):
        raise ValueError("Study IDs differ between paired prediction files")
    if not reference["truth"].equals(candidate["truth"]):
        raise ValueError("Truth labels differ between paired prediction files")

    truth = reference["truth"].to_numpy(dtype=np.int64)
    reference_score = reference[reference_score_name].to_numpy(dtype=np.float64)
    candidate_score = candidate[candidate_score_name].to_numpy(dtype=np.float64)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both classes are required")

    rng = np.random.default_rng(args.seed)
    reference_auc = np.empty(args.iterations, dtype=np.float64)
    candidate_auc = np.empty(args.iterations, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, args.iterations, chunk_size):
        stop = min(start + chunk_size, args.iterations)
        size = stop - start
        sampled_negative = rng.choice(
            negative,
            size=(size, negative.size),
            replace=True,
        )
        sampled_positive = rng.choice(
            positive,
            size=(size, positive.size),
            replace=True,
        )
        reference_auc[start:stop] = _sampled_auc(
            reference_score,
            sampled_positive,
            sampled_negative,
        )
        candidate_auc[start:stop] = _sampled_auc(
            candidate_score,
            sampled_positive,
            sampled_negative,
        )
    difference = candidate_auc - reference_auc

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]

    payload = {
        "score": args.score if reference_score_name == candidate_score_name else None,
        "reference_score": reference_score_name,
        "candidate_score": candidate_score_name,
        "n_studies": int(truth.size),
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "iterations": args.iterations,
        "seed": args.seed,
        "reference_auc": float(roc_auc_score(truth, reference_score)),
        "candidate_auc": float(roc_auc_score(truth, candidate_score)),
        "observed_difference": float(
            roc_auc_score(truth, candidate_score) - roc_auc_score(truth, reference_score)
        ),
        "reference_auc_bootstrap_95": interval(reference_auc),
        "candidate_auc_bootstrap_95": interval(candidate_auc),
        "difference_bootstrap_95": interval(difference),
        "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
