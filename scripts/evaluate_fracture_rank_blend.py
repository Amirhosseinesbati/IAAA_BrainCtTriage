"""Select a detector rank blend on one fold and confirm it on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.compare_fracture_study_predictions import _sampled_auc


def _average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks in (0, 1], preserving score ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _rank_blend(
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    candidate_weight: float,
) -> np.ndarray:
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate_weight must be in [0, 1]")
    reference_rank = _average_percentile_rank(reference_score)
    candidate_rank = _average_percentile_rank(candidate_score)
    return (1.0 - candidate_weight) * reference_rank + candidate_weight * candidate_rank


def _blend_curve(
    truth: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, float]]:
    return [
        {
            "candidate_weight": float(weight),
            "auc": float(
                roc_auc_score(
                    truth,
                    _rank_blend(reference_score, candidate_score, float(weight)),
                )
            ),
        }
        for weight in weights
    ]


def _select_weight(curve: list[dict[str, float]]) -> dict[str, float]:
    """Maximise development AUC; prefer less candidate weight on exact ties."""
    if not curve:
        raise ValueError("Blend curve must not be empty")
    return max(curve, key=lambda row: (row["auc"], -row["candidate_weight"]))


def _load_pair(
    reference_path: Path,
    candidate_path: Path,
    reference_score_name: str,
    candidate_score_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = pd.read_csv(reference_path).sort_values("study_id").reset_index(drop=True)
    candidate = pd.read_csv(candidate_path).sort_values("study_id").reset_index(drop=True)
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
    return (
        reference["truth"].to_numpy(dtype=np.int64),
        reference[reference_score_name].to_numpy(dtype=np.float64),
        candidate[candidate_score_name].to_numpy(dtype=np.float64),
    )


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-reference", type=Path, required=True)
    parser.add_argument("--development-candidate", type=Path, required=True)
    parser.add_argument("--confirmation-reference", type=Path, required=True)
    parser.add_argument("--confirmation-candidate", type=Path, required=True)
    parser.add_argument("--reference-score", required=True)
    parser.add_argument("--candidate-score", required=True)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0.0 < args.weight_step <= 1.0:
        raise ValueError("weight-step must be in (0, 1]")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    weights = np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step)
    weights = np.clip(weights, 0.0, 1.0)

    development_truth, development_reference, development_candidate = _load_pair(
        args.development_reference,
        args.development_candidate,
        args.reference_score,
        args.candidate_score,
    )
    confirmation_truth, confirmation_reference, confirmation_candidate = _load_pair(
        args.confirmation_reference,
        args.confirmation_candidate,
        args.reference_score,
        args.candidate_score,
    )
    development_curve = _blend_curve(
        development_truth,
        development_reference,
        development_candidate,
        weights,
    )
    selected = _select_weight(development_curve)
    selected_weight = selected["candidate_weight"]
    confirmation_blend = _rank_blend(
        confirmation_reference,
        confirmation_candidate,
        selected_weight,
    )

    positive = np.flatnonzero(confirmation_truth == 1)
    negative = np.flatnonzero(confirmation_truth == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both classes are required in the confirmation fold")
    confirmation_reference_rank = _average_percentile_rank(confirmation_reference)
    rng = np.random.default_rng(args.seed)
    reference_auc = np.empty(args.iterations, dtype=np.float64)
    blend_auc = np.empty(args.iterations, dtype=np.float64)
    for start in range(0, args.iterations, 2_000):
        stop = min(start + 2_000, args.iterations)
        size = stop - start
        sampled_negative = rng.choice(negative, size=(size, negative.size), replace=True)
        sampled_positive = rng.choice(positive, size=(size, positive.size), replace=True)
        reference_auc[start:stop] = _sampled_auc(
            confirmation_reference_rank,
            sampled_positive,
            sampled_negative,
        )
        blend_auc[start:stop] = _sampled_auc(
            confirmation_blend,
            sampled_positive,
            sampled_negative,
        )
    difference = blend_auc - reference_auc

    payload = {
        "method": "within_fold_average_percentile_rank_blend",
        "reference_score": args.reference_score,
        "candidate_score": args.candidate_score,
        "weight_step": args.weight_step,
        "selection_rule": "max_development_auc_then_min_candidate_weight",
        "development": {
            "n_studies": int(development_truth.size),
            "n_positive": int(np.sum(development_truth == 1)),
            "reference_auc": float(roc_auc_score(development_truth, development_reference)),
            "candidate_auc": float(roc_auc_score(development_truth, development_candidate)),
            "curve": development_curve,
            "selected_candidate_weight": selected_weight,
            "selected_blend_auc": selected["auc"],
        },
        "confirmation": {
            "n_studies": int(confirmation_truth.size),
            "n_positive": int(positive.size),
            "reference_auc": float(
                roc_auc_score(confirmation_truth, confirmation_reference)
            ),
            "candidate_auc": float(
                roc_auc_score(confirmation_truth, confirmation_candidate)
            ),
            "selected_blend_auc": float(
                roc_auc_score(confirmation_truth, confirmation_blend)
            ),
            "observed_difference_vs_reference": float(
                roc_auc_score(confirmation_truth, confirmation_blend)
                - roc_auc_score(confirmation_truth, confirmation_reference)
            ),
            "difference_bootstrap_95": _interval(difference),
            "probability_blend_not_better": float(np.mean(difference <= 0.0)),
        },
        "bootstrap": {"iterations": args.iterations, "seed": args.seed},
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
