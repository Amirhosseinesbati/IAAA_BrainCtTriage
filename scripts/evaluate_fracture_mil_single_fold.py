"""Evaluate one detector/MIL fold with train-only CDF calibration."""

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
import torch
from sklearn.metrics import roc_auc_score

from scripts.compare_fracture_study_predictions import _sampled_auc
from scripts.evaluate_fracture_mil_deployable_blend import (
    _empirical_cdf,
    _load_models,
)
from scripts.train_fracture_smooth_attention_mil import (
    _predict,
    _prepare_features,
    _study_bags,
)
from src.fracture.pooling import aggregate_study_scores


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


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
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both classes are required")
    rng = np.random.default_rng(seed)
    reference_auc = np.empty(iterations, dtype=np.float64)
    candidate_auc = np.empty(iterations, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        size = stop - start
        sampled_positive = rng.choice(
            positive, size=(size, positive.size), replace=True
        )
        sampled_negative = rng.choice(
            negative, size=(size, negative.size), replace=True
        )
        reference_auc[start:stop] = _sampled_auc(
            reference, sampled_positive, sampled_negative
        )
        candidate_auc[start:stop] = _sampled_auc(
            candidate, sampled_positive, sampled_negative
        )
    difference = candidate_auc - reference_auc
    return {
        "iterations": iterations,
        "seed": seed,
        "difference_95": _interval(difference),
        "probability_candidate_not_better": float(np.mean(difference <= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--fixed-weight", type=float, default=0.45)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be in [0, 4]")
    if not 0.0 <= args.fixed_weight <= 1.0:
        raise ValueError("fixed-weight must be in [0, 1]")
    slices = pd.read_csv(
        args.cache / "slices.csv", dtype={"study_id": str, "patient_id": str}
    )
    embeddings = np.load(
        args.cache / "embeddings.npy", allow_pickle=False, mmap_mode="r"
    )
    slice_scores = np.load(args.cache / "slice_scores.npy", allow_pickle=False)
    bags = _study_bags(slices)
    device = torch.device(args.device)
    models, standardizer, seeds = _load_models(args.model_dir, device)
    prepared = _prepare_features(
        embeddings, slice_scores, bags, standardizer, device
    )
    mil_score = np.mean(
        np.stack([_predict(model, bags, prepared)[0] for model in models]), axis=0
    )
    adjacent = np.asarray(
        [
            aggregate_study_scores(slice_scores[bag.indices])["adjacent_pair"]
            for bag in bags
        ],
        dtype=np.float64,
    )
    truth = np.asarray([bag.truth for bag in bags], dtype=np.int64)
    assigned_fold = np.asarray([bag.outer_fold for bag in bags], dtype=np.int64)
    training = assigned_fold != args.outer_fold
    validation = assigned_fold == args.outer_fold
    reference_cdf = _empirical_cdf(adjacent[training], adjacent[validation])
    candidate_cdf = _empirical_cdf(mil_score[training], mil_score[validation])
    blend = (
        (1.0 - args.fixed_weight) * reference_cdf
        + args.fixed_weight * candidate_cdf
    )
    validation_truth = truth[validation]
    validation_adjacent = adjacent[validation]
    validation_mil = mil_score[validation]
    reference_auc = float(roc_auc_score(validation_truth, validation_adjacent))
    mil_auc = float(roc_auc_score(validation_truth, validation_mil))
    blend_auc = float(roc_auc_score(validation_truth, blend))
    payload = {
        "outer_fold": args.outer_fold,
        "n_training": int(training.sum()),
        "n_validation": int(validation.sum()),
        "n_validation_positive": int(validation_truth.sum()),
        "fixed_weight": args.fixed_weight,
        "model_seeds": seeds,
        "reference_auc": reference_auc,
        "mil_auc": mil_auc,
        "blend_auc": blend_auc,
        "blend_difference": blend_auc - reference_auc,
        "bootstrap": _paired_bootstrap(
            validation_truth,
            validation_adjacent,
            blend,
            iterations=args.iterations,
            seed=args.seed,
        ),
    }
    selected_bags = [
        bag for bag, selected in zip(bags, validation, strict=True) if selected
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "study_id": [bag.study_id for bag in selected_bags],
            "patient_id": [bag.patient_id for bag in selected_bags],
            "truth": validation_truth,
            "prob_adjacent_pair": validation_adjacent,
            "mil_score": validation_mil,
            "reference_train_cdf": reference_cdf,
            "candidate_train_cdf": candidate_cdf,
            "deployable_blend_score": blend,
        }
    ).to_csv(args.output / "private_predictions.csv", index=False)
    (args.output / "private_calibration_manifest.json").write_text(
        json.dumps(
            {
                "reference_training_scores": adjacent[training].tolist(),
                "candidate_training_scores": mil_score[training].tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
