"""Evaluate a per-study deployable detector/MIL empirical-CDF blend."""

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

from scripts.evaluate_fracture_mil_oof import _interval, _macro_paired_bootstrap
from scripts.train_fracture_smooth_attention_mil import (
    Standardizer,
    _predict,
    _prepare_features,
    _study_bags,
)
from src.fracture.pooling import aggregate_study_scores
from src.fracture.smooth_attention_mil import (
    SmoothAttentionMIL,
    SmoothAttentionMILConfig,
)


def _empirical_cdf(training: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map independent values through a mid-rank CDF fit only on training data."""
    training = np.sort(np.asarray(training, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if training.ndim != 1 or training.size == 0 or not np.isfinite(training).all():
        raise ValueError("training must be a non-empty finite vector")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector")
    left = np.searchsorted(training, values, side="left")
    right = np.searchsorted(training, values, side="right")
    return (0.5 * (left + right) + 0.5) / (training.size + 1.0)


def _load_models(
    directory: Path,
    device: torch.device,
) -> tuple[list[SmoothAttentionMIL], Standardizer, list[int]]:
    model_paths = sorted(directory.glob("model_seed*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No MIL models found in {directory}")
    models: list[SmoothAttentionMIL] = []
    standardizer: Standardizer | None = None
    seeds: list[int] = []
    for path in model_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = SmoothAttentionMILConfig(**payload["config"])
        model = SmoothAttentionMIL(config).to(device)
        model.load_state_dict(payload["state_dict"])
        models.append(model)
        raw = payload["standardizer"]
        candidate_standardizer = Standardizer(
            embedding_mean=raw["embedding_mean"].numpy(),
            embedding_scale=raw["embedding_scale"].numpy(),
            score_mean=float(raw["score_mean"]),
            score_scale=float(raw["score_scale"]),
        )
        if standardizer is None:
            standardizer = candidate_standardizer
        else:
            np.testing.assert_array_equal(
                standardizer.embedding_mean,
                candidate_standardizer.embedding_mean,
            )
            np.testing.assert_array_equal(
                standardizer.embedding_scale,
                candidate_standardizer.embedding_scale,
            )
            if standardizer.score_mean != candidate_standardizer.score_mean:
                raise RuntimeError("Seed checkpoints use different score means")
            if standardizer.score_scale != candidate_standardizer.score_scale:
                raise RuntimeError("Seed checkpoints use different score scales")
        seeds.append(int(payload["metadata"]["seed"]))
    if standardizer is None:
        raise RuntimeError("MIL standardizer was not loaded")
    return models, standardizer, seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--crossfit-summary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    crossfit = json.loads(args.crossfit_summary.read_text(encoding="utf-8"))
    weight_by_fold = {
        int(row["held_out_fold"]): float(row["selected_candidate_weight"])
        for row in crossfit["selections"]
    }
    if sorted(weight_by_fold) != [0, 1, 2, 3, 4]:
        raise ValueError("Cross-fit summary must contain weights for folds 0..4")

    outer_rows: list[pd.DataFrame] = []
    calibration_manifest: list[dict[str, object]] = []
    for fold in range(5):
        cache = args.cache_root / f"fold_{fold}"
        model_directory = args.model_root / f"fold_{fold}_v2"
        slices = pd.read_csv(
            cache / "slices.csv", dtype={"study_id": str, "patient_id": str}
        )
        embeddings = np.load(
            cache / "embeddings.npy", allow_pickle=False, mmap_mode="r"
        )
        slice_scores = np.load(cache / "slice_scores.npy", allow_pickle=False)
        bags = _study_bags(slices)
        models, standardizer, seeds = _load_models(model_directory, device)
        prepared = _prepare_features(
            embeddings, slice_scores, bags, standardizer, device
        )
        seed_probabilities = [
            _predict(model, bags, prepared)[0] for model in models
        ]
        mil_score = np.mean(np.stack(seed_probabilities), axis=0)
        adjacent = np.asarray(
            [
                aggregate_study_scores(slice_scores[bag.indices])["adjacent_pair"]
                for bag in bags
            ],
            dtype=np.float64,
        )
        truth = np.asarray([bag.truth for bag in bags], dtype=np.int64)
        assigned_fold = np.asarray([bag.outer_fold for bag in bags], dtype=np.int64)
        training = assigned_fold != fold
        validation = assigned_fold == fold
        reference_cdf = _empirical_cdf(adjacent[training], adjacent[validation])
        candidate_cdf = _empirical_cdf(mil_score[training], mil_score[validation])
        weight = weight_by_fold[fold]
        blend = (1.0 - weight) * reference_cdf + weight * candidate_cdf
        selected_bags = [bag for bag, selected in zip(bags, validation, strict=True) if selected]
        outer_rows.append(
            pd.DataFrame(
                {
                    "study_id": [bag.study_id for bag in selected_bags],
                    "patient_id": [bag.patient_id for bag in selected_bags],
                    "truth": truth[validation],
                    "outer_fold": fold,
                    "prob_adjacent_pair": adjacent[validation],
                    "mil_score": mil_score[validation],
                    "reference_train_cdf": reference_cdf,
                    "candidate_train_cdf": candidate_cdf,
                    "deployable_blend_score": blend,
                }
            )
        )
        calibration_manifest.append(
            {
                "fold": fold,
                "candidate_weight": weight,
                "n_training_studies": int(training.sum()),
                "model_seeds": seeds,
                "reference_training_scores": adjacent[training].tolist(),
                "candidate_training_scores": mil_score[training].tolist(),
            }
        )

    predictions = pd.concat(outer_rows, ignore_index=True)
    if predictions["study_id"].duplicated().any():
        raise RuntimeError("Duplicate studies in deployable OOF predictions")
    per_fold = []
    for fold, frame in predictions.groupby("outer_fold", sort=True):
        reference_auc = float(
            roc_auc_score(frame["truth"], frame["prob_adjacent_pair"])
        )
        blend_auc = float(
            roc_auc_score(frame["truth"], frame["deployable_blend_score"])
        )
        per_fold.append(
            {
                "fold": int(fold),
                "reference_auc": reference_auc,
                "blend_auc": blend_auc,
                "difference": blend_auc - reference_auc,
                "candidate_weight": weight_by_fold[int(fold)],
            }
        )
    reference_bootstrap, blend_bootstrap = _macro_paired_bootstrap(
        predictions,
        "prob_adjacent_pair",
        "deployable_blend_score",
        iterations=args.iterations,
        seed=args.seed,
    )
    difference = blend_bootstrap - reference_bootstrap
    reference_auc = np.asarray([row["reference_auc"] for row in per_fold])
    blend_auc = np.asarray([row["blend_auc"] for row in per_fold])
    payload = {
        "method": "outer_train_empirical_cdf_crossfit_weight_blend",
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
    predictions.to_csv(args.output / "oof_predictions.csv", index=False)
    (args.output / "calibration_manifest.json").write_text(
        json.dumps(calibration_manifest, indent=2) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
