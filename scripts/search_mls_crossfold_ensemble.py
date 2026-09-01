"""Search a robust study-level ensemble of binary and peak-aware MLS models.

This script performs no model inference. It consumes saved CUDA predictions,
evaluates a small set of strong pooling profiles for each model family, and
blends the resulting study predictions. Leave-one-fold-out transfer is
reported so the joint three-fold ranking is not mistaken for an untouched
estimate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from search_mls_crossfold_pooling import PoolingProfile, _metrics, _predict


PROFILE_COLUMNS = list(asdict(PoolingProfile("topk", 3, 0.0, 0.5, 1, 0.75, True, 0.0)))


def _profile_from_row(row: pd.Series) -> PoolingProfile:
    return PoolingProfile(
        family=str(row["family"]),
        size=int(row["size"]),
        component_ratio=float(row["component_ratio"]),
        selector_gate=float(row["selector_gate"]),
        min_active_slices=int(row["min_active_slices"]),
        quantile=float(row["quantile"]),
        probability_weighted=bool(row["probability_weighted"]),
        heatmap_guard_ratio=float(row["heatmap_guard_ratio"]),
    )


def _candidate_profiles(
    grid_path: Path,
    count: int,
    ranking_folds: list[str],
) -> list[PoolingProfile]:
    grid = pd.read_csv(grid_path)
    ranking = grid.copy()
    ranking["_selection_mean_mae"] = ranking[
        [f"{fold}_mae_mm" for fold in ranking_folds]
    ].mean(axis=1)
    ranking["_selection_worst_mae"] = ranking[
        [f"{fold}_mae_mm" for fold in ranking_folds]
    ].max(axis=1)
    ranking["_selection_mean_rmse"] = ranking[
        [f"{fold}_rmse_mm" for fold in ranking_folds]
    ].mean(axis=1)
    ranking["_selection_boundary"] = ranking[
        [
            f"{fold}_{metric}"
            for fold in ranking_folds
            for metric in ("f1_3mm", "f1_5mm")
        ]
    ].mean(axis=1)
    mae_ranked = ranking.sort_values(
        ["_selection_mean_mae", "_selection_worst_mae", "_selection_mean_rmse"],
        ascending=[True, True, True],
    ).head(count)
    boundary_count = max(5, count // 3)
    boundary_ranked = ranking.sort_values(
        ["_selection_boundary", "_selection_mean_mae", "_selection_worst_mae"],
        ascending=[False, True, True],
    ).head(boundary_count)
    profiles: list[PoolingProfile] = []
    seen: set[PoolingProfile] = set()
    for _, row in pd.concat([mae_ranked, boundary_ranked]).iterrows():
        profile = _profile_from_row(row)
        if profile not in seen:
            profiles.append(profile)
            seen.add(profile)
    return profiles


def _load_fold(path: Path) -> dict[str, dict]:
    frame = pd.read_csv(path, dtype={"study_id": str})
    rows: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        if str(row.error).strip() not in {"", "nan", "None"}:
            raise RuntimeError(f"Evaluation error for study {row.study_id}: {row.error}")
        rows[str(row.study_id)] = {
            "truth": float(row.gt_MLS_mm),
            "items": json.loads(row.slice_predictions_json),
        }
    return rows


def _profile_payload(prefix: str, profile: PoolingProfile) -> dict:
    return {f"{prefix}_{key}": value for key, value in asdict(profile).items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    for family in ("binary", "peak"):
        for fold in range(3):
            parser.add_argument(f"--{family}-fold{fold}", type=Path, required=True)
        parser.add_argument(f"--{family}-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=30)
    args = parser.parse_args()

    paths = {
        family: {
            f"fold{fold}": getattr(args, f"{family}_fold{fold}")
            for fold in range(3)
        }
        for family in ("binary", "peak")
    }
    raw = {
        family: {fold: _load_fold(path) for fold, path in fold_paths.items()}
        for family, fold_paths in paths.items()
    }
    fold_names = ["fold0", "fold1", "fold2"]
    study_ids: dict[str, list[str]] = {}
    truths: dict[str, np.ndarray] = {}
    for fold in fold_names:
        binary_ids = set(raw["binary"][fold])
        peak_ids = set(raw["peak"][fold])
        if binary_ids != peak_ids:
            raise RuntimeError(
                f"Study mismatch in {fold}: binary={len(binary_ids)} peak={len(peak_ids)}"
            )
        study_ids[fold] = sorted(binary_ids)
        truth = np.asarray([raw["binary"][fold][study]["truth"] for study in study_ids[fold]])
        peak_truth = np.asarray([raw["peak"][fold][study]["truth"] for study in study_ids[fold]])
        if not np.allclose(truth, peak_truth, atol=1e-6):
            raise RuntimeError(f"Ground-truth mismatch in {fold}")
        truths[fold] = truth

    ranking_contexts = {
        "all": fold_names,
        **{
            f"holdout_{held_out}": [fold for fold in fold_names if fold != held_out]
            for held_out in fold_names
        },
    }
    grid_paths = {"binary": args.binary_grid, "peak": args.peak_grid}
    profiles_by_context = {
        context: {
            family: _candidate_profiles(grid_paths[family], args.candidates, ranking_folds)
            for family in ("binary", "peak")
        }
        for context, ranking_folds in ranking_contexts.items()
    }
    cache_profiles: dict[str, list[PoolingProfile]] = {}
    for family in ("binary", "peak"):
        cache_profiles[family] = list(dict.fromkeys(
            profile
            for context in profiles_by_context.values()
            for profile in context[family]
        ))
    prediction_cache: dict[str, dict[PoolingProfile, dict[str, np.ndarray]]] = {
        "binary": {},
        "peak": {},
    }
    for family in ("binary", "peak"):
        for profile in cache_profiles[family]:
            prediction_cache[family][profile] = {
                fold: np.asarray([
                    _predict(raw[family][fold][study]["items"], profile)
                    for study in study_ids[fold]
                ])
                for fold in fold_names
            }

    rows: list[dict] = []
    blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    for binary_profile in profiles_by_context["all"]["binary"]:
        for peak_profile in profiles_by_context["all"]["peak"]:
            for peak_weight in blend_weights:
                row = {
                    **_profile_payload("binary", binary_profile),
                    **_profile_payload("peak", peak_profile),
                    "peak_weight": peak_weight,
                }
                for fold in fold_names:
                    binary_prediction = prediction_cache["binary"][binary_profile][fold]
                    peak_prediction = prediction_cache["peak"][peak_profile][fold]
                    prediction = (
                        (1.0 - peak_weight) * binary_prediction
                        + peak_weight * peak_prediction
                    )
                    for key, value in _metrics(truths[fold], prediction).items():
                        row[f"{fold}_{key}"] = value
                row["mean_mae_mm"] = float(np.mean([
                    row[f"{fold}_mae_mm"] for fold in fold_names
                ]))
                row["worst_mae_mm"] = max(row[f"{fold}_mae_mm"] for fold in fold_names)
                row["mean_rmse_mm"] = float(np.mean([
                    row[f"{fold}_rmse_mm"] for fold in fold_names
                ]))
                row["mean_boundary_f1"] = float(np.mean([
                    row[f"{fold}_{metric}"]
                    for fold in fold_names
                    for metric in ("f1_3mm", "f1_5mm")
                ]))
                rows.append(row)

    results = pd.DataFrame(rows)
    robust = results.sort_values(
        ["mean_mae_mm", "worst_mae_mm", "mean_rmse_mm"],
        ascending=[True, True, True],
    )
    boundary = results.sort_values(
        ["mean_boundary_f1", "mean_mae_mm", "worst_mae_mm"],
        ascending=[False, True, True],
    )
    leave_one_fold_out: dict[str, dict] = {}
    for held_out in fold_names:
        selection_folds = [fold for fold in fold_names if fold != held_out]
        context = profiles_by_context[f"holdout_{held_out}"]
        best_key: tuple[float, float, float] | None = None
        best_selection: dict | None = None
        for binary_profile in context["binary"]:
            for peak_profile in context["peak"]:
                for peak_weight in blend_weights:
                    fold_metrics: dict[str, dict[str, float]] = {}
                    for fold in selection_folds:
                        prediction = (
                            (1.0 - peak_weight)
                            * prediction_cache["binary"][binary_profile][fold]
                            + peak_weight * prediction_cache["peak"][peak_profile][fold]
                        )
                        fold_metrics[fold] = _metrics(truths[fold], prediction)
                    key = (
                        float(np.mean([fold_metrics[fold]["mae_mm"] for fold in selection_folds])),
                        max(fold_metrics[fold]["mae_mm"] for fold in selection_folds),
                        float(np.mean([fold_metrics[fold]["rmse_mm"] for fold in selection_folds])),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_selection = {
                            "binary_profile": binary_profile,
                            "peak_profile": peak_profile,
                            "peak_weight": peak_weight,
                        }
        assert best_key is not None and best_selection is not None
        binary_profile = best_selection["binary_profile"]
        peak_profile = best_selection["peak_profile"]
        peak_weight = best_selection["peak_weight"]
        held_out_prediction = (
            (1.0 - peak_weight) * prediction_cache["binary"][binary_profile][held_out]
            + peak_weight * prediction_cache["peak"][peak_profile][held_out]
        )
        held_out_metrics = _metrics(truths[held_out], held_out_prediction)
        specification = {
            **_profile_payload("binary", binary_profile),
            **_profile_payload("peak", peak_profile),
            "peak_weight": peak_weight,
        }
        leave_one_fold_out[held_out] = {
            "selected_using": selection_folds,
            "selection_mean_mae_mm": best_key[0],
            "held_out_mae_mm": held_out_metrics["mae_mm"],
            "held_out_rmse_mm": held_out_metrics["rmse_mm"],
            "held_out_boundary_f1": float(np.mean([
                held_out_metrics["f1_3mm"], held_out_metrics["f1_5mm"]
            ])),
            "specification": specification,
        }

    best_binary = profiles_by_context["all"]["binary"][0]
    best_peak = profiles_by_context["all"]["peak"][0]
    complementarity: dict[str, dict] = {}
    for fold in fold_names:
        binary_prediction = prediction_cache["binary"][best_binary][fold]
        peak_prediction = prediction_cache["peak"][best_peak][fold]
        binary_error = binary_prediction - truths[fold]
        peak_error = peak_prediction - truths[fold]
        complementarity[fold] = {
            "error_pearson": float(np.corrcoef(binary_error, peak_error)[0, 1]),
            "binary_mae_mm": float(np.mean(np.abs(binary_error))),
            "peak_mae_mm": float(np.mean(np.abs(peak_error))),
            "midpoint_mae_mm": float(np.mean(np.abs((binary_prediction + peak_prediction) / 2 - truths[fold]))),
            "oracle_min_error_mae_mm": float(np.mean(np.minimum(np.abs(binary_error), np.abs(peak_error)))),
        }

    payload = {
        "warning": (
            "The joint ranking is diagnostic. Leave-one-fold-out transfer is the "
            "less biased robustness check; the oracle metric is unattainable."
        ),
        "candidate_counts": {
            context: {family: len(value) for family, value in families.items()}
            for context, families in profiles_by_context.items()
        },
        "n_ensemble_candidates": int(len(results)),
        "complementarity_of_individual_robust_profiles": complementarity,
        "leave_one_fold_out_transfer": leave_one_fold_out,
        "best_robust_ensemble_all_available_folds": robust.iloc[0].to_dict(),
        "best_boundary_ensemble_all_available_folds": boundary.iloc[0].to_dict(),
        "top_robust_ensembles": robust.head(20).to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "crossfold_ensemble_grid.csv", index=False)
    (args.output_dir / "crossfold_ensemble_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
