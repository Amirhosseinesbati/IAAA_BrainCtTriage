"""Cross-fold search for robust MLS study pooling.

The script consumes already-computed CUDA slice predictions.  It performs no
model inference. Profiles are evaluated independently on two or three OOF
folds. With three folds the report also performs leave-one-fold-out transfer:
a profile is selected using the other folds and evaluated unchanged on the
held-out fold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


NEGATIVE_MLS_MM = 0.1


@dataclass(frozen=True)
class PoolingProfile:
    family: str
    size: int
    component_ratio: float
    selector_gate: float
    min_active_slices: int
    quantile: float
    probability_weighted: bool
    heatmap_guard_ratio: float


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 1e-8)
    cumulative = np.cumsum(sorted_weights)
    cutoff = q * cumulative[-1]
    return float(sorted_values[min(int(np.searchsorted(cumulative, cutoff)), len(values) - 1)])


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for active in mask:
        current = current + 1 if active else 0
        longest = max(longest, current)
    return longest


def _decode(frame: pd.DataFrame) -> list[list[dict]]:
    return [json.loads(payload) for payload in frame["slice_predictions_json"]]


def _python_scalar(value):
    """Convert pandas/numpy scalars before placing them in JSON payloads."""
    return value.item() if isinstance(value, np.generic) else value


def _study_features(items: list[dict]) -> dict[str, float]:
    probabilities = np.asarray([float(item["selector_probability"]) for item in items])
    heatmap = np.asarray([float(item["heatmap_peak"]) for item in items])
    top = np.sort(probabilities)[::-1]
    return {
        "selector_max": float(top[0]),
        "selector_top3_mean": float(np.mean(top[: min(3, len(top))])),
        "selector_count_05": int(np.sum(probabilities >= 0.5)),
        "selector_count_07": int(np.sum(probabilities >= 0.7)),
        "selector_longest_run_05": _longest_true_run(probabilities >= 0.5),
        "selector_longest_run_07": _longest_true_run(probabilities >= 0.7),
        "heatmap_top3_mean": float(np.mean(np.sort(heatmap)[-min(3, len(heatmap)) :])),
    }


def _predict(items: list[dict], profile: PoolingProfile) -> float:
    probabilities = np.asarray([float(item["selector_probability"]) for item in items])
    values = np.asarray([float(item["mls_mm"]) for item in items])
    heatmap = np.asarray([float(item["heatmap_peak"]) for item in items])
    if probabilities.max() < profile.selector_gate:
        return NEGATIVE_MLS_MM
    if np.sum(probabilities >= profile.selector_gate) < profile.min_active_slices:
        return NEGATIVE_MLS_MM

    if profile.family == "topk":
        scores = probabilities
        indices = np.argsort(-scores)[: profile.size]
    else:
        if profile.family == "relative_component":
            scores = probabilities
        elif profile.family == "smooth_component":
            scores = np.convolve(probabilities, np.asarray([0.25, 0.5, 0.25]), mode="same")
        elif profile.family == "joint_component":
            peak_scale = heatmap / max(float(heatmap.max()), 1e-8)
            scores = probabilities * np.sqrt(np.maximum(peak_scale, 0.0))
        elif profile.family == "anchor_window":
            scores = probabilities
        elif profile.family == "severity_window":
            # Anchor the anatomical neighbourhood on a slice that is jointly
            # plausible as a target, spatially confident, and locally severe.
            # The square roots prevent either noisy heatmap confidence or a
            # single extreme MLS estimate from dominating on its own.
            peak_scale = heatmap / max(float(heatmap.max()), 1e-8)
            clipped_values = np.clip(values, 0.0, 30.0)
            value_scale = clipped_values / max(float(clipped_values.max()), 1e-8)
            scores = probabilities * np.sqrt(np.maximum(peak_scale, 0.0)) * np.sqrt(
                np.maximum(value_scale, 0.0)
            )
        else:
            raise ValueError(profile.family)

        anchor = int(np.argmax(scores))
        if profile.family in {"anchor_window", "severity_window"}:
            indices = np.arange(
                max(0, anchor - profile.size),
                min(len(items), anchor + profile.size + 1),
            )
        else:
            active = scores >= scores[anchor] * profile.component_ratio
            left = anchor
            right = anchor
            while left > 0 and active[left - 1]:
                left -= 1
            while right + 1 < len(items) and active[right + 1]:
                right += 1
            indices = np.arange(left, right + 1)

    if profile.heatmap_guard_ratio > 0:
        selected_heatmap = heatmap[indices]
        guarded = indices[selected_heatmap >= selected_heatmap.max() * profile.heatmap_guard_ratio]
        if len(guarded):
            indices = guarded

    selected_values = values[indices]
    if profile.probability_weighted:
        return _weighted_quantile(selected_values, probabilities[indices], profile.quantile)
    return float(np.quantile(selected_values, profile.quantile))


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "bias_mm": float(np.mean(error)),
        "f1_3mm": float(f1_score(truth >= 3, prediction >= 3, zero_division=0)),
        "f1_5mm": float(f1_score(truth >= 5, prediction >= 5, zero_division=0)),
    }


def _profiles() -> list[PoolingProfile]:
    shapes: list[tuple[str, int, float]] = []
    shapes.extend(("topk", size, 0.0) for size in (3, 5, 7))
    shapes.extend(("anchor_window", radius, 0.0) for radius in (1, 2, 3))
    shapes.extend(("severity_window", radius, 0.0) for radius in (1, 2, 3))
    for family in ("relative_component", "smooth_component", "joint_component"):
        shapes.extend((family, 0, ratio) for ratio in (0.3, 0.5, 0.7))

    return [
        PoolingProfile(family, size, ratio, gate, min_active, quantile, weighted, heat_guard)
        for family, size, ratio in shapes
        # Peak-aware soft selector targets are intentionally less extreme than
        # binary targets, so their useful absolute gates can sit below 0.5.
        # Keeping the legacy high gates in the same grid lets this search serve
        # both selector variants without silently zeroing soft-model studies.
        for gate in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        for min_active in (1, 2, 3)
        for quantile in (0.5, 0.65, 0.75, 0.9)
        for weighted in (False, True)
        for heat_guard in (0.0, 0.5)
    ]


def _selector_summary(frame: pd.DataFrame, decoded: list[list[dict]]) -> dict:
    features = pd.DataFrame([_study_features(items) for items in decoded])
    positive = frame["gt_MLS_mm"].to_numpy(float) > NEGATIVE_MLS_MM + 1e-6
    summary: dict[str, object] = {
        "n_positive_studies": int(positive.sum()),
        "n_negative_studies": int((~positive).sum()),
        "features": {},
    }
    for column in features.columns:
        values = features[column].to_numpy(float)
        feature_payload = {
            "positive_median": float(np.median(values[positive])),
            "negative_median": float(np.median(values[~positive])),
            "positive_q25": float(np.quantile(values[positive], 0.25)),
            "negative_q75": float(np.quantile(values[~positive], 0.75)),
        }
        if len(np.unique(positive)) == 2 and len(np.unique(values)) > 1:
            auc = float(roc_auc_score(positive, values))
            feature_payload["separation_auc"] = max(auc, 1.0 - auc)
        summary["features"][column] = feature_payload
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_paths = {
        "fold0": pd.read_csv(args.fold0, dtype={"study_id": str}),
        "fold1": pd.read_csv(args.fold1, dtype={"study_id": str}),
    }
    if args.fold2 is not None:
        input_paths["fold2"] = pd.read_csv(args.fold2, dtype={"study_id": str})
    frames = input_paths
    fold_names = list(frames)
    decoded = {name: _decode(frame) for name, frame in frames.items()}
    truths = {name: frame["gt_MLS_mm"].to_numpy(float) for name, frame in frames.items()}

    rows: list[dict] = []
    for profile in _profiles():
        row = asdict(profile)
        for name in fold_names:
            prediction = np.asarray([_predict(items, profile) for items in decoded[name]])
            for key, value in _metrics(truths[name], prediction).items():
                row[f"{name}_{key}"] = value
        row["mean_mae_mm"] = float(np.mean([row[f"{name}_mae_mm"] for name in fold_names]))
        row["worst_mae_mm"] = max(row[f"{name}_mae_mm"] for name in fold_names)
        row["mean_rmse_mm"] = float(np.mean([row[f"{name}_rmse_mm"] for name in fold_names]))
        row["mean_boundary_f1"] = float(np.mean([
            row[f"{name}_{metric}"]
            for name in fold_names
            for metric in ("f1_3mm", "f1_5mm")
        ]))
        rows.append(row)

    results = pd.DataFrame(rows)
    robust = results.sort_values(
        ["mean_mae_mm", "worst_mae_mm", "mean_rmse_mm"], ascending=[True, True, True]
    )
    boundary_robust = results.sort_values(
        ["mean_boundary_f1", "mean_mae_mm", "worst_mae_mm"],
        ascending=[False, True, True],
    )
    fold0_best = results.sort_values(["fold0_mae_mm", "fold0_rmse_mm"]).iloc[0]
    fold1_best = results.sort_values(["fold1_mae_mm", "fold1_rmse_mm"]).iloc[0]
    profile_columns = list(asdict(_profiles()[0]).keys())

    leave_one_fold_out: dict[str, dict] = {}
    leave_one_fold_out_boundary: dict[str, dict] = {}
    if len(fold_names) >= 3:
        for held_out in fold_names:
            train_folds = [name for name in fold_names if name != held_out]
            train_score = results[[f"{name}_mae_mm" for name in train_folds]].mean(axis=1)
            selected = results.loc[train_score.sort_values().index[0]]
            leave_one_fold_out[held_out] = {
                "selected_using": train_folds,
                "selection_mean_mae_mm": float(train_score.loc[selected.name]),
                "held_out_mae_mm": float(selected[f"{held_out}_mae_mm"]),
                "held_out_rmse_mm": float(selected[f"{held_out}_rmse_mm"]),
                "profile": {key: _python_scalar(selected[key]) for key in profile_columns},
            }
            train_boundary_score = results[[
                f"{name}_{metric}"
                for name in train_folds
                for metric in ("f1_3mm", "f1_5mm")
            ]].mean(axis=1)
            boundary_selected = results.loc[train_boundary_score.sort_values(ascending=False).index[0]]
            leave_one_fold_out_boundary[held_out] = {
                "selected_using": train_folds,
                "selection_mean_boundary_f1": float(train_boundary_score.loc[boundary_selected.name]),
                "held_out_boundary_f1": float(np.mean([
                    boundary_selected[f"{held_out}_f1_3mm"],
                    boundary_selected[f"{held_out}_f1_5mm"],
                ])),
                "held_out_mae_mm": float(boundary_selected[f"{held_out}_mae_mm"]),
                "profile": {
                    key: _python_scalar(boundary_selected[key]) for key in profile_columns
                },
            }

    payload = {
        "warning": (
            "Joint ranking is diagnostic and is not an untouched test estimate. "
            "Use leave-one-fold-out transfer to judge robustness."
        ),
        "fold_names": fold_names,
        "n_profiles": int(len(results)),
        "selector_feature_summary": {
            name: _selector_summary(frames[name], decoded[name]) for name in frames
        },
        "best_robust_profiles": robust.head(20).to_dict(orient="records"),
        "one_way_transfer": {
            "selected_on_fold0_evaluated_on_fold1": fold0_best.to_dict(),
            "selected_on_fold1_evaluated_on_fold0": fold1_best.to_dict(),
        },
        "leave_one_fold_out_transfer": leave_one_fold_out,
        "leave_one_fold_out_boundary_transfer": leave_one_fold_out_boundary,
        "frozen_candidate_for_fold2": robust.iloc[0][profile_columns].to_dict(),
        "best_robust_profile_all_available_folds": robust.iloc[0].to_dict(),
        "best_boundary_profile_all_available_folds": boundary_robust.iloc[0].to_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "crossfold_pooling_grid.csv", index=False)
    (args.output_dir / "crossfold_pooling_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
