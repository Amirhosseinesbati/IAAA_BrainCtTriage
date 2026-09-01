"""Leakage-safe cross-fitted logistic meta-head on ICH OOF slice sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.screen_ich_sequence_pooling import (
    add_truth,
    evaluation_summary,
    method_metrics,
    pool_slice_scores,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_v2.evaluation import ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
)


META_C = 0.1
META_CLASS_WEIGHT = "balanced"
META_SOLVER = "lbfgs"
META_MAX_ITER = 2000
META_TOL = 1e-7
FEATURE_POOLERS = (
    "max",
    "top_two_mean",
    "adjacent_pair_mean_max",
    "adjacent_pair_geomean_max",
    "adjacent_triple_mean_max",
)
GATE = {
    "minimum_selection_proxy_delta": 0.002,
    "minimum_macro_subtype_auc_delta": 0.005,
    "minimum_any_ich_auc_delta": -0.002,
    "minimum_nonnegative_fold_proxy_count": 3,
    "minimum_subtype_auc_delta": -0.02,
    "minimum_bootstrap_selection_probability_positive": 0.90,
    "minimum_bootstrap_macro_probability_positive": 0.90,
}


def feature_columns(label: str) -> tuple[str, ...]:
    return (
        *(f"score_{label}_{method}" for method in FEATURE_POOLERS),
        f"feature_{label}_sequence_mean",
        f"feature_{label}_sequence_std",
        f"feature_{label}_log_slice_count",
    )


def sequence_feature_frame(slice_frame: pd.DataFrame) -> pd.DataFrame:
    """Build fixed threshold-free sequence features for each study and label."""
    pooled = pool_slice_scores(slice_frame)
    rows: list[dict[str, object]] = []
    for study_id, group in slice_frame.sort_values(
        ["study_id", "slice_index"]
    ).groupby("study_id", sort=True):
        row: dict[str, object] = {"study_id": str(study_id)}
        log_count = float(np.log1p(len(group)))
        for label in OUTPUT_LABELS:
            values = group[f"prob_{label}"].to_numpy(dtype=np.float64)
            row[f"feature_{label}_sequence_mean"] = float(values.mean())
            row[f"feature_{label}_sequence_std"] = float(values.std(ddof=0))
            row[f"feature_{label}_log_slice_count"] = log_count
        rows.append(row)
    features = pooled.merge(
        pd.DataFrame(rows), on="study_id", how="inner", validate="one_to_one"
    )
    if len(features) != slice_frame["study_id"].astype(str).nunique():
        raise ValueError("Sequence features do not cover every OOF study exactly once")
    return features


def _safe_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(truth)) < 2:
        return None
    return float(roc_auc_score(truth, scores))


def _meta_metrics(frame: pd.DataFrame) -> dict[str, object]:
    any_auc = _safe_auc(
        frame["truth_any_ich"].to_numpy(dtype=np.uint8),
        frame["meta_score_any_ich"].to_numpy(dtype=np.float64),
    )
    subtype_auc = {
        label: _safe_auc(
            frame[f"truth_{label}"].to_numpy(dtype=np.uint8),
            frame[f"meta_score_{label}"].to_numpy(dtype=np.float64),
        )
        for label in OUTPUT_LABELS[1:]
    }
    available = [value for value in subtype_auc.values() if value is not None]
    return {
        "any_ich_auc": any_auc,
        "subtype_auc": subtype_auc,
        "macro_subtype_auc": float(np.mean(available)) if available else None,
    }


def _serialized_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    columns: tuple[str, ...],
) -> dict[str, object]:
    standardized = model.coef_[0].astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    mean = scaler.mean_.astype(np.float64)
    raw_coefficients = standardized / scale
    raw_intercept = float(model.intercept_[0] - np.dot(standardized, mean / scale))
    return {
        "feature_columns": list(columns),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "standardized_coefficients": standardized.tolist(),
        "standardized_intercept": float(model.intercept_[0]),
        "raw_coefficients": raw_coefficients.tolist(),
        "raw_intercept": raw_intercept,
        "iterations": int(model.n_iter_[0]),
    }


def _fit_label_model(
    development: pd.DataFrame,
    label: str,
) -> tuple[StandardScaler, LogisticRegression, tuple[str, ...]]:
    columns = feature_columns(label)
    truth_column = "truth_any_ich" if label == "any_ich" else f"truth_{label}"
    truth = development[truth_column].to_numpy(dtype=np.uint8)
    if len(np.unique(truth)) != 2:
        raise ValueError(f"Development data for {label} does not contain both classes")
    scaler = StandardScaler()
    features = scaler.fit_transform(
        development.loc[:, columns].to_numpy(dtype=np.float64)
    )
    model = LogisticRegression(
        C=META_C,
        class_weight=META_CLASS_WEIGHT,
        solver=META_SOLVER,
        l1_ratio=0.0,
        max_iter=META_MAX_ITER,
        tol=META_TOL,
    )
    model.fit(features, truth)
    return scaler, model, columns


def crossfit_sequence_meta_head(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Fit on four OOF folds and score the untouched fifth fold for every label."""
    output = frame.copy()
    fold_models: list[dict[str, object]] = []
    folds = sorted(frame["outer_fold"].astype(int).unique())
    if len(folds) < 2:
        raise ValueError("At least two outer folds are required for cross-fitting")
    for heldout_fold in folds:
        development = frame.loc[frame["outer_fold"] != heldout_fold]
        heldout_mask = frame["outer_fold"] == heldout_fold
        heldout = frame.loc[heldout_mask]
        fold_payload: dict[str, object] = {
            "heldout_fold": int(heldout_fold),
            "development_folds": sorted(
                development["outer_fold"].astype(int).unique().tolist()
            ),
            "development_studies": int(len(development)),
            "heldout_studies": int(len(heldout)),
            "labels": {},
        }
        for label in OUTPUT_LABELS:
            scaler, model, columns = _fit_label_model(development, label)
            heldout_features = scaler.transform(
                heldout.loc[:, columns].to_numpy(dtype=np.float64)
            )
            output.loc[heldout_mask, f"meta_score_{label}"] = model.predict_proba(
                heldout_features
            )[:, 1]
            truth_column = (
                "truth_any_ich" if label == "any_ich" else f"truth_{label}"
            )
            label_payload = _serialized_model(scaler, model, columns)
            label_payload.update({
                "development_positives": int(development[truth_column].sum()),
                "heldout_positives": int(heldout[truth_column].sum()),
            })
            fold_payload["labels"][label] = label_payload
        fold_models.append(fold_payload)

    expected = [f"meta_score_{label}" for label in OUTPUT_LABELS]
    if output.loc[:, expected].isna().any().any():
        raise RuntimeError("Cross-fitted meta-head left missing scores")

    final_models: dict[str, object] = {}
    for label in OUTPUT_LABELS:
        scaler, model, columns = _fit_label_model(frame, label)
        payload = _serialized_model(scaler, model, columns)
        truth_column = "truth_any_ich" if label == "any_ich" else f"truth_{label}"
        payload["training_positives"] = int(frame[truth_column].sum())
        payload["training_studies"] = int(len(frame))
        final_models[label] = payload
    return output, fold_models, final_models


def paired_bootstrap_delta(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        "any_ich_auc": [],
        "macro_subtype_auc": [],
        "selection_proxy": [],
        **{f"subtype_auc_{label}": [] for label in OUTPUT_LABELS[1:]},
    }
    for _ in range(samples):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        baseline = method_metrics(sampled, "max")
        candidate = _meta_metrics(sampled)
        if baseline["any_ich_auc"] is None or candidate["any_ich_auc"] is None:
            continue
        if any(
            baseline["subtype_auc"][label] is None
            or candidate["subtype_auc"][label] is None
            for label in OUTPUT_LABELS[1:]
        ):
            continue
        any_delta = float(candidate["any_ich_auc"]) - float(
            baseline["any_ich_auc"]
        )
        macro_delta = float(candidate["macro_subtype_auc"]) - float(
            baseline["macro_subtype_auc"]
        )
        deltas["any_ich_auc"].append(any_delta)
        deltas["macro_subtype_auc"].append(macro_delta)
        deltas["selection_proxy"].append(0.30 * any_delta + 0.15 * macro_delta)
        for label in OUTPUT_LABELS[1:]:
            deltas[f"subtype_auc_{label}"].append(
                float(candidate["subtype_auc"][label])
                - float(baseline["subtype_auc"][label])
            )

    def summarize(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "samples": int(len(array)),
            "mean": float(array.mean()),
            "q025": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "q975": float(np.quantile(array, 0.975)),
            "probability_positive": float(np.mean(array > 0)),
        }

    return {
        "resampling_unit": "study",
        "requested_samples": samples,
        "seed": seed,
        "deltas": {key: summarize(values) for key, values in deltas.items()},
    }


def promotion_decision(
    delta: dict[str, object],
    foldwise: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    subtype_deltas = delta["subtype_auc"]
    nonnegative_folds = sum(
        float(payload["delta"]["selection_proxy"]) >= 0.0
        for payload in foldwise.values()
    )
    checks = {
        "selection_proxy": float(delta["selection_proxy"])
        >= GATE["minimum_selection_proxy_delta"],
        "macro_subtype_auc": float(delta["macro_subtype_auc"])
        >= GATE["minimum_macro_subtype_auc_delta"],
        "any_ich_auc_noninferiority": float(delta["any_ich_auc"])
        >= GATE["minimum_any_ich_auc_delta"],
        "fold_stability": nonnegative_folds
        >= GATE["minimum_nonnegative_fold_proxy_count"],
        "subtype_safety": min(float(value) for value in subtype_deltas.values())
        >= GATE["minimum_subtype_auc_delta"],
        "bootstrap_selection": float(
            bootstrap["deltas"]["selection_proxy"]["probability_positive"]
        )
        >= GATE["minimum_bootstrap_selection_probability_positive"],
        "bootstrap_macro": float(
            bootstrap["deltas"]["macro_subtype_auc"]["probability_positive"]
        )
        >= GATE["minimum_bootstrap_macro_probability_positive"],
    }
    return {
        "criteria": GATE,
        "checks": checks,
        "nonnegative_fold_proxy_count": int(nonnegative_folds),
        "promotion_allowed": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-slice-predictions", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    slices = pd.read_csv(
        args.oof_slice_predictions,
        dtype={"study_id": str, "patient_id": str},
    )
    truth, truth_source = ground_truth_ich_context()
    frame = add_truth(sequence_feature_frame(slices), truth)
    scored, fold_models, final_models = crossfit_sequence_meta_head(frame)
    baseline = method_metrics(scored, "max")
    candidate = _meta_metrics(scored)
    any_delta = float(candidate["any_ich_auc"]) - float(baseline["any_ich_auc"])
    macro_delta = float(candidate["macro_subtype_auc"]) - float(
        baseline["macro_subtype_auc"]
    )
    delta: dict[str, object] = {
        "any_ich_auc": any_delta,
        "macro_subtype_auc": macro_delta,
        "selection_proxy": 0.30 * any_delta + 0.15 * macro_delta,
        "subtype_auc": {
            label: float(candidate["subtype_auc"][label])
            - float(baseline["subtype_auc"][label])
            for label in OUTPUT_LABELS[1:]
        },
    }
    foldwise: dict[str, object] = {}
    for fold in sorted(scored["outer_fold"].astype(int).unique()):
        subset = scored.loc[scored["outer_fold"] == fold]
        fold_baseline = method_metrics(subset, "max")
        fold_candidate = _meta_metrics(subset)
        fold_any_delta = (
            None
            if fold_baseline["any_ich_auc"] is None
            or fold_candidate["any_ich_auc"] is None
            else float(fold_candidate["any_ich_auc"])
            - float(fold_baseline["any_ich_auc"])
        )
        fold_macro_delta = float(fold_candidate["macro_subtype_auc"]) - float(
            fold_baseline["macro_subtype_auc"]
        )
        foldwise[str(fold)] = {
            "baseline": fold_baseline,
            "candidate": fold_candidate,
            "delta": {
                "any_ich_auc": fold_any_delta,
                "macro_subtype_auc": fold_macro_delta,
                "selection_proxy": 0.30 * float(fold_any_delta or 0.0)
                + 0.15 * fold_macro_delta,
            },
        }

    bootstrap = paired_bootstrap_delta(
        scored, samples=args.bootstrap_samples, seed=args.seed
    )
    decision = promotion_decision(delta, foldwise, bootstrap)
    baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    baseline_selection = float(evaluation_summary(baseline_payload)["selection_score"])
    result: dict[str, object] = {
        "analysis_kind": "ich_oof_crossfitted_logistic_sequence_meta_head",
        "protocol": (
            "base slice scores are OOF; each meta prediction is fit on four outer "
            "folds and applied to the untouched fifth fold"
        ),
        "features": {
            "poolers": list(FEATURE_POOLERS),
            "additional": ["sequence_mean", "sequence_std", "log_slice_count"],
            "features_per_label": len(feature_columns("any_ich")),
            "cross_label_features": False,
            "threshold_features": False,
        },
        "model": {
            "kind": "standard_scaler_plus_l2_logistic_regression",
            "C": META_C,
            "class_weight": META_CLASS_WEIGHT,
            "solver": META_SOLVER,
            "max_iter": META_MAX_ITER,
            "tol": META_TOL,
        },
        "studies": int(len(scored)),
        "oof_folds": sorted(scored["outer_fold"].astype(int).unique().tolist()),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "baseline_selection_score_unchanged_spatial": baseline_selection,
        "candidate_selection_proxy_unchanged_spatial": baseline_selection
        + float(delta["selection_proxy"]),
        "foldwise": foldwise,
        "bootstrap": bootstrap,
        "promotion_decision": decision,
        "fold_models": fold_models,
        "final_all_oof_models_research_only_until_promotion": final_models,
        "deployment_semantics_if_promoted": (
            "apply the all-OOF head independently to each base fold model sequence, "
            "then average study scores; do not fit or select on leaderboard data"
        ),
        "oof_slice_predictions_sha256": file_sha256(args.oof_slice_predictions),
        "baseline_summary_sha256": file_sha256(args.baseline_summary),
        "truth_source": str(truth_source),
        "git_commit": git_commit(),
        "row_scores_persisted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({
            "task": "ich_segmentation_volume",
            "stage": "oof_crossfitted_sequence_meta_head",
            "evaluation_scope": "ich_only_oof_no_new_inference",
            "promotion_allowed": str(decision["promotion_allowed"]).lower(),
        })
        mlflow.log_params({
            "meta_C": META_C,
            "class_weight": META_CLASS_WEIGHT,
            "solver": META_SOLVER,
            "features_per_label": len(feature_columns("any_ich")),
            "cross_label_features": False,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "oof_slice_predictions_sha256": result["oof_slice_predictions_sha256"],
            "git_commit": result["git_commit"],
        })
        mlflow.log_metrics({
            "baseline_any_auc": float(baseline["any_ich_auc"]),
            "candidate_any_auc": float(candidate["any_ich_auc"]),
            "delta_any_auc": any_delta,
            "baseline_macro_auc": float(baseline["macro_subtype_auc"]),
            "candidate_macro_auc": float(candidate["macro_subtype_auc"]),
            "delta_macro_auc": macro_delta,
            "delta_selection_proxy": float(delta["selection_proxy"]),
            "nonnegative_fold_proxy_count": float(
                decision["nonnegative_fold_proxy_count"]
            ),
            "promotion_allowed": float(decision["promotion_allowed"]),
        })
        mlflow.log_artifact(str(args.output), artifact_path="oof_sequence_meta_head")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
