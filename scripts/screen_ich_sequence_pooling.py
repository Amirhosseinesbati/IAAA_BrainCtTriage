"""Cross-fitted screen of fixed study pooling rules on ICH OOF slice scores."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
)


VOLUME_BY_LABEL = dict(zip(OUTPUT_LABELS[1:], VOLUME_KEYS, strict=True))
PRIMARY_METHOD = "max_pair_equal_blend"


def _max(values: np.ndarray) -> float:
    return float(values.max())


def _top_two_mean(values: np.ndarray) -> float:
    return float(np.sort(values)[-min(2, len(values)):].mean())


def _adjacent_pair_mean_max(values: np.ndarray) -> float:
    if len(values) < 2:
        return _max(values)
    return float(np.max((values[:-1] + values[1:]) / 2.0))


def _adjacent_pair_geomean_max(values: np.ndarray) -> float:
    if len(values) < 2:
        return _max(values)
    return float(np.max(np.sqrt(np.clip(values[:-1] * values[1:], 0.0, 1.0))))


def _adjacent_triple_mean_max(values: np.ndarray) -> float:
    if len(values) < 3:
        return _adjacent_pair_mean_max(values)
    return float(np.max((values[:-2] + values[1:-1] + values[2:]) / 3.0))


def _max_pair_equal_blend(values: np.ndarray) -> float:
    return 0.5 * (_max(values) + _adjacent_pair_mean_max(values))


POOLERS: OrderedDict[str, Callable[[np.ndarray], float]] = OrderedDict([
    ("max", _max),
    ("top_two_mean", _top_two_mean),
    ("adjacent_pair_mean_max", _adjacent_pair_mean_max),
    ("adjacent_pair_geomean_max", _adjacent_pair_geomean_max),
    ("adjacent_triple_mean_max", _adjacent_triple_mean_max),
    (PRIMARY_METHOD, _max_pair_equal_blend),
])


def _safe_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(truth)) < 2:
        return None
    return float(roc_auc_score(truth, scores))


def evaluation_summary(payload: dict[str, object]) -> dict[str, object]:
    """Resolve a direct summary or the stricter summary from a rescore audit."""
    rescored = payload.get("rescored_summary")
    if isinstance(rescored, dict) and "selection_score" in rescored:
        return rescored
    if "selection_score" in payload:
        return payload
    raise ValueError("Baseline JSON does not contain a usable selection summary")


def pool_slice_scores(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"study_id", "slice_index", "outer_fold", *{
        f"prob_{label}" for label in OUTPUT_LABELS
    }}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"OOF slice predictions are missing: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for study_id, group in frame.sort_values(
        ["study_id", "slice_index"]
    ).groupby("study_id", sort=True):
        folds = group["outer_fold"].astype(int).unique()
        if len(folds) != 1:
            raise ValueError(f"Study {study_id} occurs in multiple OOF folds")
        row: dict[str, object] = {
            "study_id": str(study_id),
            "outer_fold": int(folds[0]),
        }
        for label in OUTPUT_LABELS:
            values = group[f"prob_{label}"].to_numpy(dtype=np.float64)
            for method, pooler in POOLERS.items():
                row[f"score_{label}_{method}"] = pooler(values)
        rows.append(row)
    return pd.DataFrame(rows)


def add_truth(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    required = {"study_id", *{f"gt_{key}" for key in VOLUME_KEYS}}
    missing = required - set(truth)
    if missing:
        raise ValueError(f"ICH truth is missing: {sorted(missing)}")
    merged = scores.merge(
        truth.loc[:, sorted(required)],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(scores):
        raise ValueError("OOF scores and ICH ground truth do not match")
    merged["truth_any_ich"] = (
        merged[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1) > 0
    ).astype(np.uint8)
    for label, volume_key in VOLUME_BY_LABEL.items():
        merged[f"truth_{label}"] = (
            merged[f"gt_{volume_key}"] > 0
        ).astype(np.uint8)
    return merged


def method_metrics(frame: pd.DataFrame, method: str) -> dict[str, object]:
    any_auc = _safe_auc(
        frame["truth_any_ich"].to_numpy(dtype=np.uint8),
        frame[f"score_any_ich_{method}"].to_numpy(dtype=np.float64),
    )
    subtype_auc = {
        label: _safe_auc(
            frame[f"truth_{label}"].to_numpy(dtype=np.uint8),
            frame[f"score_{label}_{method}"].to_numpy(dtype=np.float64),
        )
        for label in OUTPUT_LABELS[1:]
    }
    available = [value for value in subtype_auc.values() if value is not None]
    return {
        "any_ich_auc": any_auc,
        "subtype_auc": subtype_auc,
        "macro_subtype_auc": float(np.mean(available)) if available else None,
    }


def _choose_method(frame: pd.DataFrame, label: str) -> str:
    truth_column = f"truth_{label}"
    truth = frame[truth_column].to_numpy(dtype=np.uint8)
    candidates: list[tuple[float, int, str]] = []
    for index, method in enumerate(POOLERS):
        auc = _safe_auc(
            truth, frame[f"score_{label}_{method}"].to_numpy(dtype=np.float64)
        )
        candidates.append((float(auc or 0.0), -index, method))
    return max(candidates)[2]


def crossfit_method_selection(frame: pd.DataFrame) -> dict[str, object]:
    output = frame.loc[:, ["study_id", "outer_fold", "truth_any_ich", *{
        f"truth_{label}" for label in OUTPUT_LABELS[1:]
    }]].copy()
    selections: list[dict[str, object]] = []
    for heldout_fold in sorted(frame["outer_fold"].astype(int).unique()):
        development = frame.loc[frame["outer_fold"] != heldout_fold]
        heldout = frame.loc[frame["outer_fold"] == heldout_fold]
        selected: dict[str, object] = {"heldout_fold": int(heldout_fold)}
        for label in OUTPUT_LABELS:
            method = _choose_method(development, label)
            selected[label] = method
            output.loc[
                output["outer_fold"] == heldout_fold, f"crossfit_score_{label}"
            ] = heldout[f"score_{label}_{method}"].to_numpy(dtype=np.float64)
        selections.append(selected)
    any_auc = _safe_auc(
        output["truth_any_ich"].to_numpy(dtype=np.uint8),
        output["crossfit_score_any_ich"].to_numpy(dtype=np.float64),
    )
    subtype_auc = {
        label: _safe_auc(
            output[f"truth_{label}"].to_numpy(dtype=np.uint8),
            output[f"crossfit_score_{label}"].to_numpy(dtype=np.float64),
        )
        for label in OUTPUT_LABELS[1:]
    }
    return {
        "protocol": "select_pooler_on_four_oof_folds_apply_to_heldout_oof_fold",
        "fold_selections": selections,
        "any_ich_auc": any_auc,
        "subtype_auc": subtype_auc,
        "macro_subtype_auc": float(
            np.mean([value for value in subtype_auc.values() if value is not None])
        ),
    }


def bootstrap_primary_delta(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    any_deltas: list[float] = []
    macro_deltas: list[float] = []
    for _ in range(samples):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        reference = method_metrics(sampled, "max")
        candidate = method_metrics(sampled, PRIMARY_METHOD)
        if reference["any_ich_auc"] is None or candidate["any_ich_auc"] is None:
            continue
        reference_subtypes = reference["subtype_auc"]
        candidate_subtypes = candidate["subtype_auc"]
        if any(
            reference_subtypes[label] is None or candidate_subtypes[label] is None
            for label in OUTPUT_LABELS[1:]
        ):
            continue
        any_deltas.append(
            float(candidate["any_ich_auc"]) - float(reference["any_ich_auc"])
        )
        macro_deltas.append(
            float(candidate["macro_subtype_auc"])
            - float(reference["macro_subtype_auc"])
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
        "any_ich_auc_delta": summarize(any_deltas),
        "macro_subtype_auc_delta": summarize(macro_deltas),
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
    scores = add_truth(pool_slice_scores(slices), truth)
    methods = {method: method_metrics(scores, method) for method in POOLERS}
    baseline = methods["max"]
    primary = methods[PRIMARY_METHOD]
    baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    baseline_summary = evaluation_summary(baseline_payload)
    baseline_selection = float(baseline_summary["selection_score"])
    primary_selection_proxy = baseline_selection + (
        0.30 * (float(primary["any_ich_auc"]) - float(baseline["any_ich_auc"]))
        + 0.15
        * (
            float(primary["macro_subtype_auc"])
            - float(baseline["macro_subtype_auc"])
        )
    )
    foldwise = {
        str(fold): {
            method: method_metrics(scores.loc[scores["outer_fold"] == fold], method)
            for method in ("max", PRIMARY_METHOD)
        }
        for fold in sorted(scores["outer_fold"].astype(int).unique())
    }
    result: dict[str, object] = {
        "analysis_kind": "ich_oof_fixed_sequence_pooling_screen",
        "primary_method": PRIMARY_METHOD,
        "primary_method_preregistered_reason": (
            "equal blend retains singleton max sensitivity while rewarding an adjacent pair"
        ),
        "studies": int(len(scores)),
        "oof_folds": sorted(scores["outer_fold"].astype(int).unique().tolist()),
        "methods": methods,
        "foldwise_primary_vs_max": foldwise,
        "primary_delta": {
            "any_ich_auc": float(primary["any_ich_auc"])
            - float(baseline["any_ich_auc"]),
            "macro_subtype_auc": float(primary["macro_subtype_auc"])
            - float(baseline["macro_subtype_auc"]),
            "selection_proxy": primary_selection_proxy - baseline_selection,
        },
        "baseline_selection_score_unchanged_spatial": baseline_selection,
        "primary_selection_proxy_unchanged_spatial": primary_selection_proxy,
        "crossfit_exploratory_method_selection": crossfit_method_selection(scores),
        "bootstrap_primary_vs_max": bootstrap_primary_delta(
            scores, samples=args.bootstrap_samples, seed=args.seed
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
            "stage": "oof_sequence_pooling_screen",
            "evaluation_scope": "ich_only_oof_no_new_inference",
            "primary_method": PRIMARY_METHOD,
        })
        mlflow.log_params({
            "primary_method": PRIMARY_METHOD,
            "methods": json.dumps(list(POOLERS)),
            "studies": len(scores),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "oof_slice_predictions_sha256": result["oof_slice_predictions_sha256"],
            "git_commit": result["git_commit"],
        })
        mlflow.log_metrics({
            "baseline_any_auc": float(baseline["any_ich_auc"]),
            "primary_any_auc": float(primary["any_ich_auc"]),
            "delta_any_auc": float(result["primary_delta"]["any_ich_auc"]),
            "baseline_macro_auc": float(baseline["macro_subtype_auc"]),
            "primary_macro_auc": float(primary["macro_subtype_auc"]),
            "delta_macro_auc": float(result["primary_delta"]["macro_subtype_auc"]),
            "delta_selection_proxy": float(result["primary_delta"]["selection_proxy"]),
        })
        mlflow.log_artifact(str(args.output), artifact_path="oof_sequence_pooling")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
