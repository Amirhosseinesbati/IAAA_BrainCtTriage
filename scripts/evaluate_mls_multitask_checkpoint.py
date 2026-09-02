"""Strict fold-level, CUDA-only end-to-end evaluation of MLS v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import load_fold_manifest
from src.evaluation.triage import triage_from_intermediates
from src.config import config_section
from src.mlops.tracking import configure_tracking_environment
from src.strategies.mls_heatmap.predict_multitask import (
    aggregate_study_mls,
    load_multitask_model,
    predict_study_slices,
)

VOLUME_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = [
        f1_score(truth == label, prediction == label, zero_division=0)
        for label in (0, 1, 2)
    ]
    return float(np.mean(scores))


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_mm": float(np.mean(np.abs(prediction - truth))),
        "rmse_mm": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "bias_mm": float(np.mean(prediction - truth)),
        "f1_3mm": float(f1_score(truth >= 3, prediction >= 3, zero_division=0)),
        "f1_5mm": float(f1_score(truth >= 5, prediction >= 5, zero_division=0)),
    }


def _combined_macro(frame: pd.DataFrame, predictions: np.ndarray) -> float | None:
    reference_path = PROJECT_ROOT / "reports" / "checkpoint_evaluation" / "fold_0_predictions.csv"
    if not reference_path.is_file():
        return None
    reference = pd.read_csv(reference_path, dtype={"study_id": str})
    joined = frame[["study_id", "gt_MLS_mm"]].copy()
    joined["candidate_MLS_mm"] = predictions
    joined = joined.merge(reference, on="study_id", how="inner", validate="one_to_one")
    if len(joined) != len(frame):
        return None
    truth_labels: list[int] = []
    predicted_labels: list[int] = []
    for _, row in joined.iterrows():
        truth_values = {key: float(row[f"gt_{key}"]) for key in VOLUME_KEYS}
        truth_values.update({
            "fracture_prob": float(row["gt_fracture_prob"]),
            "MLS_mm": float(row["gt_MLS_mm_y"]),
        })
        predicted_values = {key: float(row[f"pred_{key}"]) for key in VOLUME_KEYS}
        predicted_values.update({
            "fracture_prob": float(row["pred_fracture_prob"]),
            "MLS_mm": float(row["candidate_MLS_mm"]),
        })
        truth_labels.append(triage_from_intermediates(truth_values))
        predicted_labels.append(triage_from_intermediates(predicted_values))
    return _macro_f1(np.asarray(truth_labels), np.asarray(predicted_labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlflow-run-id", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS evaluation requested, but CUDA is unavailable")
    device = torch.device("cuda:0")
    model, config = load_multitask_model(args.checkpoint, device)

    folds = load_fold_manifest()
    manifest = folds.loc[folds["fold"] == args.fold, ["study_id", "patient_id", "triage_class"]].copy()
    truth = pd.read_csv(
        PROJECT_ROOT / "reports" / "eda" / "deep" / "deep_series_table.csv",
        dtype={"dicom_series.id": str},
    )[["dicom_series.id", "MLS_mm"]].rename(
        columns={"dicom_series.id": "study_id", "MLS_mm": "gt_MLS_mm"}
    )
    frame = manifest.merge(truth, on="study_id", how="left", validate="one_to_one")
    output_csv = args.output_dir / "study_slice_predictions.csv"
    if output_csv.is_file():
        previous = pd.read_csv(output_csv, dtype={"study_id": str})
        frame = frame.merge(
            previous[["study_id", "slice_predictions_json", "runtime_s", "error"]],
            on="study_id", how="left", validate="one_to_one",
        )
    else:
        frame["slice_predictions_json"] = ""
        frame["runtime_s"] = np.nan
        frame["error"] = ""

    data_root = PROJECT_ROOT / "Data" / "raw" / "training"
    for index, row in frame.iterrows():
        existing = str(row.get("slice_predictions_json", "") or "")
        if existing and existing != "nan":
            continue
        started = time.perf_counter()
        try:
            slices = predict_study_slices(
                data_root / str(row["study_id"]), model, config, device,
                batch_size=args.batch_size,
            )
            frame.at[index, "slice_predictions_json"] = json.dumps([
                {
                    "index": item.index,
                    "selector_probability": item.selector_probability,
                    "peak_probability": item.peak_probability,
                    "mls_mm": item.mls_mm,
                    "heatmap_peak": item.heatmap_peak,
                }
                for item in slices
            ])
            frame.at[index, "error"] = ""
        except Exception as exc:
            frame.at[index, "error"] = f"{type(exc).__name__}: {exc}"
        frame.at[index, "runtime_s"] = time.perf_counter() - started
        _atomic_csv(frame, output_csv)
        completed = int(frame["slice_predictions_json"].fillna("").astype(bool).sum())
        print(f"MLS v2 eval {row['study_id']}: {completed}/{len(frame)}", flush=True)
        torch.cuda.empty_cache()

    failures = frame["error"].fillna("") != ""
    if failures.any() or frame["slice_predictions_json"].fillna("").eq("").any():
        raise RuntimeError(f"Strict MLS evaluation incomplete: failures={int(failures.sum())}")

    decoded = [
        [
            type("Slice", (), item)()
            for item in json.loads(payload)
        ]
        for payload in frame["slice_predictions_json"]
    ]
    truth_values = frame["gt_MLS_mm"].to_numpy(float)
    profiles: list[dict] = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for top_k in (1, 3, 5):
            for aggregation in ("median", "p90", "max"):
                predictions = np.asarray([
                    aggregate_study_mls(
                        study, selector_threshold=threshold, top_k=top_k,
                        aggregation=aggregation,
                    )
                    for study in decoded
                ])
                metrics = _metrics(truth_values, predictions)
                profiles.append({
                    "selector_threshold": threshold,
                    "top_k": top_k,
                    "aggregation": aggregation,
                    **metrics,
                    "combined_macro_f1": _combined_macro(frame, predictions),
                })
    fixed = next(
        profile for profile in profiles
        if profile["selector_threshold"] == 0.5
        and profile["top_k"] == 3 and profile["aggregation"] == "p90"
    )
    best_mae = min(profiles, key=lambda item: item["mae_mm"])
    eligible_combined = [item for item in profiles if item["combined_macro_f1"] is not None]
    best_combined = max(eligible_combined, key=lambda item: item["combined_macro_f1"]) if eligible_combined else None
    locked_predictions = np.asarray([
        aggregate_study_mls(
            study, selector_threshold=0.5, aggregation="relative_component",
            relative_ratio=0.3, aggregation_quantile=0.75,
        )
        for study in decoded
    ])
    locked_candidate = {
        "selector_threshold": 0.5,
        "aggregation": "relative_component",
        "relative_ratio": 0.3,
        "aggregation_quantile": 0.75,
        **_metrics(truth_values, locked_predictions),
        "combined_macro_f1": _combined_macro(frame, locked_predictions),
    }
    transfer_specs = {
        "peakaware_fold01_robust_frozen_for_fold2": {
            "selector_threshold": 0.7,
            "aggregation": "relative_component",
            "relative_ratio": 0.3,
            "aggregation_quantile": 0.75,
            "min_active_slices": 1,
            "probability_weighted": True,
            "heatmap_guard_ratio": 0.0,
        },
        "peakaware_fold0_balanced_frozen_for_fold1": {
            "selector_threshold": 0.6,
            "aggregation": "relative_component",
            "relative_ratio": 0.7,
            "aggregation_quantile": 0.75,
            "min_active_slices": 3,
            "probability_weighted": False,
            "heatmap_guard_ratio": 0.5,
        },
        "w32_fold0_relative_component_balanced": {
            "selector_threshold": 0.5,
            "aggregation": "relative_component",
            "relative_ratio": 0.5,
            "aggregation_quantile": 0.75,
        },
        "w32_fold0_anchor_window_mae": {
            "selector_threshold": 0.7,
            "aggregation": "anchor_window",
            "anchor_window_radius": 2,
            "aggregation_quantile": 0.65,
        },
        "w32_fold0_top7_balanced": {
            "selector_threshold": 0.7,
            "top_k": 7,
            "aggregation": "quantile",
            "aggregation_quantile": 0.75,
        },
        "w32_crossfold_joint_component_frozen_for_fold2": {
            "selector_threshold": 0.9,
            "aggregation": "joint_component",
            "relative_ratio": 0.5,
            "aggregation_quantile": 0.9,
            "min_active_slices": 3,
            "heatmap_guard_ratio": 0.5,
        },
    }
    locked_transfer_profiles: dict[str, dict] = {}
    for name, specification in transfer_specs.items():
        transfer_predictions = np.asarray([
            aggregate_study_mls(study, **specification) for study in decoded
        ])
        locked_transfer_profiles[name] = {
            **specification,
            **_metrics(truth_values, transfer_predictions),
            "combined_macro_f1": _combined_macro(frame, transfer_predictions),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "fold": args.fold,
        "n_studies": int(len(frame)),
        "failures": int(failures.sum()),
        "runtime_total_s": float(frame["runtime_s"].sum()),
        "fixed_profile_pre_registered": fixed,
        "locked_candidate_profile": locked_candidate,
        "locked_transfer_profiles": locked_transfer_profiles,
        "best_in_sample_mae_profile": best_mae,
        "best_in_sample_combined_profile": best_combined,
        "profiles": profiles,
        "warning": "Best profiles are in-sample diagnostics; only the fixed 0.5/top3/p90 profile is pre-registered.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.mlflow_run_id:
        from mlflow.tracking import MlflowClient

        configure_tracking_environment()
        client = MlflowClient()
        logged_metrics = {
            f"e2e_fixed_{key}": float(value)
            for key, value in fixed.items()
            if isinstance(value, (int, float)) and key not in {"selector_threshold", "top_k"}
        }
        logged_metrics.update({
            "e2e_best_diagnostic_mae_mm": float(best_mae["mae_mm"]),
            "e2e_n_studies": float(len(frame)),
            "e2e_runtime_total_s": float(result["runtime_total_s"]),
        })
        for profile_name, profile in locked_transfer_profiles.items():
            logged_metrics.update({
                f"e2e_transfer_{profile_name}_{key}": float(value)
                for key, value in profile.items()
                if isinstance(value, (int, float))
                and key not in {
                    "selector_threshold", "relative_ratio", "aggregation_quantile",
                    "anchor_window_radius", "top_k", "min_active_slices",
                    "heatmap_guard_ratio",
                }
            })
        logged_metrics.update({
            f"e2e_locked_{key}": float(value)
            for key, value in locked_candidate.items()
            if isinstance(value, (int, float))
            and key not in {"selector_threshold", "relative_ratio", "aggregation_quantile"}
        })
        if best_combined is not None:
            logged_metrics["e2e_best_diagnostic_combined_macro_f1"] = float(
                best_combined["combined_macro_f1"]
            )
        for key, value in logged_metrics.items():
            client.log_metric(args.mlflow_run_id, key, value)
        report_path = config_section("mlflow", "artifact_paths", "reports")
        client.log_artifact(
            args.mlflow_run_id, str(metrics_path), f"{report_path}/end_to_end"
        )
        client.set_tag(args.mlflow_run_id, "end_to_end_evaluated", "true")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))


if __name__ == "__main__":
    main()
