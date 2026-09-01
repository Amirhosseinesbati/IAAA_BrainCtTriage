"""Compare frozen MLS checkpoints with a calibration-aware pooling grid.

This script performs no model inference.  It consumes full-series CUDA
predictions produced by ``evaluate_mls_multitask_checkpoint.py`` and only
evaluates inexpensive study-level aggregation rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.triage import triage_from_intermediates

from search_mls_crossfold_pooling import (
    _decode,
    _metrics,
    _predict,
    _profiles,
    _python_scalar,
    _selector_summary,
)


VOLUME_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=CSV_PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return label.strip(), Path(raw_path)


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean([
        f1_score(truth == label, prediction == label, zero_division=0)
        for label in (0, 1, 2)
    ]))


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


def _profile_payload(row: pd.Series, frame: pd.DataFrame, decoded: list[list[dict]]) -> dict:
    profile_keys = list(asdict(_profiles()[0]))
    profile_type = type(_profiles()[0])
    profile = profile_type(**{
        key: _python_scalar(row[key]) for key in profile_keys
    })
    predictions = np.asarray([_predict(items, profile) for items in decoded])
    payload = {key: _python_scalar(row[key]) for key in row.index}
    payload["combined_macro_f1"] = _combined_macro(frame, predictions)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", type=_parse_candidate, required=True,
        help="Repeat LABEL=study_slice_predictions.csv for each frozen checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.candidate]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    candidate_state: dict[str, tuple[pd.DataFrame, list[list[dict]]]] = {}
    profiles = _profiles()
    for label, path in args.candidate:
        frame = pd.read_csv(path, dtype={"study_id": str})
        decoded = _decode(frame)
        truth = frame["gt_MLS_mm"].to_numpy(float)
        candidate_state[label] = (frame, decoded)
        summaries[label] = _selector_summary(frame, decoded)
        for profile in profiles:
            prediction = np.asarray([_predict(items, profile) for items in decoded])
            metrics = _metrics(truth, prediction)
            boundary_f1 = float(np.mean([metrics["f1_3mm"], metrics["f1_5mm"]]))
            all_rows.append({
                "candidate": label,
                **asdict(profile),
                **metrics,
                "boundary_f1": boundary_f1,
                "selection_objective": metrics["mae_mm"] + 2.0 * (1.0 - boundary_f1),
            })

    results = pd.DataFrame(all_rows)
    selection: dict[str, dict] = {}
    for label in labels:
        subset = results.loc[results["candidate"] == label]
        frame, decoded = candidate_state[label]
        selected_rows = {
            "best_mae": subset.sort_values(["mae_mm", "rmse_mm"]).iloc[0],
            "best_boundary": subset.sort_values(
                ["boundary_f1", "mae_mm"], ascending=[False, True]
            ).iloc[0],
            "best_balanced": subset.sort_values(
                ["selection_objective", "mae_mm"]
            ).iloc[0],
        }
        selection[label] = {
            name: _profile_payload(row, frame, decoded)
            for name, row in selected_rows.items()
        }

    global_balanced = results.sort_values(
        ["selection_objective", "mae_mm", "rmse_mm"]
    ).iloc[0]
    global_label = str(global_balanced["candidate"])
    global_frame, global_decoded = candidate_state[global_label]
    payload = {
        "warning": (
            "All profiles are diagnostics on the same OOF fold. Use cross-fold "
            "transfer before locking a production profile. No model inference was run here."
        ),
        "n_candidates": len(labels),
        "n_profiles_per_candidate": len(profiles),
        "selector_feature_summary": summaries,
        "candidate_selection": selection,
        "global_best_balanced": _profile_payload(
            global_balanced, global_frame, global_decoded
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "checkpoint_pooling_grid.csv", index=False)
    (args.output_dir / "checkpoint_pooling_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
