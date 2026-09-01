"""Diagnostic search for robust study-level MLS pooling profiles.

This uses saved GPU predictions only.  The search is in-sample on one fold and
is therefore evidence for selecting a candidate to lock and test on another
OOF fold, not an unbiased performance estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.triage import triage_from_intermediates

VOLUME_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _candidate(items: list[dict], family: str, gate: float, size: int, q: float, ratio: float) -> float:
    probabilities = np.asarray([float(item["selector_probability"]) for item in items])
    values = np.asarray([float(item["mls_mm"]) for item in items])
    anchor = int(np.argmax(probabilities))
    if probabilities[anchor] < gate:
        return 0.1
    if family == "topk":
        indices = np.argsort(-probabilities)[:size]
    elif family == "anchor_window":
        indices = np.arange(max(0, anchor - size), min(len(items), anchor + size + 1))
    elif family == "relative_component":
        active = probabilities >= probabilities[anchor] * ratio
        left = anchor
        right = anchor
        while left > 0 and active[left - 1]:
            left -= 1
        while right + 1 < len(items) and active[right + 1]:
            right += 1
        indices = np.arange(left, right + 1)
    else:
        raise ValueError(family)
    return _quantile(values[indices], q)


def _combined(reference: pd.DataFrame, predictions: np.ndarray) -> float:
    labels: list[int] = []
    for prediction, (_, row) in zip(predictions, reference.iterrows()):
        values = {key: float(row[f"pred_{key}"]) for key in VOLUME_KEYS}
        values.update({
            "fracture_prob": float(row["pred_fracture_prob"]),
            "MLS_mm": float(prediction),
        })
        labels.append(triage_from_intermediates(values))
    return float(f1_score(
        reference["triage_class"].to_numpy(int), np.asarray(labels),
        labels=[0, 1, 2], average="macro", zero_division=0,
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path,
        default=PROJECT_ROOT / "reports/checkpoint_evaluation/fold_0_predictions.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, dtype={"study_id": str})
    reference = pd.read_csv(args.reference, dtype={"study_id": str})
    reference = frame[["study_id", "gt_MLS_mm"]].merge(
        reference.drop(columns=["gt_MLS_mm"]), on="study_id", validate="one_to_one"
    )
    decoded = [json.loads(payload) for payload in frame["slice_predictions_json"]]
    truth = frame["gt_MLS_mm"].to_numpy(float)
    rows: list[dict] = []
    gates = (0.3, 0.4, 0.5, 0.6, 0.7)
    quantiles = (0.25, 0.5, 0.65, 0.75, 0.9, 1.0)

    specifications: list[tuple[str, int, float]] = []
    specifications.extend(("topk", top_k, 0.0) for top_k in (1, 3, 5, 7))
    specifications.extend(("anchor_window", radius, 0.0) for radius in (1, 2, 3, 4))
    specifications.extend(
        ("relative_component", 0, ratio) for ratio in (0.3, 0.5, 0.7, 0.8, 0.9)
    )
    for family, size, ratio in specifications:
        for gate in gates:
            for q in quantiles:
                prediction = np.asarray([
                    _candidate(items, family, gate, size, q, ratio)
                    for items in decoded
                ])
                rows.append({
                    "family": family,
                    "size": size,
                    "relative_ratio": ratio,
                    "selector_gate": gate,
                    "quantile": q,
                    "mae_mm": float(np.mean(np.abs(prediction - truth))),
                    "rmse_mm": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                    "bias_mm": float(np.mean(prediction - truth)),
                    "f1_3mm": float(f1_score(truth >= 3, prediction >= 3, zero_division=0)),
                    "f1_5mm": float(f1_score(truth >= 5, prediction >= 5, zero_division=0)),
                    "combined_macro_f1": _combined(reference, prediction),
                })
    results = pd.DataFrame(rows)
    best_mae = results.nsmallest(10, ["mae_mm", "rmse_mm"])
    best_combined = results.nlargest(10, ["combined_macro_f1", "mae_mm"])
    payload = {
        "warning": "In-sample fold-0 diagnostic search; lock a profile before the next OOF fold.",
        "n_profiles": int(len(results)),
        "best_mae_profiles": best_mae.to_dict(orient="records"),
        "best_combined_profiles": best_combined.to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
