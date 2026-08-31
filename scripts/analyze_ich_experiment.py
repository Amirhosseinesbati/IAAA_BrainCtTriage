"""Compare an ICH candidate against a study-aligned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategies.ich_v2.evaluation import VOLUME_KEYS, summarize_ich_predictions


def _metrics(frame: pd.DataFrame) -> dict:
    summary = summarize_ich_predictions(frame)
    return {
        "macro_f1": summary["oracle_context_macro_f1"],
        "normal_fpr": summary["total"]["normal_false_positive_rate"],
        "presence_f1": summary["total"]["presence_f1_at_0_1ml"],
        "mae_ml": summary["total"]["mae_ml"],
        "bias_ml": summary["total"]["bias_ml"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline, dtype={"study_id": str, "patient_id": str})
    candidate = pd.read_csv(args.candidate, dtype={"study_id": str, "patient_id": str})
    if set(baseline["study_id"]) != set(candidate["study_id"]):
        raise ValueError("Baseline and candidate study sets differ")
    candidate = baseline[["study_id"]].merge(
        candidate, on="study_id", how="left", validate="one_to_one"
    )

    gt_total = candidate[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    pred_total = candidate[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    payload = {
        "studies": int(len(candidate)),
        "baseline": _metrics(baseline),
        "candidate": _metrics(candidate),
        "delta_macro_f1": _metrics(candidate)["macro_f1"] - _metrics(baseline)["macro_f1"],
        "mean_predicted_volume_ml": {
            "baseline": {
                key: float(baseline[f"pred_{key}"].mean()) for key in VOLUME_KEYS
            },
            "candidate": {
                key: float(candidate[f"pred_{key}"].mean()) for key in VOLUME_KEYS
            },
        },
        "candidate_total_volume_quantiles_ml": {
            "ground_truth_absent": np.quantile(
                pred_total[gt_total <= 0], [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "ground_truth_present": np.quantile(
                pred_total[gt_total > 0], [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
        },
        "diagnostic_total_volume_gates": [],
    }
    for threshold in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20):
        gated = candidate.copy()
        total = gated[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1)
        gated.loc[
            total < threshold, [f"pred_{key}" for key in VOLUME_KEYS]
        ] = 0.0
        payload["diagnostic_total_volume_gates"].append({
            "threshold_ml": threshold,
            **_metrics(gated),
        })

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
