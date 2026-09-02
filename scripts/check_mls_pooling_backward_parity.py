"""Verify that old single-head prediction artifacts retain frozen pooling metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_mls_crossfold_pooling import PoolingProfile, _decode, _metrics, _predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-mae", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"study_id": str})
    profile = PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.0,
    )
    prediction = np.asarray([_predict(items, profile) for items in _decode(frame)])
    metrics = _metrics(frame["gt_MLS_mm"].to_numpy(float), prediction)
    metrics["boundary_f1"] = 0.5 * (metrics["f1_3mm"] + metrics["f1_5mm"])
    delta = abs(metrics["mae_mm"] - args.expected_mae)
    if delta > args.tolerance:
        raise AssertionError(
            f"Backward pooling parity failed: MAE delta={delta} > {args.tolerance}"
        )
    print(json.dumps({
        "n_studies": len(frame),
        "old_artifact_has_peak_probability": "peak_probability" in frame.columns,
        "metrics": metrics,
        "mae_absolute_delta": delta,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
