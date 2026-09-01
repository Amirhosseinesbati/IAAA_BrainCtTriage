"""Report study exposure for MLS sampler policies without model computation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import normalize_study_id, split_study_ids
from src.strategies.mls_heatmap.dataset import build_mls_sampling_weights


MODES = (
    "slice_class_balanced",
    "hybrid_study_class_balanced",
    "study_class_balanced",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "Data"
        / "processed"
        / "mls_multitask_v2"
        / "mls_labels_multitask.csv",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame["patient_id"] = frame["patient_id"].map(normalize_study_id)
    train_studies, _ = split_study_ids(frame["patient_id"].unique(), args.fold)
    train = frame.loc[frame["patient_id"].isin(train_studies)].reset_index(drop=True)
    payload: dict[str, object] = {
        "fold": args.fold,
        "train_rows": int(len(train)),
        "train_studies": int(train["patient_id"].nunique()),
        "modes": {},
    }
    for mode in MODES:
        weights = build_mls_sampling_weights(train, mode).numpy()
        weighted = train[["patient_id", "is_target"]].copy()
        weighted["sampling_mass"] = weights
        study_mass = weighted.groupby("patient_id")["sampling_mass"].sum()
        class_mass = weighted.groupby("is_target")["sampling_mass"].sum()
        payload["modes"][mode] = {
            "study_mass_min": float(study_mass.min()),
            "study_mass_max": float(study_mass.max()),
            "study_mass_max_min_ratio": float(study_mass.max() / study_mass.min()),
            "study_mass_q90_q10_ratio": float(
                np.quantile(study_mass, 0.9) / np.quantile(study_mass, 0.1)
            ),
            "class_mass": {
                str(int(label)): float(value) for label, value in class_mass.items()
            },
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
