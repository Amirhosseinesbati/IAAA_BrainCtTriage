"""
ground_truth.py — Extract study-level ground truth labels from training_df.csv.

The CSV is slice-level (7,508 rows). This module aggregates to study-level
(338 unique dicom_series.id) and extracts the triage_class label for each study.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from leaderboard import normalize_study_id

logger = logging.getLogger(__name__)

# Columns in the CSV that represent hemorrhage subtype areas (pixels)
HEMORRHAGE_AREA_COLS = [
    "IntraventricularHemorrhage_Area",
    "IntraparenchymalHemorrhage_Area",
    "SubarachnoidHemorrhage_Area",
    "EpiduralHemorrhage_Area",
    "SubduralHemorrhage_Area",
]

# Mapping from area column to short key (matches model output keys)
AREA_TO_KEY = {
    "IntraventricularHemorrhage_Area": "V_IVH",
    "IntraparenchymalHemorrhage_Area": "V_IPH",
    "SubarachnoidHemorrhage_Area": "V_SAH",
    "EpiduralHemorrhage_Area": "V_EDH",
    "SubduralHemorrhage_Area": "V_SDH",
}


def load_study_labels(csv_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load training_df.csv and aggregate to study-level ground truth.

    Args:
        csv_path: Path to ``Data/metadata/training_df.csv``.

    Returns:
        Dict mapping ``study_id`` (str) to a dict with keys:
            - ``triage_class`` (int): 0, 1, or 2
            - ``patient_id`` (str): DICOM PatientID
            - ``num_slices`` (int): number of DICOM slices in the series
            - ``MLS_mm`` (float): max midline shift in mm
            - ``SkullFracture`` (bool): any fracture present
            - ``V_IVH``, ``V_IPH``, ``V_SAH``, ``V_EDH``, ``V_SDH`` (float):
              total hemorrhage volumes in mL (computed from area × spacing).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading training metadata from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d slice-level rows.", len(df))

    # ---- Aggregate per series ------------------------------------------------
    agg_dict: Dict[str, Any] = {
        "triage_class": "first",               # same for all slices
        "dicom_series.PatientID": "first",
        "dicom_series.NumDicomFiles": "first",
        "MidlineShiftMM": "max",
        "SkullFracture": "max",                 # boolean OR (True if any slice)
        # Hemorrhage areas: sum across slices
        **{col: "sum" for col in HEMORRHAGE_AREA_COLS},
        # Keep spacing info (same across slices) for volume calculation
        "dicom_series.PixelSpacing0": "first",
        "dicom_series.PixelSpacing1": "first",
    }

    series_df = (
        df.groupby("dicom_series.id")
        .agg(agg_dict)
        .reset_index()
    )

    # ---- Compute volumes from pixel areas ------------------------------------
    spacing_x = series_df["dicom_series.PixelSpacing0"]
    spacing_y = series_df["dicom_series.PixelSpacing1"]
    # SliceThickness is not aggregated above; use a reasonable proxy.
    # Most series have uniform thickness; we derive it from the first slice.
    thickness_map = (
        df.groupby("dicom_series.id")["dicom_series.SliceThickness"]
        .first()
    )
    thickness = series_df["dicom_series.id"].map(thickness_map)

    # Volume (mL) = area_pixels × spacing_x × spacing_y × thickness / 1000
    factor = spacing_x * spacing_y * thickness / 1000.0

    for area_col in HEMORRHAGE_AREA_COLS:
        vol_key = AREA_TO_KEY[area_col]
        series_df[vol_key] = series_df[area_col] * factor

    series_df["total_volume_ml"] = series_df[
        list(AREA_TO_KEY.values())
    ].sum(axis=1)

    # ---- Build result dict ---------------------------------------------------
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in series_df.iterrows():
        study_id = normalize_study_id(row["dicom_series.id"])
        result[study_id] = {
            "triage_class": int(row["triage_class"]),
            "patient_id": str(row["dicom_series.PatientID"]),
            "num_slices": int(row["dicom_series.NumDicomFiles"]),
            "MLS_mm": float(row["MidlineShiftMM"]),
            "SkullFracture": bool(row["SkullFracture"]),
            **{
                vol_key: float(row[vol_key])
                for vol_key in AREA_TO_KEY.values()
            },
            "total_volume_ml": float(row["total_volume_ml"]),
        }

    logger.info(
        "Aggregated to %d unique studies (from %d slice rows).",
        len(result),
        len(df),
    )
    return result


def get_study_ids(csv_path: str | Path) -> set[str]:
    """Return the set of all study IDs present in the CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return set(normalize_study_id(x) for x in df["dicom_series.id"].unique())


if __name__ == "__main__":
    # Quick self-test
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    csv = Path(__file__).resolve().parent.parent / "Data" / "metadata" / "training_df.csv"
    labels = load_study_labels(csv)

    # Print first 3 as sample
    for study_id, info in list(labels.items())[:3]:
        print(f"\nStudy {study_id}:")
        for k, v in info.items():
            print(f"  {k}: {v}")

    # Distribution
    from collections import Counter
    dist = Counter(info["triage_class"] for info in labels.values())
    print(f"\nTriage distribution: {dict(dist)}")
