from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.extract_fracture_mil_features import build_full_sequence_catalog


def _write_fold(root: Path, fold: int, study_id: int, patient_slice: int = 0) -> None:
    fold_root = root / f"fold_{fold}"
    image = fold_root / "images" / "val" / f"{study_id}.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg-placeholder")
    pd.DataFrame(
        [
            {
                "study_id": study_id,
                "split": "val",
                "slice_index": patient_slice,
                "image": f"images/val/{study_id}.jpg",
                "study_fracture": int(fold == 1),
            }
        ]
    ).to_csv(fold_root / "manifest.csv", index=False)


def test_full_sequence_catalog_preserves_patient_disjoint_folds(tmp_path: Path) -> None:
    folds = tmp_path / "folds"
    _write_fold(folds, 0, 100)
    _write_fold(folds, 1, 200)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "dicom_series.id": [100, 200],
            "dicom_series.PatientID": ["p100", "p200"],
        }
    ).to_csv(metadata, index=False)

    catalog = build_full_sequence_catalog(folds, metadata, n_folds=2)

    assert catalog[["study_id", "patient_id", "outer_fold"]].to_dict("records") == [
        {"study_id": "100", "patient_id": "p100", "outer_fold": 0},
        {"study_id": "200", "patient_id": "p200", "outer_fold": 1},
    ]


def test_full_sequence_catalog_rejects_patient_crossing_folds(tmp_path: Path) -> None:
    folds = tmp_path / "folds"
    _write_fold(folds, 0, 100)
    _write_fold(folds, 1, 200)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "dicom_series.id": [100, 200],
            "dicom_series.PatientID": ["same", "same"],
        }
    ).to_csv(metadata, index=False)

    with pytest.raises(ValueError, match="patient crosses outer folds"):
        build_full_sequence_catalog(folds, metadata, n_folds=2)
