"""Build the MLS multitask-v2 index with explicit negative supervision.

The existing positive PNGs are reused by absolute path. Only additional hard
negatives and clean-study negatives are decoded from DICOM and written to the
new directory. The build is deterministic and safe to resume.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_TRAINING_DIR, WINDOWS
from src.preprocessing.core.dicom_reader import BrainDicomReader

SOURCE_ROOT = PROJECT_ROOT / "Data" / "processed" / "mls_dataset"
OUTPUT_ROOT = PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2"
RAW_METADATA_PATH = PROJECT_ROOT / "Data" / "raw" / "training_df.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "mls_experiments" / "dataset_v2_build.log"


def _log(message: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with REPORT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"[{stamp}] {message}\n")
    print(message, flush=True)


def _source_dicom_name(study_id: str, image_name: str) -> str:
    prefix = f"{study_id}_"
    name = image_name[len(prefix):] if image_name.startswith(prefix) else image_name
    return f"{Path(name).stem}.dcm"


def _quantile_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted({int(round(value)) for value in np.linspace(0.2, 0.8, count) * (length - 1)})


def _select_negative_indices(
    names: list[str], target_names: set[str], existing_names: set[str], count: int,
) -> list[int]:
    allowed = [
        index for index, name in enumerate(names)
        if name not in target_names and name not in existing_names
    ]
    if not allowed:
        return []
    selected: list[int] = []
    target_indices = [index for index, name in enumerate(names) if name in target_names]
    for target in target_indices:
        for delta in (-2, -1, 1, 2):
            candidate = target + delta
            if candidate in allowed and candidate not in selected:
                selected.append(candidate)
                if len(selected) == count:
                    return sorted(selected)
    for relative in _quantile_indices(len(allowed), count):
        candidate = allowed[relative]
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break
    return sorted(selected)


def _clean_negative_study_ids(
    metadata: pd.DataFrame,
    target_study_ids: set[str],
) -> set[str]:
    """Return studies whose authoritative maximum MLS is effectively zero."""
    required = {"dicom_series.id", "MidlineShiftMM"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Raw metadata is missing required columns: {sorted(missing)}")
    working = metadata.loc[:, ["dicom_series.id", "MidlineShiftMM"]].copy()
    working["dicom_series.id"] = working["dicom_series.id"].astype(str)
    working["MidlineShiftMM"] = pd.to_numeric(working["MidlineShiftMM"], errors="coerce")
    if working["MidlineShiftMM"].isna().any():
        raise ValueError("Raw metadata contains invalid MidlineShiftMM values")
    study_maximum = working.groupby("dicom_series.id")["MidlineShiftMM"].max()
    clean = set(study_maximum.index[study_maximum <= 0.100001].astype(str))
    return clean - {str(value) for value in target_study_ids}


def _window_slice(reader: BrainDicomReader, dataset) -> np.ndarray:
    hu = reader._pixel_to_hu(dataset.pixel_array, dataset)
    channels = [BrainDicomReader.apply_windowing(hu, WINDOWS[name]) for name in ("brain", "subdural", "bone")]
    rgb = np.stack(channels, axis=-1) * 255.0
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def main() -> None:
    source_csv = SOURCE_ROOT / "mls_labels.csv"
    source_images = SOURCE_ROOT / "images"
    output_images = OUTPUT_ROOT / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_csv, dtype={"patient_id": str})
    source["patient_id"] = source["patient_id"].astype(str)
    source["image_path"] = source["image_name"].map(lambda name: str((source_images / name).resolve()))
    for column in ("spacing_x", "spacing_y"):
        if column not in source:
            source[column] = np.nan

    target_by_study: dict[str, set[str]] = {}
    existing_by_study: dict[str, set[str]] = {}
    for study_id, group in source.groupby("patient_id"):
        existing_by_study[study_id] = {
            _source_dicom_name(study_id, name) for name in group["image_name"]
        }
        target_by_study[study_id] = {
            _source_dicom_name(study_id, name)
            for name in group.loc[group["is_target"] == 1, "image_name"]
        }

    if not RAW_METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing DVC-tracked raw metadata: {RAW_METADATA_PATH}")
    metadata = pd.read_pickle(RAW_METADATA_PATH)
    clean_negative_ids = _clean_negative_study_ids(metadata, set(target_by_study))
    studies = sorted(set(target_by_study) | clean_negative_ids, key=lambda value: int(value))
    _log(
        f"start studies={len(studies)} target_studies={len(target_by_study)} "
        f"clean_negative_studies={len(clean_negative_ids)} source_rows={len(source)}"
    )

    generated: list[dict] = []
    failures: list[dict] = []
    for position, study_id in enumerate(studies, start=1):
        try:
            reader = BrainDicomReader(str(RAW_TRAINING_DIR / study_id)).load_and_sort()
            names = [Path(item.filename).name for item in reader.slices]
            target_names = target_by_study.get(study_id, set())
            existing_names = existing_by_study.get(study_id, set())
            indices = _select_negative_indices(
                names, target_names, existing_names,
                count=4,
            )
            for index in indices:
                dataset = reader.slices[index]
                dicom_name = Path(dataset.filename).name
                output_name = f"{study_id}_{Path(dicom_name).stem}_negative.png"
                output_path = output_images / output_name
                if not output_path.exists():
                    image = _window_slice(reader, dataset)
                    if not cv2.imwrite(str(output_path), image):
                        raise OSError(f"cv2.imwrite failed for {output_path}")
                spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
                generated.append({
                    "patient_id": study_id,
                    "image_name": output_name,
                    "image_path": str(output_path.resolve()),
                    "is_target": 0,
                    "spacing_x": float(spacing[1]),
                    "spacing_y": float(spacing[0]),
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0,
                    "negative_source": "clean_study" if study_id in clean_negative_ids else "hard_within_study",
                })
            if position % 25 == 0 or position == len(studies):
                _log(f"progress={position}/{len(studies)} generated_rows={len(generated)}")
        except Exception as exc:
            failures.append({"study_id": study_id, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"failure study={study_id} error={type(exc).__name__}: {exc}")

    generated_frame = pd.DataFrame(generated)
    source["negative_source"] = np.where(source["is_target"] == 1, "target", "legacy_within_study")
    combined = pd.concat([source, generated_frame], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["patient_id", "image_path"], keep="first")
    combined = combined.sort_values(["patient_id", "is_target", "image_name"]).reset_index(drop=True)
    output_csv = OUTPUT_ROOT / "mls_labels_multitask.csv"
    temporary = output_csv.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output_csv)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv.resolve()),
        "truth_source": str(RAW_METADATA_PATH.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(combined)),
        "positive_rows": int((combined["is_target"] == 1).sum()),
        "negative_rows": int((combined["is_target"] == 0).sum()),
        "studies": int(combined["patient_id"].nunique()),
        "negative_studies": int(combined.loc[combined["is_target"] == 0, "patient_id"].nunique()),
        "failures": failures,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"complete {json.dumps(manifest, ensure_ascii=False)}")
    if failures:
        raise RuntimeError(f"Dataset build had {len(failures)} study failures")


if __name__ == "__main__":
    main()
