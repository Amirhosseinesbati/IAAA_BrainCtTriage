"""Build a compact multi-window slice cache from the audited ICH-v2 volumes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.strategies.ich_v2.evaluation import load_slice_metadata
from src.strategies.ich_v2.geometry import voxel_volume_ml


WINDOWS = (
    (40.0, 80.0),    # brain
    (75.0, 215.0),   # subdural
    (600.0, 2800.0), # bone/context
)
OUTPUT_LABELS = ("any_ich", "IVH", "IPH", "SDH", "EDH", "SAH")
CLASS_IDS = (1, 2, 3, 4, 5)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_hu(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map HU to uint8 for one CT window."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    lower = center - width / 2.0
    scaled = (np.asarray(image, dtype=np.float32) - lower) / width
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def multi_window_slice(image: np.ndarray, image_size: int) -> np.ndarray:
    """Return three registered CT windows as ``(3, H, W)`` uint8."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    channels = []
    for center, width in WINDOWS:
        channel = window_hu(image, center, width)
        if channel.shape != (image_size, image_size):
            channel = np.asarray(
                Image.fromarray(channel, mode="L").resize(
                    (image_size, image_size), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        channels.append(channel)
    return np.stack(channels, axis=0)


def resize_label_slice(label: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a categorical label map with nearest-neighbour interpolation."""
    value = np.asarray(label, dtype=np.uint8)
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional label map, got {value.shape}")
    if value.min() < 0 or value.max() > max(CLASS_IDS):
        raise ValueError("ICH label map contains an unsupported class id")
    if value.shape == (image_size, image_size):
        return value.copy()
    return np.asarray(
        Image.fromarray(value, mode="L").resize(
            (image_size, image_size), resample=Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _slice_thickness_by_study() -> tuple[pd.Series, Path]:
    metadata, source = load_slice_metadata()
    required = {"dicom_series.id", "dicom_series.SliceThickness"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Metadata is missing slice-thickness columns: {sorted(missing)}")
    frame = metadata.loc[:, list(required)].copy()
    frame["study_id"] = frame.pop("dicom_series.id").astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    frame["slice_thickness_mm"] = pd.to_numeric(
        frame.pop("dicom_series.SliceThickness"), errors="raise"
    )
    variation = frame.groupby("study_id")["slice_thickness_mm"].nunique()
    if (variation > 1).any():
        raise ValueError("SliceThickness varies within at least one DICOM study")
    values = frame.groupby("study_id")["slice_thickness_mm"].first()
    if (values <= 0).any():
        raise ValueError("SliceThickness must be positive")
    return values, source


def build_slice_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    image_size: int = 320,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create per-study uint8 windows and a slice-level supervision manifest."""
    source = Path(dataset_dir)
    output = Path(output_dir)
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "manifest.csv"
    source_slice_targets_path = source / "slice_targets.csv"
    studies = pd.read_csv(
        source_manifest_path,
        dtype={"study_id": str, "patient_id": str},
    )
    if not source_slice_targets_path.is_file():
        raise FileNotFoundError(
            "Audited ICH slice targets are missing; rebuild the ICH-v2 dataset first: "
            f"{source_slice_targets_path}"
        )
    slice_targets = pd.read_csv(
        source_slice_targets_path, dtype={"study_id": str, "patient_id": str}
    )
    target_columns = {
        "study_id", "slice_index", "classification_known", "segmentation_known",
        "metadata_missing", "supervision_mismatch", *OUTPUT_LABELS[1:],
    }
    missing_target_columns = target_columns - set(slice_targets)
    if missing_target_columns:
        raise ValueError(
            f"Audited slice targets are missing columns: {sorted(missing_target_columns)}"
        )
    slice_thickness, thickness_metadata_source = _slice_thickness_by_study()
    missing_thickness = sorted(set(studies["study_id"]) - set(slice_thickness.index))
    if missing_thickness:
        raise ValueError(
            f"SliceThickness is unavailable for studies: {missing_thickness[:10]}"
        )
    rows: list[dict[str, object]] = []
    for study in tqdm(studies.itertuples(index=False), total=len(studies), desc="ICH 2.5D cache"):
        study_id = str(study.study_id)
        study_slice_targets = slice_targets.loc[
            slice_targets["study_id"] == study_id
        ].sort_values("slice_index").reset_index(drop=True)
        cache_path = images_dir / f"BRN_{study_id}.npy"
        label_cache_path = labels_dir / f"BRN_{study_id}.npy"
        image_nifti = nib.load(str(study.image))
        image = np.asarray(image_nifti.dataobj, dtype=np.float32)
        label = np.asarray(nib.load(str(study.label)).dataobj, dtype=np.uint8)
        supervision = np.asarray(
            nib.load(str(study.supervision)).dataobj,
            dtype=np.uint8,
        )
        if image.shape != label.shape or image.shape != supervision.shape:
            raise ValueError(f"Shape mismatch in study {study_id}")
        if image.ndim != 3:
            raise ValueError(f"Expected HWD volume for study {study_id}, got {image.shape}")
        if len(study_slice_targets) != image.shape[2] or not np.array_equal(
            study_slice_targets["slice_index"].to_numpy(dtype=np.int64),
            np.arange(image.shape[2]),
        ):
            raise ValueError(f"Audited slice targets do not match depth for {study_id}")

        if overwrite or not cache_path.is_file():
            cached = np.stack([
                multi_window_slice(image[:, :, index], image_size)
                for index in range(image.shape[2])
            ], axis=0)
            _atomic_save_array(cache_path, cached)
        else:
            cached = np.load(cache_path, mmap_mode="r")
            expected = (image.shape[2], len(WINDOWS), image_size, image_size)
            if tuple(cached.shape) != expected or cached.dtype != np.uint8:
                raise ValueError(
                    f"Invalid existing cache for {study_id}: {cached.shape}, {cached.dtype}"
                )

        if overwrite or not label_cache_path.is_file():
            cached_labels = np.stack([
                resize_label_slice(label[:, :, index], image_size)
                for index in range(label.shape[2])
            ], axis=0)
            _atomic_save_array(label_cache_path, cached_labels)
        else:
            cached_labels = np.load(label_cache_path, mmap_mode="r")
            expected_labels = (label.shape[2], image_size, image_size)
            if tuple(cached_labels.shape) != expected_labels or cached_labels.dtype != np.uint8:
                raise ValueError(
                    f"Invalid existing label cache for {study_id}: "
                    f"{cached_labels.shape}, {cached_labels.dtype}"
                )

        native_height, native_width = image.shape[:2]
        resized_affine_voxel_volume_ml = voxel_volume_ml(image_nifti.affine) * (
            native_height * native_width / float(image_size * image_size)
        )
        row_axis = image_nifti.affine[:3, 0]
        column_axis = image_nifti.affine[:3, 1]
        native_pixel_area_mm2 = float(np.linalg.norm(np.cross(row_axis, column_axis)))
        slice_spacing_mm = float(image_nifti.header.get_zooms()[2])
        slice_thickness_mm = float(slice_thickness.loc[study_id])
        resized_voxel_volume_ml = (
            native_pixel_area_mm2
            * slice_thickness_mm
            / 1000.0
            * native_height
            * native_width
            / float(image_size * image_size)
        )

        study_targets: list[list[int]] = []
        for index in range(image.shape[2]):
            known = bool(np.max(supervision[:, :, index]) > 0)
            target_row = study_slice_targets.iloc[index]
            if known != bool(target_row["segmentation_known"]):
                raise ValueError(f"NIfTI/sidecar supervision mismatch for {study_id}/{index}")
            subtype_targets = [int(target_row[label]) for label in OUTPUT_LABELS[1:]]
            targets = [int(any(subtype_targets)), *subtype_targets]
            study_targets.append(targets)
            rows.append({
                "study_id": study_id,
                "patient_id": str(study.patient_id),
                "fold": int(study.fold),
                "triage_class": int(study.triage_class),
                "supervision_type": str(study.supervision_type),
                "slice_index": index,
                "slice_count": int(image.shape[2]),
                "known": int(known),
                "segmentation_known": int(target_row["segmentation_known"]),
                "classification_known": int(target_row["classification_known"]),
                "metadata_missing": int(target_row["metadata_missing"]),
                "supervision_mismatch": int(target_row["supervision_mismatch"]),
                "cache_path": str(cache_path),
                "label_cache_path": str(label_cache_path),
                "resized_voxel_volume_ml": float(resized_voxel_volume_ml),
                "resized_affine_voxel_volume_ml": float(
                    resized_affine_voxel_volume_ml
                ),
                "slice_spacing_mm": slice_spacing_mm,
                "slice_thickness_mm": slice_thickness_mm,
                "spacing_to_thickness_ratio": slice_spacing_mm / slice_thickness_mm,
                "native_height": int(native_height),
                "native_width": int(native_width),
                **{name: value for name, value in zip(OUTPUT_LABELS, targets, strict=True)},
            })

        known_targets = np.asarray([
            target for target, row_known in zip(
                study_targets,
                np.max(supervision, axis=(0, 1)) > 0,
                strict=True,
            ) if row_known
        ])
        if str(study.supervision_type) == "clean_negative" and known_targets[:, 0].any():
            raise ValueError(f"Clean-negative study {study_id} contains ICH labels")

    frame = pd.DataFrame(rows).sort_values(["study_id", "slice_index"]).reset_index(drop=True)
    manifest_path = output / "slice_manifest.csv"
    frame.to_csv(manifest_path, index=False)
    payload = {
        "schema_version": 4,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_slice_targets": str(source_slice_targets_path),
        "source_slice_targets_sha256": file_sha256(source_slice_targets_path),
        "image_size": image_size,
        "windows": [list(window) for window in WINDOWS],
        "adjacent_radius": 1,
        "input_channels": 9,
        "output_labels": list(OUTPUT_LABELS),
        "segmentation_classes": [0, *CLASS_IDS],
        "label_cache": "uint8 categorical center-slice masks",
        "resized_voxel_volume": "in-plane affine pixel area times SliceThickness, scaled to cache resolution",
        "volume_convention": "in-plane affine pixel area times DICOM SliceThickness",
        "affine_spacing_volume_column": "resized_affine_voxel_volume_ml",
        "slice_thickness_metadata_source": str(thickness_metadata_source),
        "slices": int(len(frame)),
        "known_slices": int(frame["known"].sum()),
        "segmentation_known_slices": int(frame["segmentation_known"].sum()),
        "classification_known_slices": int(frame["classification_known"].sum()),
        "metadata_missing_slices": int(frame["metadata_missing"].sum()),
        "spatial_mismatch_slices": int(frame["supervision_mismatch"].sum()),
        "unknown_slices": int((frame["known"] == 0).sum()),
        "positive_slices": int(
            frame.loc[frame["classification_known"] == 1, "any_ich"].sum()
        ),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (output / "cache.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return frame
