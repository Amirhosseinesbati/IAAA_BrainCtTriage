"""Build a compact multi-window slice cache from the audited ICH-v2 volumes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


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
            channel = cv2.resize(
                channel,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,
            )
        channels.append(channel)
    return np.stack(channels, axis=0)


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


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
    images_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "manifest.csv"
    studies = pd.read_csv(
        source_manifest_path,
        dtype={"study_id": str, "patient_id": str},
    )
    rows: list[dict[str, object]] = []
    for study in tqdm(studies.itertuples(index=False), total=len(studies), desc="ICH 2.5D cache"):
        study_id = str(study.study_id)
        cache_path = images_dir / f"BRN_{study_id}.npy"
        image = np.asarray(nib.load(str(study.image)).dataobj, dtype=np.float32)
        label = np.asarray(nib.load(str(study.label)).dataobj, dtype=np.uint8)
        supervision = np.asarray(
            nib.load(str(study.supervision)).dataobj,
            dtype=np.uint8,
        )
        if image.shape != label.shape or image.shape != supervision.shape:
            raise ValueError(f"Shape mismatch in study {study_id}")
        if image.ndim != 3:
            raise ValueError(f"Expected HWD volume for study {study_id}, got {image.shape}")

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

        study_targets: list[list[int]] = []
        for index in range(image.shape[2]):
            known = bool(np.max(supervision[:, :, index]) > 0)
            subtype_targets = [
                int(np.any(label[:, :, index] == class_id)) for class_id in CLASS_IDS
            ]
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
                "cache_path": str(cache_path),
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
        "schema_version": 1,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "image_size": image_size,
        "windows": [list(window) for window in WINDOWS],
        "adjacent_radius": 1,
        "input_channels": 9,
        "output_labels": list(OUTPUT_LABELS),
        "slices": int(len(frame)),
        "known_slices": int(frame["known"].sum()),
        "unknown_slices": int((frame["known"] == 0).sum()),
        "positive_slices": int(frame.loc[frame["known"] == 1, "any_ich"].sum()),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (output / "cache.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return frame
