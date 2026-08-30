"""Leakage-safe, study-aware YOLO dataset builder for skull fractures."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, WINDOWS
from src.evaluation.splits import split_study_ids
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FractureDatasetConfig:
    fold: int = 0
    seed: int = RANDOM_SEED
    negative_slices_per_study: int = 12
    positive_extra_negative_slices: int = 8
    positive_context_radius: int = 1
    positive_slice_repeat: int = 1
    neighbor_channels: bool = False
    jpeg_quality: int = 95
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if self.negative_slices_per_study < 1:
            raise ValueError("negative_slices_per_study must be positive")
        if self.positive_extra_negative_slices < 0:
            raise ValueError("positive_extra_negative_slices cannot be negative")
        if self.positive_context_radius < 0:
            raise ValueError("positive_context_radius cannot be negative")
        if self.positive_slice_repeat < 1:
            raise ValueError("positive_slice_repeat must be positive")
        if not 80 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 80 and 100")


def _uniform_indices(candidates: list[int], count: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)
    positions = np.linspace(0, len(candidates) - 1, num=count)
    return sorted({candidates[int(round(position))] for position in positions})


def _neighbor_channel_bgr(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
) -> np.ndarray:
    """Store neighbors as BGR so Ultralytics yields RGB=[previous,current,next]."""
    if previous.shape != current.shape or following.shape != current.shape:
        raise ValueError("Neighbor slices must have identical shapes")
    return np.stack([following, current, previous], axis=-1)


class FractureDatasetV2Builder:
    """Build a detector dataset whose validation distribution matches inference.

    Validation contains every DICOM slice. Training keeps every positive box,
    nearby context slices, and a deterministic z-spanning sample from every
    negative study. A CSV manifest records every materialized slice.
    """

    MARKER = ".fracture_dataset_v2.json"

    def __init__(
        self,
        raw_dicom_dir: str | Path,
        raw_json_dir: str | Path,
        output_dir: str | Path,
        config: FractureDatasetConfig | None = None,
    ) -> None:
        self.raw_dicom_dir = Path(raw_dicom_dir)
        self.raw_json_dir = Path(raw_json_dir)
        self.output_dir = Path(output_dir)
        self.config = config or FractureDatasetConfig()
    def _prepare_output(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            marker = self.output_dir / self.MARKER
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Refusing to mix a new dataset with existing files: {self.output_dir}. "
                    "Use overwrite=True for an intentional rebuild."
                )
            if not marker.is_file():
                raise RuntimeError(
                    f"Refusing to remove an unrecognized directory without {self.MARKER}: "
                    f"{self.output_dir}"
                )
            shutil.rmtree(self.output_dir)
        for relative in ("images/train", "images/val", "labels/train", "labels/val"):
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)

    def _box_map(self, study_id: str) -> dict[str, list[list[float]]]:
        annotation_dir = self.raw_json_dir / study_id
        if not annotation_dir.is_dir():
            return {}
        parser = AnnotationParser(str(annotation_dir))
        result: dict[str, list[list[float]]] = {}
        for json_path in sorted(annotation_dir.glob("*.json")):
            dicom_name = json_path.with_suffix(".dcm").name
            boxes = parser.parse_slice(dicom_name)["bboxes"]
            if boxes:
                result[dicom_name] = boxes
        return result

    def _selected_indices(
        self,
        split: str,
        slice_names: list[str],
        positive_names: set[str],
    ) -> list[int]:
        if split == "val":
            return list(range(len(slice_names)))
        positive_indices = {index for index, name in enumerate(slice_names) if name in positive_names}
        if not positive_indices:
            return _uniform_indices(list(range(len(slice_names))), self.config.negative_slices_per_study)

        selected = set(positive_indices)
        for index in positive_indices:
            start = max(0, index - self.config.positive_context_radius)
            stop = min(len(slice_names), index + self.config.positive_context_radius + 1)
            selected.update(range(start, stop))
        remaining = [index for index in range(len(slice_names)) if index not in selected]
        selected.update(_uniform_indices(remaining, self.config.positive_extra_negative_slices))
        return sorted(selected)

    def build(self) -> dict[str, object]:
        self._prepare_output()
        study_ids = sorted(
            path.name for path in self.raw_dicom_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        train_ids, val_ids = split_study_ids(study_ids, self.config.fold)
        records: list[dict[str, object]] = []
        study_summary: list[dict[str, object]] = []

        for study_id in study_ids:
            split = "train" if study_id in train_ids else "val"
            boxes_by_name = self._box_map(study_id)
            reader = BrainDicomReader(str(self.raw_dicom_dir / study_id)).load_and_sort()
            slice_names = [os.path.basename(dataset.filename) for dataset in reader.slices]
            selected = self._selected_indices(split, slice_names, set(boxes_by_name))
            bone_cache: dict[int, np.ndarray] = {}

            def bone_at(slice_index: int) -> np.ndarray:
                clamped = min(max(slice_index, 0), len(reader.slices) - 1)
                if clamped not in bone_cache:
                    source = reader.slices[clamped]
                    hu = reader._pixel_to_hu(source.pixel_array, source)
                    bone = BrainDicomReader.apply_windowing(hu, WINDOWS["bone"])
                    bone_cache[clamped] = (bone * 255).astype(np.uint8)
                return bone_cache[clamped]

            study_summary.append({
                "study_id": study_id,
                "split": split,
                "fracture": int(bool(boxes_by_name)),
                "available_slices": len(slice_names),
                "selected_slices": len(selected),
                "positive_slices": len(boxes_by_name),
            })

            for index in selected:
                dicom_name = slice_names[index]
                current = bone_at(index)
                if self.config.neighbor_channels:
                    image = _neighbor_channel_bgr(
                        bone_at(index - 1),
                        current,
                        bone_at(index + 1),
                    )
                else:
                    image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
                stem = f"{study_id}__{index:04d}__{Path(dicom_name).stem}"
                suffix = ".png" if self.config.neighbor_channels else ".jpg"
                image_path = self.output_dir / "images" / split / f"{stem}{suffix}"
                label_path = self.output_dir / "labels" / split / f"{stem}.txt"
                write_parameters = (
                    [cv2.IMWRITE_PNG_COMPRESSION, 3]
                    if self.config.neighbor_channels
                    else [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                if not cv2.imwrite(
                    str(image_path), image, write_parameters
                ):
                    raise IOError(f"Failed to write {image_path}")

                rows, cols = image.shape[:2]
                label_lines: list[str] = []
                for x, y, width, height in boxes_by_name.get(dicom_name, []):
                    cx = (x + width / 2.0) / cols
                    cy = (y + height / 2.0) / rows
                    wn = width / cols
                    hn = height / rows
                    values = (cx, cy, wn, hn)
                    if not all(np.isfinite(values)) or not all(0.0 <= value <= 1.0 for value in values):
                        raise ValueError(f"Invalid normalized box in {study_id}/{dicom_name}: {values}")
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
                label_path.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
                )
                records.append({
                    "study_id": study_id,
                    "split": split,
                    "slice_index": index,
                    "dicom_name": dicom_name,
                    "image": image_path.relative_to(self.output_dir).as_posix(),
                    "label": label_path.relative_to(self.output_dir).as_posix(),
                    "study_fracture": int(bool(boxes_by_name)),
                    "slice_fracture": int(bool(label_lines)),
                    "n_boxes": len(label_lines),
                })

        manifest = pd.DataFrame.from_records(records)
        studies = pd.DataFrame.from_records(study_summary)
        manifest.to_csv(self.output_dir / "manifest.csv", index=False)
        studies.to_csv(self.output_dir / "studies.csv", index=False)
        train_entries: list[str] = []
        for record in records:
            if record["split"] != "train":
                continue
            repeats = (
                self.config.positive_slice_repeat
                if record["slice_fracture"]
                else 1
            )
            image_path = (self.output_dir / str(record["image"])).resolve().as_posix()
            train_entries.extend([image_path] * repeats)
        (self.output_dir / "train.txt").write_text(
            "\n".join(train_entries) + "\n", encoding="utf-8"
        )
        (self.output_dir / "dataset.yaml").write_text(
            "\n".join([
                f"path: {self.output_dir.resolve().as_posix()}",
                "train: train.txt",
                "val: images/val",
                "names:",
                "  0: fracture",
                "",
            ]),
            encoding="utf-8",
        )
        summary = {
            "schema_version": 2,
            "config": asdict(self.config),
            "studies": int(len(studies)),
            "fracture_studies": int(studies["fracture"].sum()),
            "images": int(len(manifest)),
            "positive_images": int(manifest["slice_fracture"].sum()),
            "train_entries": len(train_entries),
            "split_studies": studies.groupby("split").size().astype(int).to_dict(),
            "split_images": manifest.groupby("split").size().astype(int).to_dict(),
            "split_fracture_studies": studies.groupby("split")["fracture"].sum().astype(int).to_dict(),
        }
        (self.output_dir / self.MARKER).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Fracture dataset v2 built: %s", summary)
        return summary
