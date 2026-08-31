"""Build the all-study ICH-v2 NIfTI dataset with partial supervision masks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    FOLD_MANIFEST_PATH,
    ICH_LABELS,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_ANNOTATIONS_DIR,
    RAW_TRAINING_DIR,
    TRAINING_CSV_PATH,
    get_patient_ids,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser
from src.strategies.ich_v2.geometry import dicom_affine_ras
from src.strategies.ich_v2.supervision import (
    clean_negative_study_ids,
    full_negative_targets,
    stack_partial_targets,
)

logger = logging.getLogger(__name__)


def _load_metadata(path: Path) -> tuple[pd.DataFrame, Path]:
    """Load generated CSV when present, otherwise use the DVC-tracked pickle."""
    selected = path
    if not selected.is_file():
        selected = RAW_DIR / "training_df.pkl"
    if not selected.is_file():
        raise FileNotFoundError(
            f"Neither metadata table exists: {path} or {RAW_DIR / 'training_df.pkl'}"
        )
    if selected.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(selected), selected
    return pd.read_csv(selected), selected


class ICHV2DatasetBuilder:
    """Create image, label and supervision NIfTI files for all 338 studies."""

    def __init__(
        self,
        *,
        raw_dicom_dir: str | Path = RAW_TRAINING_DIR,
        raw_json_dir: str | Path = RAW_ANNOTATIONS_DIR,
        metadata_path: str | Path = TRAINING_CSV_PATH,
        fold_manifest_path: str | Path = FOLD_MANIFEST_PATH,
        output_dir: str | Path | None = None,
    ) -> None:
        self.raw_dicom_dir = Path(raw_dicom_dir)
        self.raw_json_dir = Path(raw_json_dir)
        self.metadata_path = Path(metadata_path)
        self.fold_manifest_path = Path(fold_manifest_path)
        self.output_dir = Path(output_dir or (PROCESSED_DIR / "ich_v2" / "BrainICHPartial"))
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.supervision_dir = self.output_dir / "supervision"

    def build(
        self,
        *,
        study_ids: list[str] | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        metadata, metadata_source = _load_metadata(self.metadata_path)
        clean_negatives = clean_negative_study_ids(metadata)
        available = get_patient_ids(self.raw_dicom_dir)
        selected = sorted(str(value) for value in (study_ids or available))
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Requested studies are missing from raw DICOM data: {missing[:10]}")

        annotated = {
            path.name for path in self.raw_json_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        }
        unsafe_unannotated = sorted(set(selected) - annotated - clean_negatives)
        if unsafe_unannotated:
            raise ValueError(
                "Unannotated studies failed the clean-negative metadata gate: "
                f"{unsafe_unannotated[:10]}"
            )

        for directory in (self.images_dir, self.labels_dir, self.supervision_dir):
            directory.mkdir(parents=True, exist_ok=True)

        fold_manifest = pd.read_csv(
            self.fold_manifest_path, dtype={"study_id": str, "patient_id": str}
        ).set_index("study_id")
        rows: list[dict[str, object]] = []
        for study_id in tqdm(selected, desc="ICH-v2 dataset"):
            result = self._build_study(
                study_id,
                clean_negative=study_id in clean_negatives and study_id not in annotated,
                overwrite=overwrite,
            )
            if study_id not in fold_manifest.index:
                raise ValueError(f"Study {study_id} is missing from the fold manifest")
            fold_row = fold_manifest.loc[study_id]
            result.update({
                "patient_id": str(fold_row["patient_id"]),
                "fold": int(fold_row["fold"]),
                "triage_class": int(fold_row["triage_class"]),
            })
            rows.append(result)

        manifest = pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)
        manifest.to_csv(self.output_dir / "manifest.csv", index=False)
        payload = {
            "dataset_name": "BrainICHPartial",
            "schema_version": 2,
            "labels": ICH_LABELS,
            "studies": int(len(manifest)),
            "clean_negative_studies": int((manifest["supervision_type"] == "clean_negative").sum()),
            "partially_annotated_studies": int((manifest["supervision_type"] == "partial_json").sum()),
            "unknown_slices": int(manifest["unknown_slices"].sum()),
            "geometry": "DICOM LPS converted to NIfTI RAS",
            "unknown_label_policy": "ignored by supervision mask",
            "metadata_source": str(metadata_source),
        }
        (self.output_dir / "dataset.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    def _build_study(
        self,
        study_id: str,
        *,
        clean_negative: bool,
        overwrite: bool,
    ) -> dict[str, object]:
        paths = {
            "image": self.images_dir / f"BRN_{study_id}.nii.gz",
            "label": self.labels_dir / f"BRN_{study_id}.nii.gz",
            "supervision": self.supervision_dir / f"BRN_{study_id}.nii.gz",
        }
        if not overwrite and all(path.is_file() for path in paths.values()):
            label = np.asarray(nib.load(str(paths["label"])).dataobj)
            supervision = np.asarray(nib.load(str(paths["supervision"])).dataobj)
            depth = int(label.shape[2])
            known_slices = int(np.count_nonzero(np.max(supervision, axis=(0, 1))))
            return {
                "study_id": study_id,
                "image": str(paths["image"]),
                "label": str(paths["label"]),
                "supervision": str(paths["supervision"]),
                "slices": depth,
                "known_slices": known_slices,
                "unknown_slices": depth - known_slices,
                "supervision_type": "clean_negative" if clean_negative else "partial_json",
            }

        reader = BrainDicomReader(str(self.raw_dicom_dir / study_id)).load_and_sort()
        image_hwd = reader.get_3d_volume_hu().astype(np.float32, copy=False)
        height, width, depth = image_hwd.shape

        if clean_negative:
            label_dhw, supervision_dhw = full_negative_targets(depth, (height, width))
            supervision_type = "clean_negative"
        else:
            parser = AnnotationParser(str(self.raw_json_dir / study_id))
            parsed = [parser.parse_slice(str(ds.SOPInstanceUID)) for ds in reader.slices]
            label_dhw, supervision_dhw = stack_partial_targets(
                parsed, shape=(height, width)
            )
            if not supervision_dhw.any():
                raise ValueError(f"Annotated study {study_id} has no matched JSON slices")
            supervision_type = "partial_json"

        if label_dhw.shape[0] != depth:
            raise ValueError(
                f"Depth mismatch for {study_id}: image={depth}, label={label_dhw.shape[0]}"
            )
        affine = dicom_affine_ras(reader.slices)
        label_hwd = label_dhw.transpose(1, 2, 0)
        supervision_hwd = supervision_dhw.transpose(1, 2, 0)
        nib.save(nib.Nifti1Image(image_hwd, affine), str(paths["image"]))
        nib.save(nib.Nifti1Image(label_hwd, affine), str(paths["label"]))
        nib.save(nib.Nifti1Image(supervision_hwd, affine), str(paths["supervision"]))

        known_slices = int(np.count_nonzero(np.max(supervision_dhw, axis=(1, 2))))
        return {
            "study_id": study_id,
            "image": str(paths["image"]),
            "label": str(paths["label"]),
            "supervision": str(paths["supervision"]),
            "slices": depth,
            "known_slices": known_slices,
            "unknown_slices": depth - known_slices,
            "supervision_type": supervision_type,
        }
