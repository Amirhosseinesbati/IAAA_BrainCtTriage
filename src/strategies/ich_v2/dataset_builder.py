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
    ICH_AREA_COLUMNS,
    clean_negative_study_ids,
    full_negative_targets,
    stack_audited_partial_targets,
)

logger = logging.getLogger(__name__)


def _load_metadata(path: Path) -> tuple[pd.DataFrame, Path]:
    """Load generated CSV when present, otherwise use the DVC-tracked pickle."""
    raw_pickle = RAW_DIR / "training_df.pkl"
    selected = raw_pickle if path == TRAINING_CSV_PATH and raw_pickle.is_file() else path
    if not selected.is_file():
        selected = raw_pickle
    if not selected.is_file():
        raise FileNotFoundError(
            f"Neither metadata table exists: {path} or {RAW_DIR / 'training_df.pkl'}"
        )
    if selected.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(selected), selected
    return pd.read_csv(selected), selected


def _slice_targets_by_study(
    metadata: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    required = {"dicom_series.id", "dicom_series.SOPInstanceUID", *ICH_AREA_COLUMNS}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Metadata is missing ICH slice-target columns: {sorted(missing)}")
    frame = metadata.loc[
        :, ["dicom_series.id", "dicom_series.SOPInstanceUID", *ICH_AREA_COLUMNS]
    ].copy()
    frame["study_id"] = frame.pop("dicom_series.id").astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    frame["sop_instance_uid"] = frame.pop("dicom_series.SOPInstanceUID").astype(str)
    for column in ICH_AREA_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if frame.duplicated(["study_id", "sop_instance_uid"]).any():
        raise ValueError("Metadata contains duplicate study/SOP slice rows")
    result: dict[str, dict[str, np.ndarray]] = {}
    for study_id, group in frame.groupby("study_id", sort=False):
        result[str(study_id)] = {
            str(row.sop_instance_uid): np.asarray(
                [float(getattr(row, column)) > 0.0 for column in ICH_AREA_COLUMNS],
                dtype=np.uint8,
            )
            for row in group.itertuples(index=False)
        }
    return result


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
        slice_targets = _slice_targets_by_study(metadata)
        rows: list[dict[str, object]] = []
        slice_rows: list[dict[str, object]] = []
        for study_id in tqdm(selected, desc="ICH-v2 dataset"):
            if study_id not in slice_targets:
                raise ValueError(f"Study {study_id} is missing slice-level metadata")
            result = self._build_study(
                study_id,
                clean_negative=study_id in clean_negatives and study_id not in annotated,
                overwrite=overwrite,
                metadata_targets=slice_targets[study_id],
            )
            study_slice_rows = result.pop("_slice_rows")
            if study_id not in fold_manifest.index:
                raise ValueError(f"Study {study_id} is missing from the fold manifest")
            fold_row = fold_manifest.loc[study_id]
            fold_values = {
                "patient_id": str(fold_row["patient_id"]),
                "fold": int(fold_row["fold"]),
                "triage_class": int(fold_row["triage_class"]),
            }
            result.update(fold_values)
            for slice_row in study_slice_rows:
                slice_row.update(fold_values)
                slice_rows.append(slice_row)
            rows.append(result)

        manifest = pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)
        manifest.to_csv(self.output_dir / "manifest.csv", index=False)
        slice_manifest = pd.DataFrame(slice_rows).sort_values(
            ["study_id", "slice_index"]
        ).reset_index(drop=True)
        slice_manifest.to_csv(self.output_dir / "slice_targets.csv", index=False)
        payload = {
            "dataset_name": "BrainICHPartial",
            "schema_version": 3,
            "labels": ICH_LABELS,
            "studies": int(len(manifest)),
            "clean_negative_studies": int((manifest["supervision_type"] == "clean_negative").sum()),
            "partially_annotated_studies": int((manifest["supervision_type"] == "partial_json").sum()),
            "unknown_slices": int(manifest["unknown_slices"].sum()),
            "classification_target_source": "slice-level subtype area metadata",
            "classification_known_slices": int(slice_manifest["classification_known"].sum()),
            "segmentation_known_slices": int(slice_manifest["segmentation_known"].sum()),
            "metadata_missing_slices": int(slice_manifest["metadata_missing"].sum()),
            "metadata_missing_studies": int(
                slice_manifest.loc[
                    slice_manifest["metadata_missing"] == 1, "study_id"
                ].nunique()
            ),
            "spatial_mismatch_slices": int(slice_manifest["supervision_mismatch"].sum()),
            "spatial_mismatch_studies": int(
                slice_manifest.loc[
                    slice_manifest["supervision_mismatch"] == 1, "study_id"
                ].nunique()
            ),
            "geometry": "DICOM LPS converted to NIfTI RAS",
            "unknown_label_policy": (
                "DICOM slices missing metadata are excluded from classification and voxel "
                "loss; voxel loss is also masked when decoded mask subtype presence "
                "disagrees with metadata"
            ),
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
        metadata_targets: dict[str, np.ndarray],
    ) -> dict[str, object]:
        paths = {
            "image": self.images_dir / f"BRN_{study_id}.nii.gz",
            "label": self.labels_dir / f"BRN_{study_id}.nii.gz",
            "supervision": self.supervision_dir / f"BRN_{study_id}.nii.gz",
        }
        reader = BrainDicomReader(str(self.raw_dicom_dir / study_id)).load_and_sort()
        image_hwd = reader.get_3d_volume_hu().astype(np.float32, copy=False)
        height, width, depth = image_hwd.shape
        sop_uids = [str(ds.SOPInstanceUID) for ds in reader.slices]
        extra_sops = sorted(set(metadata_targets) - set(sop_uids))
        if extra_sops:
            raise ValueError(
                f"DICOM/metadata SOP mismatch for {study_id}: "
                f"metadata_without_dicom={extra_sops[:3]}"
            )
        classification_known = np.asarray(
            [sop_uid in metadata_targets for sop_uid in sop_uids], dtype=np.uint8
        )
        ordered_metadata_targets = np.stack(
            [
                metadata_targets.get(sop_uid, np.zeros(5, dtype=np.uint8))
                for sop_uid in sop_uids
            ],
            axis=0,
        ).astype(np.uint8)

        if clean_negative:
            if ordered_metadata_targets.any():
                raise ValueError(f"Clean-negative study {study_id} has positive metadata")
            label_dhw, supervision_dhw = full_negative_targets(depth, (height, width))
            supervision_dhw *= classification_known[:, None, None]
            spatially_known = classification_known.copy()
            parsed = [
                {"has_label": False, "has_segmentation": False}
                for _ in range(depth)
            ]
            supervision_type = "clean_negative"
        else:
            parser = AnnotationParser(str(self.raw_json_dir / study_id))
            parsed = [parser.parse_slice(str(ds.SOPInstanceUID)) for ds in reader.slices]
            label_dhw, supervision_dhw, spatially_known = stack_audited_partial_targets(
                parsed,
                ordered_metadata_targets,
                shape=(height, width),
                metadata_known=classification_known,
            )
            supervision_type = "partial_json"

        if label_dhw.shape[0] != depth:
            raise ValueError(
                f"Depth mismatch for {study_id}: image={depth}, label={label_dhw.shape[0]}"
            )
        affine = dicom_affine_ras(reader.slices)
        label_hwd = label_dhw.transpose(1, 2, 0)
        supervision_hwd = supervision_dhw.transpose(1, 2, 0)
        if overwrite or not paths["image"].is_file():
            nib.save(nib.Nifti1Image(image_hwd, affine), str(paths["image"]))
        for key, value in (("label", label_hwd), ("supervision", supervision_hwd)):
            should_write = overwrite or not paths[key].is_file()
            if not should_write:
                existing = np.asarray(nib.load(str(paths[key])).dataobj, dtype=np.uint8)
                should_write = existing.shape != value.shape or not np.array_equal(
                    existing, value
                )
            if should_write:
                nib.save(nib.Nifti1Image(value, affine), str(paths[key]))

        known_slices = int(np.count_nonzero(np.max(supervision_dhw, axis=(1, 2))))
        slice_rows: list[dict[str, object]] = []
        for index, (sop_uid, metadata_target, parsed_slice) in enumerate(
            zip(sop_uids, ordered_metadata_targets, parsed, strict=True)
        ):
            mask_target = np.asarray(
                [np.any(label_dhw[index] == class_id) for class_id in range(1, 6)],
                dtype=np.uint8,
            )
            slice_rows.append({
                "study_id": study_id,
                "sop_instance_uid": sop_uid,
                "slice_index": index,
                "classification_known": int(classification_known[index]),
                "segmentation_known": int(spatially_known[index]),
                "metadata_missing": int(not classification_known[index]),
                "supervision_mismatch": int(
                    classification_known[index] and not spatially_known[index]
                ),
                "json_exists": int(bool(parsed_slice.get("has_label", False))),
                "json_has_segmentation": int(
                    bool(parsed_slice.get("has_segmentation", False))
                ),
                **{
                    label: int(value)
                    for label, value in zip(
                        ("IVH", "IPH", "SDH", "EDH", "SAH"),
                        metadata_target,
                        strict=True,
                    )
                },
                **{
                    f"mask_{label}": int(value)
                    for label, value in zip(
                        ("IVH", "IPH", "SDH", "EDH", "SAH"),
                        mask_target,
                        strict=True,
                    )
                },
            })
        return {
            "study_id": study_id,
            "image": str(paths["image"]),
            "label": str(paths["label"]),
            "supervision": str(paths["supervision"]),
            "slices": depth,
            "known_slices": known_slices,
            "unknown_slices": depth - known_slices,
            "supervision_type": supervision_type,
            "_slice_rows": slice_rows,
        }
