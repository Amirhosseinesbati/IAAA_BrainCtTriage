"""Audit slice-level ICH metadata against decoded spatial JSON masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RAW_ANNOTATIONS_DIR, RAW_TRAINING_DIR
from src.preprocessing.core.json_parser import AnnotationParser
from src.strategies.ich_2p5d.cache import CLASS_IDS, OUTPUT_LABELS
from src.strategies.ich_v2.evaluation import load_slice_metadata
from src.strategies.ich_v2.supervision import ICH_AREA_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, default=RAW_ANNOTATIONS_DIR)
    parser.add_argument("--dicom-dir", type=Path, default=RAW_TRAINING_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ich_experiments/supervision_audit"),
    )
    args = parser.parse_args()

    metadata, metadata_source = load_slice_metadata()
    required = {"dicom_series.id", "dicom_series.SOPInstanceUID", *ICH_AREA_COLUMNS}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Metadata is missing supervision audit columns: {sorted(missing)}")
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
        raise ValueError("Metadata contains duplicate study/SOP rows")
    metadata_lookup = {
        (str(row.study_id), str(row.sop_instance_uid)): np.asarray(
            [float(getattr(row, column)) for column in ICH_AREA_COLUMNS],
            dtype=np.float64,
        )
        for row in frame.itertuples(index=False)
    }
    dicom_keys = sorted(
        (path.parent.name, path.stem)
        for path in args.dicom_dir.glob("*/*.dcm")
    )
    if len(dicom_keys) != len(set(dicom_keys)):
        raise ValueError("Raw DICOM tree contains duplicate study/SOP paths")
    metadata_without_dicom = sorted(set(metadata_lookup) - set(dicom_keys))
    if metadata_without_dicom:
        raise ValueError(
            "Metadata references slices absent from the DICOM tree: "
            f"{metadata_without_dicom[:5]}"
        )

    parsers: dict[str, AnnotationParser] = {}
    rows: list[dict[str, object]] = []
    for study_id, sop_uid in tqdm(
        dicom_keys, total=len(dicom_keys), desc="ICH supervision audit"
    ):
        if study_id not in parsers:
            parsers[study_id] = AnnotationParser(str(args.annotations_dir / study_id))
        parsed = parsers[study_id].parse_slice(sop_uid)
        mask = np.asarray(parsed["mask_2d"], dtype=np.uint8)
        metadata_known = (study_id, sop_uid) in metadata_lookup
        metadata_areas = metadata_lookup.get(
            (study_id, sop_uid), np.zeros(len(ICH_AREA_COLUMNS), dtype=np.float64)
        )
        metadata_presence = metadata_areas > 0.0
        mask_pixels = np.asarray(
            [np.count_nonzero(mask == class_id) for class_id in CLASS_IDS],
            dtype=np.int64,
        )
        mask_presence = mask_pixels > 0
        agreement = metadata_presence == mask_presence
        spatial_supervision_safe = bool(metadata_known and agreement.all())
        row: dict[str, object] = {
            "study_id": study_id,
            "sop_instance_uid": sop_uid,
            "metadata_known": int(metadata_known),
            "json_exists": int(bool(parsed["has_label"])),
            "json_has_segmentation": int(bool(parsed["has_segmentation"])),
            "metadata_any_ich": int(metadata_presence.any()),
            "mask_any_ich": int(mask_presence.any()),
            "metadata_mask_agreement": int(agreement.all()),
            "supervision_mismatch": int(metadata_known and not agreement.all()),
            "spatial_supervision_safe": int(spatial_supervision_safe),
            "positive_without_spatial_mask": int(
                metadata_presence.any() and not mask_presence.any()
            ),
            "mask_without_metadata_positive": int(
                mask_presence.any() and not metadata_presence.any()
            ),
        }
        for index, label in enumerate(OUTPUT_LABELS[1:]):
            row[f"metadata_{label}"] = int(metadata_presence[index])
            row[f"mask_{label}"] = int(mask_presence[index])
            row[f"metadata_area_{label}"] = float(metadata_areas[index])
            row[f"mask_pixels_{label}"] = int(mask_pixels[index])
            row[f"agreement_{label}"] = int(agreement[index])
        rows.append(row)

    audit = pd.DataFrame(rows)
    subtype_summary: dict[str, dict[str, int]] = {}
    for label in OUTPUT_LABELS[1:]:
        metadata_positive = audit[f"metadata_{label}"] > 0
        mask_positive = audit[f"mask_{label}"] > 0
        subtype_summary[label] = {
            "metadata_positive_slices": int(metadata_positive.sum()),
            "mask_positive_slices": int(mask_positive.sum()),
            "metadata_positive_mask_negative": int((metadata_positive & ~mask_positive).sum()),
            "metadata_negative_mask_positive": int((~metadata_positive & mask_positive).sum()),
            "mismatched_studies": int(
                audit.loc[metadata_positive != mask_positive, "study_id"].nunique()
            ),
        }
    unsafe = audit[audit["spatial_supervision_safe"] == 0].copy()
    mismatched = audit[audit["supervision_mismatch"] == 1].copy()
    payload = {
        "schema_version": 2,
        "metadata_source": str(metadata_source),
        "annotations_dir": str(args.annotations_dir),
        "dicom_dir": str(args.dicom_dir),
        "studies": int(audit["study_id"].nunique()),
        "slices": int(len(audit)),
        "metadata_slices": int(audit["metadata_known"].sum()),
        "metadata_missing_slices": int((audit["metadata_known"] == 0).sum()),
        "metadata_missing_studies": int(
            audit.loc[audit["metadata_known"] == 0, "study_id"].nunique()
        ),
        "json_missing_slices": int((audit["json_exists"] == 0).sum()),
        "json_without_segmentation_slices": int(
            ((audit["json_exists"] == 1) & (audit["json_has_segmentation"] == 0)).sum()
        ),
        "metadata_positive_slices": int(audit["metadata_any_ich"].sum()),
        "mask_positive_slices": int(audit["mask_any_ich"].sum()),
        "unsafe_spatial_supervision_slices": int(len(unsafe)),
        "unsafe_spatial_supervision_studies": int(unsafe["study_id"].nunique()),
        "metadata_mask_mismatch_slices": int(len(mismatched)),
        "metadata_mask_mismatch_studies": int(mismatched["study_id"].nunique()),
        "positive_without_spatial_mask_slices": int(
            audit["positive_without_spatial_mask"].sum()
        ),
        "positive_without_spatial_mask_studies": int(
            audit.loc[
                audit["positive_without_spatial_mask"] == 1, "study_id"
            ].nunique()
        ),
        "mask_without_metadata_positive_slices": int(
            audit["mask_without_metadata_positive"].sum()
        ),
        "safe_spatial_supervision_policy": (
            "classification and voxel loss exclude DICOM slices missing metadata; "
            "voxel loss additionally requires metadata subtype presence to exactly "
            "match decoded mask subtype presence"
        ),
        "subtypes": subtype_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "slice_audit.csv", index=False)
    unsafe.to_csv(args.output_dir / "unsafe_spatial_slices.csv", index=False)
    mismatched.to_csv(args.output_dir / "metadata_mask_mismatches.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
