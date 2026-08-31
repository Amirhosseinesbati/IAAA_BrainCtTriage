"""Fail-closed integrity audit for the generated ICH-v2 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.strategies.ich_v2.geometry import voxel_volume_ml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Data/processed/ich_v2/BrainICHPartial"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.dataset_dir
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "patient_id": str})
    if manifest["study_id"].duplicated().any():
        raise ValueError("Duplicate study IDs in ICH-v2 manifest")

    expected = set(manifest["study_id"])
    report: dict[str, object] = {
        "studies": int(len(manifest)),
        "manifest_sha256": _sha256(manifest_path),
        "clean_negative_studies": int((manifest["supervision_type"] == "clean_negative").sum()),
        "partial_json_studies": int((manifest["supervision_type"] == "partial_json").sum()),
        "known_slices": 0,
        "unknown_slices": 0,
        "foreground_voxels": 0,
        "unknown_foreground_voxels": 0,
        "min_voxel_volume_ml": float("inf"),
        "max_voxel_volume_ml": 0.0,
        "bytes": 0,
        "errors": [],
    }

    actual_by_kind: dict[str, set[str]] = {}
    for kind in ("images", "labels", "supervision"):
        files = list((root / kind).glob("BRN_*.nii.gz"))
        actual_by_kind[kind] = {path.name[4:-7] for path in files}
        report["bytes"] = int(report["bytes"]) + sum(path.stat().st_size for path in files)
        if actual_by_kind[kind] != expected:
            missing = sorted(expected - actual_by_kind[kind])
            extra = sorted(actual_by_kind[kind] - expected)
            report["errors"].append(f"{kind}: missing={missing[:5]} extra={extra[:5]}")

    for row in manifest.itertuples(index=False):
        study_id = str(row.study_id)
        image = nib.load(str(root / "images" / f"BRN_{study_id}.nii.gz"))
        label = nib.load(str(root / "labels" / f"BRN_{study_id}.nii.gz"))
        supervision = nib.load(str(root / "supervision" / f"BRN_{study_id}.nii.gz"))
        if not (image.shape == label.shape == supervision.shape):
            report["errors"].append(
                f"{study_id}: shape mismatch {image.shape}/{label.shape}/{supervision.shape}"
            )
            continue
        if not (
            np.allclose(image.affine, label.affine, atol=1e-5)
            and np.allclose(image.affine, supervision.affine, atol=1e-5)
        ):
            report["errors"].append(f"{study_id}: affine mismatch")

        physical_volume = voxel_volume_ml(image.affine)
        report["min_voxel_volume_ml"] = min(float(report["min_voxel_volume_ml"]), physical_volume)
        report["max_voxel_volume_ml"] = max(float(report["max_voxel_volume_ml"]), physical_volume)
        labels = np.asarray(label.dataobj, dtype=np.uint8)
        known = np.asarray(supervision.dataobj, dtype=np.uint8)
        if not set(np.unique(labels)).issubset(set(range(6))):
            report["errors"].append(f"{study_id}: invalid label values")
        if not set(np.unique(known)).issubset({0, 1}):
            report["errors"].append(f"{study_id}: invalid supervision values")
        if row.supervision_type == "clean_negative":
            if labels.any() or not known.all():
                report["errors"].append(f"{study_id}: invalid clean-negative target")
        unknown_foreground = int(np.count_nonzero((known == 0) & (labels > 0)))
        report["unknown_foreground_voxels"] = int(report["unknown_foreground_voxels"]) + unknown_foreground
        report["foreground_voxels"] = int(report["foreground_voxels"]) + int(np.count_nonzero(labels))
        known_per_slice = np.max(known, axis=(0, 1)).astype(bool)
        report["known_slices"] = int(report["known_slices"]) + int(known_per_slice.sum())
        report["unknown_slices"] = int(report["unknown_slices"]) + int((~known_per_slice).sum())
        if int(known_per_slice.sum()) != int(row.known_slices):
            report["errors"].append(f"{study_id}: known-slice manifest mismatch")

    report["valid"] = not report["errors"]
    output = args.output or (root / "integrity_audit.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
