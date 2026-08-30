"""Cache full-sequence YOLO embeddings and slice scores for fracture MIL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


REQUIRED_MANIFEST_COLUMNS = {
    "study_id",
    "split",
    "slice_index",
    "image",
    "study_fracture",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_full_sequence_catalog(
    folds_root: Path,
    metadata_path: Path,
    *,
    n_folds: int = 5,
    require_images: bool = True,
) -> pd.DataFrame:
    """Union each fold's full-coverage validation rows into one patient catalog."""
    frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        fold_root = folds_root / f"fold_{fold}"
        manifest_path = fold_root / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        validation = manifest.loc[manifest["split"].eq("val")].copy()
        validation["outer_fold"] = fold
        validation["source_fold"] = fold
        validation["image_path"] = validation["image"].map(
            lambda value: str((fold_root / str(value)).resolve())
        )
        frames.append(validation)

    catalog = pd.concat(frames, ignore_index=True)
    if catalog.empty:
        raise ValueError("Full-sequence catalog is empty")
    if catalog.duplicated(["study_id", "slice_index"]).any():
        raise ValueError("Duplicate study/slice rows found across validation folds")
    fold_counts = catalog.groupby("study_id", sort=False)["outer_fold"].nunique()
    if not fold_counts.eq(1).all():
        raise ValueError("At least one study is assigned to multiple outer folds")
    truth_counts = catalog.groupby("study_id", sort=False)["study_fracture"].nunique()
    if not truth_counts.eq(1).all():
        raise ValueError("Inconsistent study labels in full-sequence catalog")

    metadata = pd.read_csv(metadata_path)
    metadata_columns = {"dicom_series.id", "dicom_series.PatientID"}
    missing = metadata_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
    study_patient = metadata[list(metadata_columns)].drop_duplicates()
    patient_counts = study_patient.groupby("dicom_series.id")[
        "dicom_series.PatientID"
    ].nunique()
    if not patient_counts.eq(1).all():
        raise ValueError("A study maps to multiple patient IDs")
    catalog = catalog.merge(
        study_patient,
        left_on="study_id",
        right_on="dicom_series.id",
        how="left",
        validate="many_to_one",
    )
    if catalog["dicom_series.PatientID"].isna().any():
        missing_studies = sorted(
            catalog.loc[catalog["dicom_series.PatientID"].isna(), "study_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Missing patient mapping for studies: {missing_studies[:10]}")
    patient_folds = catalog.groupby("dicom_series.PatientID")["outer_fold"].nunique()
    if not patient_folds.eq(1).all():
        raise ValueError("At least one patient crosses outer folds")

    catalog = catalog.rename(columns={"dicom_series.PatientID": "patient_id"})
    catalog["study_id"] = catalog["study_id"].astype(str)
    catalog["patient_id"] = catalog["patient_id"].astype(str)
    catalog["slice_index"] = catalog["slice_index"].astype(np.int64)
    catalog["study_fracture"] = catalog["study_fracture"].astype(np.int64)
    catalog = catalog.sort_values(
        ["study_id", "slice_index"], kind="stable"
    ).reset_index(drop=True)
    if require_images:
        missing_images = [path for path in catalog["image_path"] if not Path(path).is_file()]
        if missing_images:
            raise FileNotFoundError(f"Missing catalog image: {missing_images[0]}")
    return catalog[
        [
            "study_id",
            "patient_id",
            "study_fracture",
            "outer_fold",
            "slice_index",
            "source_fold",
            "image_path",
        ]
    ]


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _slice_max_score(result: object) -> float:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0
    confidence = boxes.conf.detach().float().cpu().numpy()
    return float(np.max(confidence)) if confidence.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--folds-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    catalog = build_full_sequence_catalog(args.folds_root, args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)

    # Separate predictor objects avoid state leakage between Ultralytics embed
    # and detection modes while sharing the same immutable checkpoint.
    detector = YOLO(str(args.checkpoint))
    embedder = YOLO(str(args.checkpoint))
    embedding_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    paths = catalog["image_path"].tolist()
    for start in range(0, len(paths), args.batch_size):
        stop = min(start + args.batch_size, len(paths))
        batch_paths = paths[start:stop]
        results = detector.predict(
            batch_paths,
            imgsz=args.image_size,
            conf=args.confidence,
            batch=len(batch_paths),
            device=args.device,
            verbose=False,
        )
        scores = np.asarray([_slice_max_score(result) for result in results], dtype=np.float32)
        embedded = embedder.embed(
            batch_paths,
            imgsz=args.image_size,
            batch=len(batch_paths),
            device=args.device,
            verbose=False,
        )
        features = np.stack(
            [item.detach().float().cpu().numpy() for item in embedded], axis=0
        ).astype(np.float16)
        if features.shape[0] != len(batch_paths) or scores.shape[0] != len(batch_paths):
            raise RuntimeError("Ultralytics output order/count does not match input batch")
        embedding_chunks.append(features)
        score_chunks.append(scores)
        print(f"cached {stop}/{len(paths)} slices", flush=True)

    embeddings = np.concatenate(embedding_chunks, axis=0)
    slice_scores = np.concatenate(score_chunks, axis=0)
    if embeddings.shape[0] != len(catalog) or slice_scores.shape != (len(catalog),):
        raise RuntimeError("Final feature cache length does not match catalog")
    if not np.isfinite(embeddings).all() or not np.isfinite(slice_scores).all():
        raise RuntimeError("Non-finite values found in feature cache")

    catalog_path = args.output / "slices.csv"
    embeddings_path = args.output / "embeddings.npy"
    scores_path = args.output / "slice_scores.npy"
    catalog.to_csv(catalog_path, index=False)
    _atomic_save_array(embeddings_path, embeddings)
    _atomic_save_array(scores_path, slice_scores)
    manifest = {
        "schema_version": 1,
        "outer_model_fold": args.fold,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "folds_root": str(args.folds_root.resolve()),
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": _sha256(args.metadata),
        "image_size": args.image_size,
        "confidence": args.confidence,
        "n_slices": len(catalog),
        "n_studies": int(catalog["study_id"].nunique()),
        "n_patients": int(catalog["patient_id"].nunique()),
        "n_positive_studies": int(
            catalog[["study_id", "study_fracture"]]
            .drop_duplicates()["study_fracture"]
            .sum()
        ),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "slice_score_dtype": str(slice_scores.dtype),
        "artifacts": {
            "slices.csv": _sha256(catalog_path),
            "embeddings.npy": _sha256(embeddings_path),
            "slice_scores.npy": _sha256(scores_path),
        },
    }
    manifest_path = args.output / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
