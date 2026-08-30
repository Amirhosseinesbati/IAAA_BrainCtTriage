"""Verify fracture MIL cache integrity and detector-score inference parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.fracture.pooling import aggregate_study_scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_study_predictions(
    slices: pd.DataFrame,
    slice_scores: np.ndarray,
    validation_fold: int,
) -> pd.DataFrame:
    if len(slices) != len(slice_scores):
        raise ValueError("Slice table and score cache lengths differ")
    selected = slices.loc[slices["outer_fold"].eq(validation_fold)].copy()
    if selected.empty:
        raise ValueError(f"No studies found for validation fold {validation_fold}")
    rows: list[dict[str, object]] = []
    for study_id, group in selected.groupby("study_id", sort=True):
        ordered = group.sort_values("slice_index", kind="stable")
        truth_values = ordered["study_fracture"].unique()
        if truth_values.size != 1:
            raise ValueError(f"Inconsistent truth for study {study_id}")
        pooled = aggregate_study_scores(slice_scores[ordered.index.to_numpy()])
        rows.append(
            {
                "study_id": str(study_id),
                "truth": int(truth_values[0]),
                **{f"prob_{name}": value for name, value in pooled.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.cache / "manifest.json"
    slices_path = args.cache / "slices.csv"
    embeddings_path = args.cache / "embeddings.npy"
    scores_path = args.cache / "slice_scores.npy"
    for path in (manifest_path, slices_path, embeddings_path, scores_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_artifacts = manifest.get("artifacts", {})
    actual_hashes = {
        "slices.csv": _sha256(slices_path),
        "embeddings.npy": _sha256(embeddings_path),
        "slice_scores.npy": _sha256(scores_path),
    }
    if actual_hashes != expected_artifacts:
        raise RuntimeError(
            f"Cache artifact hashes differ: expected={expected_artifacts}, "
            f"actual={actual_hashes}"
        )

    slices = pd.read_csv(slices_path, dtype={"study_id": str, "patient_id": str})
    embeddings = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
    slice_scores = np.load(scores_path, allow_pickle=False)
    expected_shape = tuple(manifest["embedding_shape"])
    if embeddings.shape != expected_shape:
        raise RuntimeError(f"Embedding shape {embeddings.shape} != {expected_shape}")
    if embeddings.shape[0] != len(slices) or slice_scores.shape != (len(slices),):
        raise RuntimeError("Cache array lengths do not match slices.csv")
    if not np.isfinite(embeddings).all() or not np.isfinite(slice_scores).all():
        raise RuntimeError("Cache contains non-finite values")

    actual = build_study_predictions(slices, slice_scores, args.validation_fold)
    reference = pd.read_csv(args.reference_predictions, dtype={"study_id": str})
    reference = reference.sort_values("study_id").reset_index(drop=True)
    if not actual["study_id"].equals(reference["study_id"]):
        raise RuntimeError("Cache and reference study IDs differ")
    if not actual["truth"].equals(reference["truth"].astype(np.int64)):
        raise RuntimeError("Cache and reference truth labels differ")
    score_columns = [column for column in actual.columns if column.startswith("prob_")]
    missing = set(score_columns).difference(reference.columns)
    if missing:
        raise ValueError(f"Reference predictions are missing columns: {sorted(missing)}")
    maximum_difference = {
        column: float(
            np.max(
                np.abs(
                    actual[column].to_numpy(dtype=np.float64)
                    - reference[column].to_numpy(dtype=np.float64)
                )
            )
        )
        for column in score_columns
    }
    if max(maximum_difference.values(), default=0.0) > args.tolerance:
        raise RuntimeError(
            f"Inference parity exceeded tolerance {args.tolerance}: {maximum_difference}"
        )
    payload = {
        "cache": str(args.cache),
        "validation_fold": args.validation_fold,
        "reference_predictions": str(args.reference_predictions),
        "n_slices": len(slices),
        "n_studies": len(actual),
        "embedding_shape": list(embeddings.shape),
        "artifact_sha256": actual_hashes,
        "maximum_absolute_score_difference": maximum_difference,
        "tolerance": args.tolerance,
        "status": "passed",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
