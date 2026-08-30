"""Build a leakage-safe YOLO train list from held-out fracture error evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _top_context_indices(
    scores: np.ndarray, top_k: int, radius: int
) -> list[int]:
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite vector")
    if top_k < 1 or radius < 0:
        raise ValueError("top_k must be positive and radius non-negative")
    anchors = np.argsort(-scores, kind="stable")[: min(top_k, scores.size)]
    selected: set[int] = set()
    for anchor in anchors:
        selected.update(
            range(max(0, int(anchor) - radius), min(scores.size, int(anchor) + radius + 1))
        )
    return sorted(selected)


def _label_for_image(path: Path) -> Path:
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images directory: {path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fold", type=int, required=True)
    parser.add_argument("--folds-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-negative-studies", type=int, default=40)
    parser.add_argument("--hard-negative-top-k", type=int, default=3)
    parser.add_argument("--hard-negative-context", type=int, default=1)
    parser.add_argument("--hard-positive-extra-repeat", type=int, default=2)
    args = parser.parse_args()

    if args.target_fold not in range(5):
        raise ValueError("target-fold must be in [0, 4]")
    if args.hard_negative_studies < 1 or args.hard_positive_extra_repeat < 0:
        raise ValueError("Invalid mining counts")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    base = args.folds_root / f"fold_{args.target_fold}"
    marker_path = base / ".fracture_dataset_v2.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(
        args.oof_predictions, dtype={"study_id": str, "patient_id": str}
    )
    if predictions["study_id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate studies")
    development = predictions.loc[predictions["outer_fold"].ne(args.target_fold)].copy()
    if development.empty:
        raise ValueError("Mining development set is empty")
    # Multiple studies from one patient are allowed, but every patient must stay
    # in one outer fold and therefore remain outside the target validation fold.
    patient_folds = predictions.groupby("patient_id")["outer_fold"].nunique()
    if not patient_folds.eq(1).all():
        raise RuntimeError("A patient crosses outer folds in mining predictions")
    if development["outer_fold"].eq(args.target_fold).any():
        raise RuntimeError("Target validation studies entered the mining pool")

    cache_tables: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    fold_manifests: dict[int, pd.DataFrame] = {}
    for fold in range(5):
        cache_directory = args.cache_root / f"fold_{fold}"
        cache_tables[fold] = (
            pd.read_csv(cache_directory / "slices.csv", dtype={"study_id": str}),
            np.load(cache_directory / "slice_scores.npy", allow_pickle=False),
        )
        fold_manifests[fold] = pd.read_csv(
            args.folds_root / f"fold_{fold}" / "manifest.csv", dtype={"study_id": str}
        )

    negative_studies = (
        development.loc[development["truth"].eq(0)]
        .sort_values("deployable_blend_score", ascending=False, kind="stable")
        .head(args.hard_negative_studies)
    )
    hard_negative_entries: list[str] = []
    mining_rows: list[dict[str, object]] = []
    for row in negative_studies.itertuples(index=False):
        source_fold = int(row.outer_fold)
        slices, scores = cache_tables[source_fold]
        study = slices.loc[slices["study_id"].eq(str(row.study_id))].sort_values(
            "slice_index", kind="stable"
        )
        study_scores = scores[study.index.to_numpy(dtype=np.int64)]
        positions = _top_context_indices(
            study_scores, args.hard_negative_top_k, args.hard_negative_context
        )
        for position in positions:
            image = Path(str(study.iloc[position]["image_path"])).resolve()
            label = _label_for_image(image)
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(image if not image.is_file() else label)
            if label.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"Hard-negative label is not empty: {label}")
            hard_negative_entries.append(image.as_posix())
            mining_rows.append(
                {
                    "target_fold": args.target_fold,
                    "source_outer_fold": source_fold,
                    "study_id": str(row.study_id),
                    "truth": 0,
                    "mining_role": "hard_negative",
                    "oof_study_score": float(row.deployable_blend_score),
                    "slice_index": int(study.iloc[position]["slice_index"]),
                    "slice_score": float(study_scores[position]),
                    "image_path": image.as_posix(),
                }
            )

    false_negatives = development.loc[
        development["truth"].eq(1) & development["candidate_binary"].eq(0)
    ]
    hard_positive_entries: list[str] = []
    for row in false_negatives.itertuples(index=False):
        source_fold = int(row.outer_fold)
        manifest = fold_manifests[source_fold]
        positives = manifest.loc[
            manifest["study_id"].eq(str(row.study_id))
            & manifest["split"].eq("val")
            & manifest["slice_fracture"].eq(1)
        ]
        if positives.empty:
            raise RuntimeError(f"False-negative study has no positive slice: {row.study_id}")
        source_root = args.folds_root / f"fold_{source_fold}"
        for positive in positives.itertuples(index=False):
            image = (source_root / str(positive.image)).resolve()
            label = _label_for_image(image)
            if not image.is_file() or not label.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"Hard-positive image/label invalid: {image}")
            for _ in range(args.hard_positive_extra_repeat):
                hard_positive_entries.append(image.as_posix())
            mining_rows.append(
                {
                    "target_fold": args.target_fold,
                    "source_outer_fold": source_fold,
                    "study_id": str(row.study_id),
                    "truth": 1,
                    "mining_role": "hard_positive",
                    "oof_study_score": float(row.deployable_blend_score),
                    "slice_index": int(positive.slice_index),
                    "slice_score": None,
                    "image_path": image.as_posix(),
                }
            )

    base_entries = [
        line.strip() for line in (base / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    combined = base_entries + hard_negative_entries + hard_positive_entries
    train_path = (args.output / "train.txt").resolve()
    train_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    (args.output / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {base.resolve().as_posix()}",
                f"train: {train_path.as_posix()}",
                "val: images/val",
                "names:",
                "  0: fracture",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name in ("manifest.csv", "studies.csv"):
        shutil.copy2(base / name, args.output / name)
    for name in ("images", "labels"):
        os.symlink(base / name, args.output / name, target_is_directory=True)

    marker["config"] = {
        **marker.get("config", {}),
        "hardmine_schema_version": 1,
        "hard_negative_studies": args.hard_negative_studies,
        "hard_negative_top_k": args.hard_negative_top_k,
        "hard_negative_context": args.hard_negative_context,
        "hard_positive_extra_repeat": args.hard_positive_extra_repeat,
        "mining_source": "patient_disjoint_outer_fold_oof_predictions",
    }
    marker["hardmine"] = {
        "base_dataset": str(base.resolve()),
        "base_entries": len(base_entries),
        "hard_negative_entries": len(hard_negative_entries),
        "hard_positive_entries": len(hard_positive_entries),
        "total_entries": len(combined),
        "selected_negative_studies": int(negative_studies["study_id"].nunique()),
        "selected_false_negative_studies": int(false_negatives["study_id"].nunique()),
        "leakage_gate": "passed_outer_fold_and_patient_disjoint",
    }
    (args.output / ".fracture_dataset_v2.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame.from_records(mining_rows).to_csv(
        args.output / "private_mining_manifest.csv", index=False
    )
    print(json.dumps(marker["hardmine"], indent=2))


if __name__ == "__main__":
    main()
