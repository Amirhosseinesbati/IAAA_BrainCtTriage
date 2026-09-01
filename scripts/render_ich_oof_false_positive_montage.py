"""Render the most severe OOF false-positive ICH studies for label QA.

The stored prediction tables contain pixel counts rather than masks, so this
montage shows the three registered CT windows at the slice with the largest
predicted foreground mass.  It is intended to detect label defects and common
mimics before those studies are used for hard-negative mining.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compare_ich_2p5d_segmentation_oof import _load_variant
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_v2.evaluation import ground_truth_ich_context


WINDOW_NAMES = ("brain", "subdural", "bone/context")


def _select_false_positive_slices(
    study_errors: pd.DataFrame,
    slices: pd.DataFrame,
    *,
    top_studies: int,
) -> pd.DataFrame:
    false_positive = (
        study_errors.loc[study_errors["presence_error"] == "false_positive"]
        .sort_values(["pred_total_ml", "study_id"], ascending=[False, True])
        .head(top_studies)
        .copy()
    )
    if false_positive.empty:
        raise ValueError("No false-positive studies were found")
    slices = slices.copy()
    slices["study_id"] = slices["study_id"].astype(str)
    pixel_columns = [f"pred_pixels_{label}" for label in OUTPUT_LABELS[1:]]
    slices["predicted_foreground_pixels"] = slices[pixel_columns].sum(axis=1)
    selected = (
        slices[slices["study_id"].isin(false_positive["study_id"].astype(str))]
        .sort_values(
            ["study_id", "predicted_foreground_pixels", "slice_index"],
            ascending=[True, False, True],
        )
        .drop_duplicates("study_id", keep="first")
    )
    keep = [
        "study_id",
        "patient_id",
        "outer_fold",
        "slice_index",
        "predicted_foreground_pixels",
    ]
    return false_positive.merge(
        selected[keep],
        on="study_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_slice"),
    ).sort_values(["pred_total_ml", "study_id"], ascending=[False, True])


def _render_page(
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    destination: Path,
) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(
        len(rows),
        len(WINDOW_NAMES),
        figsize=(11.5, max(3.0, 3.1 * len(rows))),
        squeeze=False,
    )
    rendered: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        matches = manifest.loc[
            (manifest["study_id"] == str(row.study_id))
            & (manifest["slice_index"] == int(row.slice_index))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one manifest row for study={row.study_id}, "
                f"slice={row.slice_index}; got {len(matches)}"
            )
        manifest_row = matches.iloc[0]
        cached = np.load(str(manifest_row["cache_path"]), mmap_mode="r")
        image = np.asarray(cached[int(row.slice_index)])
        if image.shape[0] != len(WINDOW_NAMES):
            raise ValueError(f"Unexpected cached window shape: {image.shape}")
        for window_index, window_name in enumerate(WINDOW_NAMES):
            axis = axes[row_index, window_index]
            axis.imshow(image[window_index], cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(window_name, fontsize=10)
        axes[row_index, 0].text(
            -0.03,
            0.5,
            f"study {row.study_id} | fold {int(row.outer_fold)}\n"
            f"pred {float(row.pred_total_ml):.2f} mL | "
            f"Any {float(row.score_any_ich):.3f}\n"
            f"slice {int(row.slice_index)} | {row.dominant_predicted_subtype}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        rendered.append({
            "study_id": str(row.study_id),
            "patient_id": str(row.patient_id),
            "outer_fold": int(row.outer_fold),
            "slice_index": int(row.slice_index),
            "predicted_total_volume_ml": float(row.pred_total_ml),
            "score_any_ich": float(row.score_any_ich),
            "dominant_predicted_subtype": str(row.dominant_predicted_subtype),
            "predicted_foreground_pixels_on_rendered_slice": int(
                row.predicted_foreground_pixels
            ),
            "cache_path": str(manifest_row["cache_path"]),
        })
    figure.suptitle(
        "Standalone ICH incumbent: severe patient-disjoint OOF false positives\n"
        "Displayed slice maximizes predicted foreground pixels; no prediction mask stored",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.97))
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--study-error-table", required=True, type=Path)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("Data/processed/ich_2p5d/slice_manifest.csv"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--top-studies", type=int, default=12)
    parser.add_argument("--studies-per-page", type=int, default=6)
    args = parser.parse_args()
    if args.top_studies <= 0 or args.studies_per_page <= 0:
        raise ValueError("top-studies and studies-per-page must be positive")

    truth, metadata_source = ground_truth_ich_context()
    variant = _load_variant(
        "hardpixel_fprselect_incumbent",
        args.run_dir,
        truth,
        args.expected_studies,
    )
    errors = pd.read_csv(args.study_error_table, dtype={"study_id": str})
    selected = _select_false_positive_slices(
        errors,
        variant.slices,
        top_studies=args.top_studies,
    )
    manifest = pd.read_csv(
        args.manifest_path,
        dtype={"study_id": str, "patient_id": str},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    pages = math.ceil(len(selected) / args.studies_per_page)
    for page in range(pages):
        start = page * args.studies_per_page
        stop = start + args.studies_per_page
        destination = args.output_dir / f"false_positive_montage_page_{page + 1:02d}.png"
        rendered.extend(_render_page(selected.iloc[start:stop], manifest, destination))
    payload = {
        "analysis_kind": "label_qa_before_oof_hard_negative_mining",
        "metadata_source": str(metadata_source),
        "study_error_table": str(args.study_error_table),
        "manifest_path": str(args.manifest_path),
        "studies_rendered": len(rendered),
        "pages": pages,
        "limitation": (
            "Prediction masks were not persisted. The displayed slice is selected by "
            "predicted foreground pixel count, but the montage contains CT windows only."
        ),
        "rows": rendered,
    }
    (args.output_dir / "montage_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
