"""Audit MLS slice targets against the study-level competition target.

This is a lightweight metadata/CSV analysis.  It never loads images and never
runs a model.  The output is intended to make future MLS experiment design
reproducible instead of repeatedly rediscovering the target structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAINING_CSV_PATH
from src.evaluation.splits import normalize_study_id


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "min": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _mae(frame: pd.DataFrame, prediction: str) -> float:
    return float(np.mean(np.abs(frame[prediction] - frame["study_mls_mm"])))


def _render(summary: dict) -> str:
    count = summary["annotation_count_per_target_study"]
    pooling = summary["local_geometry_to_study_target"]
    lines = [
        "# MLS training-target audit",
        "",
        "This audit uses only label CSVs and geometry; no image or model was loaded.",
        "",
        "## Dataset contract",
        "",
        f"- Rows: `{summary['rows']}`; studies: `{summary['studies']}`.",
        f"- Target rows: `{summary['target_rows']}` across `{summary['target_studies']}` studies.",
        f"- Negative rows: `{summary['negative_rows']}` across `{summary['negative_studies']}` studies.",
        f"- Target annotations/study range from `{count['min']:.0f}` to `{count['max']:.0f}` "
        f"(median `{count['median']:.1f}`), so uniform row sampling gives highly unequal study weight.",
        f"- Spearman correlation between annotation count and study MLS: "
        f"`{summary['annotation_count_vs_study_mls_spearman']:.3f}`.",
        "",
        "## What the slice annotations represent",
        "",
        "The study label is compared with statistics of the exact MLS reconstructed from every "
        "annotated slice's three keypoints and DICOM pixel spacing.",
        "",
        "| Slice-geometry pooling | Study MAE (mm) | Study bias (mm) |",
        "|---|---:|---:|",
    ]
    for key in ("min", "median", "p75", "p90", "max"):
        row = pooling[key]
        lines.append(f"| {key} | {row['mae_mm']:.4f} | {row['bias_mm']:+.4f} |")
    lines.extend([
        "",
        f"Best pure label-geometry statistic: `{pooling['best_by_mae']}`. This establishes whether "
        "study-level high-quantile pooling is intrinsic to the annotation contract rather than a "
        "post-processing accident.",
        "",
        "## Training implications",
        "",
        "1. Sampling should balance studies within target/non-target classes; balancing rows alone "
        "overweights heavily annotated studies.",
        "2. The heatmap head should continue learning local three-point geometry. The final study "
        "target must be formed by a robust high quantile across target slices.",
        "3. Checkpoint selection must include a study-level aggregation metric; slice MAE or selector "
        "AUC alone is insufficient.",
        "4. The documented extreme target must be handled explicitly rather than silently dominating "
        "the regression loss.",
        "",
        "## Extreme studies",
        "",
        "| Study | GT MLS | Target slices | median geometry | p90 geometry | max geometry |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["largest_study_targets"]:
        lines.append(
            f"| {row['study_id']} | {row['study_mls_mm']:.3f} | {row['target_slices']} | "
            f"{row['median']:.3f} | {row['p90']:.3f} | {row['max']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2" / "mls_labels_multitask.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "mls_experiments" / "mls_training_target_analysis",
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    labels["patient_id"] = labels["patient_id"].map(normalize_study_id)
    labels["is_target"] = pd.to_numeric(labels["is_target"], errors="raise").astype(int)
    metadata = pd.read_csv(
        TRAINING_CSV_PATH,
        usecols=["dicom_series.id", "dicom_series.PixelSpacing1", "MidlineShiftMM"],
    )
    metadata["study_id"] = metadata["dicom_series.id"].map(normalize_study_id)
    truth = metadata.groupby("study_id", as_index=False).agg(
        # Official series-level MLS is the maximum annotated slice value.
        # This matches src.evaluation.oof and the deep EDA series table.
        study_mls_mm=("MidlineShiftMM", "max"),
        metadata_spacing_x=("dicom_series.PixelSpacing1", "median"),
    )
    labels = labels.merge(
        truth, left_on="patient_id", right_on="study_id", how="left", validate="many_to_one"
    )
    spacing = pd.to_numeric(labels["spacing_x"], errors="coerce")
    labels["effective_spacing_x"] = spacing.fillna(labels["metadata_spacing_x"])
    if labels[["study_mls_mm", "effective_spacing_x"]].isna().any().any():
        raise ValueError("MLS labels could not be joined to complete study truth/spacing")

    target = labels.loc[labels["is_target"] == 1].copy()
    p1 = target[["x1", "y1"]].to_numpy(dtype=float)
    p2 = target[["x2", "y2"]].to_numpy(dtype=float)
    p3 = target[["x3", "y3"]].to_numpy(dtype=float)
    direction = p2 - p1
    numerator = np.abs(direction[:, 0] * (p1[:, 1] - p3[:, 1]) - (p1[:, 0] - p3[:, 0]) * direction[:, 1])
    denominator = np.linalg.norm(direction, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Degenerate falx line in target keypoints")
    target["slice_geometry_mls_mm"] = (
        numerator / denominator * target["effective_spacing_x"].to_numpy(dtype=float)
    )

    grouped = target.groupby("patient_id")
    study = grouped.agg(
        study_mls_mm=("study_mls_mm", "first"),
        target_slices=("slice_geometry_mls_mm", "size"),
        min=("slice_geometry_mls_mm", "min"),
        median=("slice_geometry_mls_mm", "median"),
        max=("slice_geometry_mls_mm", "max"),
    ).reset_index(names="study_id")
    for q, name in ((0.75, "p75"), (0.90, "p90")):
        quantile = grouped["slice_geometry_mls_mm"].quantile(q).rename(name)
        study = study.merge(quantile, left_on="study_id", right_index=True, validate="one_to_one")

    pooling: dict[str, dict[str, float] | str] = {}
    for name in ("min", "median", "p75", "p90", "max"):
        pooling[name] = {
            "mae_mm": _mae(study, name),
            "rmse_mm": float(np.sqrt(np.mean((study[name] - study["study_mls_mm"]) ** 2))),
            "bias_mm": float(np.mean(study[name] - study["study_mls_mm"])),
        }
    pooling["best_by_mae"] = min(
        ("min", "median", "p75", "p90", "max"),
        key=lambda name: float(pooling[name]["mae_mm"]),
    )

    non_extreme = study.loc[study["study_mls_mm"] <= 30.0]
    pooling_non_extreme: dict[str, dict[str, float] | str] = {}
    for name in ("min", "median", "p75", "p90", "max"):
        pooling_non_extreme[name] = {
            "mae_mm": _mae(non_extreme, name),
            "bias_mm": float(np.mean(non_extreme[name] - non_extreme["study_mls_mm"])),
        }
    pooling_non_extreme["best_by_mae"] = min(
        ("min", "median", "p75", "p90", "max"),
        key=lambda name: float(pooling_non_extreme[name]["mae_mm"]),
    )

    bins = pd.cut(
        study["study_mls_mm"],
        bins=[-np.inf, 1.0, 3.0, 5.0, np.inf],
        labels=["lt_1", "1_to_3", "3_to_5", "ge_5"],
        right=False,
    )
    count_by_bucket = {
        str(bucket): {
            "studies": int(len(group)),
            "target_slices_mean": float(group["target_slices"].mean()),
            "target_slices_median": float(group["target_slices"].median()),
        }
        for bucket, group in study.groupby(bins, observed=True)
    }

    largest = study.nlargest(8, "study_mls_mm").copy()
    summary = {
        "labels_path": str(args.labels.resolve()),
        "truth_path": str(TRAINING_CSV_PATH.resolve()),
        "rows": int(len(labels)),
        "studies": int(labels["patient_id"].nunique()),
        "target_rows": int((labels["is_target"] == 1).sum()),
        "negative_rows": int((labels["is_target"] == 0).sum()),
        "target_studies": int(target["patient_id"].nunique()),
        "negative_studies": int(labels.loc[labels["is_target"] == 0, "patient_id"].nunique()),
        "annotation_count_per_target_study": _quantiles(study["target_slices"]),
        "local_slice_geometry_mm": _quantiles(target["slice_geometry_mls_mm"]),
        "local_geometry_to_study_target": pooling,
        "local_geometry_to_study_target_excluding_gt_above_30mm": pooling_non_extreme,
        "annotation_count_vs_study_mls_spearman": float(
            study[["target_slices", "study_mls_mm"]].corr(method="spearman").iloc[0, 1]
        ),
        "annotation_count_by_study_mls_bucket": count_by_bucket,
        "largest_study_targets": largest.to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
