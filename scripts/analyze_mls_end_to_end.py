"""Decompose MLS study-level error into slice retrieval and measurement error.

Only DICOM headers are read (``stop_before_pixels=True``).  No model forward,
pixel decoding, or other heavy CPU computation is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "Data/raw/training")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _uid_from_image_name(study_id: str, image_name: str) -> str:
    prefix = f"{study_id}_"
    value = Path(image_name).stem
    return value[len(prefix):] if value.startswith(prefix) else value


def _sorted_uids(study_dir: Path) -> list[str]:
    headers: list[tuple[float, str]] = []
    for path in study_dir.glob("*.dcm"):
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        position = getattr(dataset, "ImagePositionPatient", [0.0, 0.0, 0.0])
        headers.append((float(position[2]), str(getattr(dataset, "SOPInstanceUID", ""))))
    headers.sort(key=lambda item: item[0])
    return [uid for _, uid in headers]


def _aggregate(values: np.ndarray, mode: str) -> float:
    if mode == "median":
        return float(np.median(values))
    if mode == "p90":
        return float(np.percentile(values, 90))
    if mode == "max":
        return float(np.max(values))
    raise ValueError(mode)


def _study_prediction(items: list[dict], *, top_k: int, mode: str, threshold: float = 0.5) -> float:
    ranked = sorted(items, key=lambda item: item["selector_probability"], reverse=True)
    if not ranked or float(ranked[0]["selector_probability"]) < threshold:
        return 0.1
    return _aggregate(
        np.asarray([float(item["mls_mm"]) for item in ranked[:top_k]], dtype=float), mode
    )


def _mae(frame: pd.DataFrame, column: str) -> float:
    return float(np.mean(np.abs(frame[column] - frame["gt_MLS_mm"])))


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, dtype={"study_id": str})
    labels = pd.read_csv(args.labels, dtype={"patient_id": str})
    positives = labels.loc[labels["is_target"].astype(int) == 1].copy()
    positives["sop_uid"] = [
        _uid_from_image_name(study, name)
        for study, name in zip(positives["patient_id"], positives["image_name"])
    ]
    positive_map = positives.groupby("patient_id")["sop_uid"].apply(set).to_dict()

    rows: list[dict] = []
    global_truth: list[int] = []
    global_probability: list[float] = []
    for _, source in predictions.iterrows():
        study_id = str(source["study_id"])
        items: list[dict] = json.loads(source["slice_predictions_json"])
        uids = _sorted_uids(args.raw_root / study_id)
        if len(uids) != len(items):
            raise ValueError(f"Slice count mismatch for {study_id}: {len(uids)} != {len(items)}")
        target_uids = positive_map.get(study_id, set())
        target_indices = [index for index, uid in enumerate(uids) if uid in target_uids]
        probabilities = np.asarray([float(item["selector_probability"]) for item in items])
        mls_values = np.asarray([float(item["mls_mm"]) for item in items])
        target_mask = np.zeros(len(items), dtype=bool)
        target_mask[target_indices] = True
        ranks = np.argsort(-probabilities)
        rank_positions = np.empty(len(items), dtype=int)
        rank_positions[ranks] = np.arange(1, len(items) + 1)
        best_target_rank = int(rank_positions[target_mask].min()) if target_mask.any() else np.nan
        target_values = mls_values[target_mask]
        global_truth.extend(target_mask.astype(int).tolist())
        global_probability.extend(probabilities.tolist())

        row = {
            "study_id": study_id,
            "gt_MLS_mm": float(source["gt_MLS_mm"]),
            "n_slices": len(items),
            "n_target_slices": int(target_mask.sum()),
            "best_target_rank": best_target_rank,
            "selector_target_mean": float(probabilities[target_mask].mean()) if target_mask.any() else np.nan,
            "selector_nontarget_mean": float(probabilities[~target_mask].mean()),
            "selector_target_max": float(probabilities[target_mask].max()) if target_mask.any() else np.nan,
            "selector_nontarget_max": float(probabilities[~target_mask].max()),
            "top3_target_count": int(target_mask[ranks[:3]].sum()),
            "top5_target_count": int(target_mask[ranks[:5]].sum()),
            "oracle_target_median": _aggregate(target_values, "median") if target_mask.any() else 0.1,
            "oracle_target_p90": _aggregate(target_values, "p90") if target_mask.any() else 0.1,
            "oracle_target_max": _aggregate(target_values, "max") if target_mask.any() else 0.1,
            "pred_top3_median": _study_prediction(items, top_k=3, mode="median"),
            "pred_top3_p90": _study_prediction(items, top_k=3, mode="p90"),
            "pred_top5_median": _study_prediction(items, top_k=5, mode="median"),
        }
        for key in (
            "oracle_target_median", "oracle_target_p90", "oracle_target_max",
            "pred_top3_median", "pred_top3_p90", "pred_top5_median",
        ):
            row[f"abs_error_{key}"] = abs(row[key] - row["gt_MLS_mm"])
        rows.append(row)

    detail = pd.DataFrame(rows)
    annotated = detail["n_target_slices"] > 0
    summary = {
        "n_studies": int(len(detail)),
        "n_slices": int(detail["n_slices"].sum()),
        "slice_selector_auc_annotated_targets_vs_other": float(
            roc_auc_score(global_truth, global_probability)
        ),
        "target_retrieval_recall": {
            f"top_{top_k}": float((detail.loc[annotated, "best_target_rank"] <= top_k).mean())
            for top_k in (1, 3, 5, 10)
        },
        "studies_with_selector_gate_miss": int(
            (detail[["selector_target_max", "selector_nontarget_max"]].max(axis=1) < 0.5).sum()
        ),
        "mean_target_slices_in_top3": float(detail["top3_target_count"].mean()),
        "mean_target_slices_in_top5": float(detail["top5_target_count"].mean()),
        "mae_mm": {
            key: _mae(detail, key)
            for key in (
                "oracle_target_median", "oracle_target_p90", "oracle_target_max",
                "pred_top3_median", "pred_top3_p90", "pred_top5_median",
            )
        },
        "error_attribution_mm": {
            "measurement_floor_oracle_target_median": _mae(detail, "oracle_target_median"),
            "additional_top3_median_retrieval_and_pooling": (
                _mae(detail, "pred_top3_median") - _mae(detail, "oracle_target_median")
            ),
        },
        "largest_top3_median_errors": detail.nlargest(
            10, "abs_error_pred_top3_median"
        )[[
            "study_id", "gt_MLS_mm", "pred_top3_median", "best_target_rank",
            "top3_target_count", "selector_target_max", "selector_nontarget_max",
            "abs_error_pred_top3_median",
        ]].to_dict(orient="records"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "selector_measurement_decomposition.csv", index=False)
    (args.output_dir / "decomposition.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    recall = summary["target_retrieval_recall"]
    mae = summary["mae_mm"]
    report = f"""# MLS end-to-end error decomposition

- Studies: `{summary['n_studies']}`; slices: `{summary['n_slices']}`
- Slice selector AUC (annotated targets vs other slices): `{summary['slice_selector_auc_annotated_targets_vs_other']:.4f}`
- Target retrieval recall: top-1 `{recall['top_1']:.3f}`, top-3 `{recall['top_3']:.3f}`, top-5 `{recall['top_5']:.3f}`, top-10 `{recall['top_10']:.3f}`
- Mean annotated target slices inside selected top-3/top-5: `{summary['mean_target_slices_in_top3']:.2f}` / `{summary['mean_target_slices_in_top5']:.2f}`
- Selector gate misses at threshold 0.5: `{summary['studies_with_selector_gate_miss']}`

## MAE decomposition

| profile | MAE mm |
|---|---:|
| Oracle annotated targets, median | {mae['oracle_target_median']:.4f} |
| Oracle annotated targets, p90 | {mae['oracle_target_p90']:.4f} |
| Predicted selector top-3, median | {mae['pred_top3_median']:.4f} |
| Predicted selector top-3, p90 | {mae['pred_top3_p90']:.4f} |
| Predicted selector top-5, median | {mae['pred_top5_median']:.4f} |

The oracle-target median is the approximate keypoint/measurement floor.  The
difference to predicted top-3 median estimates the added selector/pooling cost.
This is diagnostic on fold 0 and must not be treated as an independent test.
"""
    (args.output_dir / "decomposition_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "largest_top3_median_errors"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
