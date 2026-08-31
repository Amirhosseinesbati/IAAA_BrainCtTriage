"""ICH-only evaluation for direct 2.5D segmentation and physical volumes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.strategies.ich_v2.evaluation import VOLUME_KEYS

from .cache import OUTPUT_LABELS


VOLUME_TO_LABEL = {
    "V_IVH": "IVH",
    "V_IPH": "IPH",
    "V_SDH": "SDH",
    "V_EDH": "EDH",
    "V_SAH": "SAH",
}
LABEL_TO_CLASS_ID = {label: index for index, label in enumerate(OUTPUT_LABELS[1:], start=1)}


def _safe_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(truth)) < 2:
        return None
    return float(roc_auc_score(truth, scores))


def summarize_segmentation_predictions(
    slice_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return study volumes and metrics without MLS, fracture, or triage context."""
    required = {
        "study_id", "known", "voxel_volume_ml", "prob_any_ich",
        *{f"prob_{label}" for label in OUTPUT_LABELS[1:]},
        *{f"pred_pixels_{label}" for label in OUTPUT_LABELS[1:]},
        *{f"intersection_{label}" for label in OUTPUT_LABELS[1:]},
        *{f"predicted_known_pixels_{label}" for label in OUTPUT_LABELS[1:]},
        *{f"observed_known_pixels_{label}" for label in OUTPUT_LABELS[1:]},
    }
    missing = required - set(slice_predictions)
    if missing:
        raise ValueError(f"Segmentation predictions are missing: {sorted(missing)}")
    truth_required = {"study_id", *{f"gt_{key}" for key in VOLUME_KEYS}}
    truth_missing = truth_required - set(ground_truth)
    if truth_missing:
        raise ValueError(f"ICH ground truth is missing: {sorted(truth_missing)}")

    frame = slice_predictions.copy()
    frame["study_id"] = frame["study_id"].astype(str)
    rows: list[dict[str, object]] = []
    for study_id, group in frame.groupby("study_id", sort=True):
        row: dict[str, object] = {
            "study_id": str(study_id),
            "score_any_ich": float(group["prob_any_ich"].max()),
        }
        for volume_key, label in VOLUME_TO_LABEL.items():
            row[f"score_{label}"] = float(group[f"prob_{label}"].max())
            row[f"pred_{volume_key}"] = float(
                (
                    group[f"pred_pixels_{label}"].to_numpy(dtype=np.float64)
                    * group["voxel_volume_ml"].to_numpy(dtype=np.float64)
                ).sum()
            )
        rows.append(row)
    studies = pd.DataFrame(rows).merge(
        ground_truth.loc[:, sorted(truth_required)],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    if len(studies) != frame["study_id"].nunique():
        raise ValueError("Predicted studies and ICH ground truth do not match")

    dice: dict[str, float | None] = {}
    for label in OUTPUT_LABELS[1:]:
        intersection = float(frame[f"intersection_{label}"].sum())
        predicted = float(frame[f"predicted_known_pixels_{label}"].sum())
        observed = float(frame[f"observed_known_pixels_{label}"].sum())
        dice[label] = None if observed <= 0 else (2.0 * intersection) / max(
            1.0, predicted + observed
        )
    available_dice = [value for value in dice.values() if value is not None]

    gt_total = studies[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    pred_total = studies[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    gt_any = gt_total > 0.0
    pred_any = pred_total >= 0.1
    any_auc = _safe_auc(gt_any.astype(np.uint8), studies["score_any_ich"].to_numpy(float))
    subtype_summary: dict[str, dict[str, float | int | None]] = {}
    subtype_aucs: list[float] = []
    for volume_key, label in VOLUME_TO_LABEL.items():
        truth = studies[f"gt_{volume_key}"].to_numpy(float)
        predicted = studies[f"pred_{volume_key}"].to_numpy(float)
        auc = _safe_auc((truth > 0).astype(np.uint8), studies[f"score_{label}"].to_numpy(float))
        if auc is not None:
            subtype_aucs.append(auc)
        subtype_summary[label] = {
            "dice_known_pixels": dice[label],
            "study_auc": auc,
            "positive_studies": int(np.count_nonzero(truth > 0)),
            "mae_ml": float(np.mean(np.abs(predicted - truth))),
            "bias_ml": float(np.mean(predicted - truth)),
            "spearman_volume": float(pd.Series(predicted).corr(pd.Series(truth), method="spearman"))
            if len(np.unique(truth)) > 1 and len(np.unique(predicted)) > 1
            else None,
        }

    mean_dice = float(np.mean(available_dice)) if available_dice else 0.0
    macro_subtype_auc = float(np.mean(subtype_aucs)) if subtype_aucs else 0.0
    selection_score = (
        0.55 * mean_dice
        + 0.30 * float(any_auc or 0.0)
        + 0.15 * macro_subtype_auc
    )
    summary: dict[str, Any] = {
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "studies": int(len(studies)),
        "known_slices": int(frame["known"].sum()),
        "mean_foreground_dice": mean_dice,
        "any_ich_study_auc": any_auc,
        "macro_subtype_study_auc": macro_subtype_auc,
        "selection_score": selection_score,
        "presence_f1_at_0_1ml": float(f1_score(gt_any, pred_any, zero_division=0)),
        "normal_false_positive_rate_at_0_1ml": float(
            np.mean(pred_any[~gt_any]) if np.any(~gt_any) else 0.0
        ),
        "total_volume_mae_ml": float(np.mean(np.abs(pred_total - gt_total))),
        "total_volume_bias_ml": float(np.mean(pred_total - gt_total)),
        "subtypes": subtype_summary,
    }
    return studies, summary
