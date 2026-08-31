"""Calibrate subtype mask thresholds without touching the outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, file_sha256
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    load_segmentation_manifest,
    split_segmentation_slices,
)
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


DEFAULT_THRESHOLDS = (
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)


def _loader(frame: pd.DataFrame, *, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        ICHAdjacentSegmentationDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _scan_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    threshold_tensor = torch.as_tensor(
        thresholds, dtype=torch.float32, device=device
    )[:, None, None, None]
    intersections = torch.zeros(
        (len(OUTPUT_LABELS) - 1, len(thresholds)), device=device
    )
    predicted = torch.zeros_like(intersections)
    observed = torch.zeros(len(OUTPUT_LABELS) - 1, device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            known = batch["segmentation_known"].to(device, non_blocking=True) > 0.5
            if not torch.any(known):
                continue
            images = batch["image"].to(device, non_blocking=True)[known]
            masks = batch["mask"].to(device, non_blocking=True)[known]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_logits, _ = model(images)
            probabilities = torch.softmax(mask_logits.float(), dim=1)
            for class_index in range(1, len(OUTPUT_LABELS)):
                truth = masks == class_index
                candidates = probabilities[:, class_index].unsqueeze(0) >= threshold_tensor
                intersections[class_index - 1] += (
                    candidates & truth.unsqueeze(0)
                ).sum(dim=(1, 2, 3))
                predicted[class_index - 1] += candidates.sum(dim=(1, 2, 3))
                observed[class_index - 1] += truth.sum()

    rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for label_index, label in enumerate(OUTPUT_LABELS[1:]):
        observed_pixels = float(observed[label_index].cpu())
        if observed_pixels <= 0:
            raise ValueError(f"Calibration fold contains no spatial pixels for {label}")
        dice = (
            2.0 * intersections[label_index]
            / (predicted[label_index] + observed[label_index]).clamp_min(1.0)
        ).cpu().numpy()
        best_index = max(
            range(len(thresholds)),
            key=lambda index: (float(dice[index]), float(thresholds[index])),
        )
        selected[label] = float(thresholds[best_index])
        for index, threshold in enumerate(thresholds):
            rows.append({
                "label": label,
                "threshold": float(threshold),
                "dice": float(dice[index]),
                "intersection_pixels": int(intersections[label_index, index].cpu()),
                "predicted_pixels": int(predicted[label_index, index].cpu()),
                "observed_pixels": int(observed_pixels),
                "selected": int(index == best_index),
            })
    return selected, pd.DataFrame(rows)


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    thresholds: dict[str, float],
    *,
    device: torch.device,
) -> pd.DataFrame:
    threshold_tensor = torch.as_tensor(
        [thresholds[label] for label in OUTPUT_LABELS[1:]],
        dtype=torch.float32,
        device=device,
    )[None, :, None, None]
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_logits, class_logits = model(images)
            probabilities = torch.softmax(mask_logits.float(), dim=1)[:, 1:]
            normalized = probabilities / threshold_tensor
            best_score, best_class = normalized.max(dim=1)
            predicted_masks = torch.where(
                best_score >= 1.0,
                best_class + 1,
                torch.zeros_like(best_class),
            ).cpu()
            class_probabilities = torch.sigmoid(class_logits.float()).cpu().numpy()
            true_masks = batch["mask"]
            known = batch["segmentation_known"].numpy()
            voxel_volumes = batch["voxel_volume_ml"].numpy()
            slice_indices = batch["slice_index"].numpy()
            for index, study_id in enumerate(batch["study_id"]):
                predicted_mask = predicted_masks[index]
                observed_mask = true_masks[index]
                is_known = bool(known[index] > 0.5)
                row: dict[str, object] = {
                    "study_id": str(study_id),
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_index": int(slice_indices[index]),
                    "known": int(is_known),
                    "voxel_volume_ml": float(voxel_volumes[index]),
                }
                for output_index, label in enumerate(OUTPUT_LABELS):
                    row[f"prob_{label}"] = float(
                        class_probabilities[index, output_index]
                    )
                for class_index, label in enumerate(OUTPUT_LABELS[1:], start=1):
                    predicted_class = predicted_mask == class_index
                    row[f"pred_pixels_{label}"] = int(predicted_class.sum())
                    if is_known:
                        observed_class = observed_mask == class_index
                        row[f"intersection_{label}"] = int(
                            (predicted_class & observed_class).sum()
                        )
                        row[f"predicted_known_pixels_{label}"] = int(
                            predicted_class.sum()
                        )
                        row[f"observed_known_pixels_{label}"] = int(
                            observed_class.sum()
                        )
                    else:
                        row[f"intersection_{label}"] = 0
                        row[f"predicted_known_pixels_{label}"] = 0
                        row[f"observed_known_pixels_{label}"] = 0
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest-path", default="Data/processed/ich_2p5d/slice_manifest.csv"
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Mask threshold calibration requires CUDA BF16")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    model = build_segmentation_model(
        architecture=str(config.get("architecture", "unetplusplus")),
        encoder_name=str(config.get("encoder_name", "efficientnet-b2")),
        pretrained=False,
        dropout=float(config.get("dropout", 0.2)),
    )
    load_segmentation_weights(model, args.checkpoint)
    device = torch.device("cuda")
    model.to(device).eval()

    manifest = load_segmentation_manifest(args.manifest_path)
    _, calibration, outer = split_segmentation_slices(
        manifest,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
    )
    calibration_loader = _loader(
        calibration, batch_size=args.batch_size, workers=args.workers
    )
    outer_loader = _loader(outer, batch_size=args.batch_size, workers=args.workers)
    threshold_values = np.asarray(DEFAULT_THRESHOLDS, dtype=np.float32)
    selected, curves = _scan_thresholds(
        model, calibration_loader, threshold_values, device=device
    )

    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    calibration_predictions = _predict(
        model, calibration_loader, selected, device=device
    )
    outer_predictions = _predict(model, outer_loader, selected, device=device)
    calibration_studies, calibration_summary = summarize_segmentation_predictions(
        calibration_predictions, truth
    )
    outer_studies, outer_summary = summarize_segmentation_predictions(
        outer_predictions, truth
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_dir / "calibration_threshold_curves.csv", index=False)
    calibration_predictions.to_csv(
        args.output_dir / "calibration_slice_predictions.csv", index=False
    )
    calibration_studies.to_csv(
        args.output_dir / "calibration_study_predictions.csv", index=False
    )
    outer_predictions.to_csv(args.output_dir / "outer_slice_predictions.csv", index=False)
    outer_studies.to_csv(args.output_dir / "outer_study_predictions.csv", index=False)
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest": str(args.manifest_path),
        "manifest_sha256": file_sha256(args.manifest_path),
        "metadata_source": str(metadata_source),
        "selection_policy": (
            "per-subtype softmax thresholds maximize pixel Dice on calibration fold only; "
            "overlaps resolve by maximum probability-to-threshold ratio"
        ),
        "thresholds": selected,
        "calibration_summary": calibration_summary,
        "outer_summary": outer_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
