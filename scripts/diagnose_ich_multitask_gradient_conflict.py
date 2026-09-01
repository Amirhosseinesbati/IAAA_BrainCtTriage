"""Measure encoder-gradient conflict between ICH segmentation and classification.

This is a diagnostic, not a training experiment.  It measures whether the
hard-negative spatial objective pushes the shared encoder against the Any/IVH/
SAH classification objectives before spending GPU time on gradient surgery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_classification_weights,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import ICH25DSegmentationLoss
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import _unpack_outputs
from src.strategies.ich_v2.operations import file_sha256


TARGET_CLASSIFICATION_LABELS = ("any_ich", "IVH", "SAH")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _gradient_geometry(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> tuple[float | None, float, float]:
    dot: torch.Tensor | None = None
    left_sq: torch.Tensor | None = None
    right_sq: torch.Tensor | None = None
    for left_grad, right_grad in zip(left, right, strict=True):
        if left_grad is not None:
            left_value = left_grad.detach().double()
            term = torch.sum(left_value * left_value)
            left_sq = term if left_sq is None else left_sq + term
        if right_grad is not None:
            right_value = right_grad.detach().double()
            term = torch.sum(right_value * right_value)
            right_sq = term if right_sq is None else right_sq + term
        if left_grad is not None and right_grad is not None:
            term = torch.sum(
                left_grad.detach().double() * right_grad.detach().double()
            )
            dot = term if dot is None else dot + term
    left_norm = math.sqrt(float(left_sq.cpu())) if left_sq is not None else 0.0
    right_norm = math.sqrt(float(right_sq.cpu())) if right_sq is not None else 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return None, left_norm, right_norm
    cosine = float(dot.cpu()) / (left_norm * right_norm) if dot is not None else 0.0
    return max(-1.0, min(1.0, cosine)), left_norm, right_norm


def _classification_loss_for_label(
    class_logits: torch.Tensor,
    class_targets: torch.Tensor,
    classification_known: torch.Tensor,
    *,
    label_index: int,
    pos_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    rows = classification_known > 0.5
    logits = class_logits[rows, label_index].float()
    targets = class_targets[rows, label_index].float()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight[label_index],
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    correct_probability = (
        targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    )
    return (bce * (1.0 - correct_probability).pow(focal_gamma)).mean()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
            "negative_fraction": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


def _finite_summary_for_rows(
    rows: list[dict[str, float | int | str | None]],
    column: str,
) -> dict[str, float | int | None]:
    values = [
        float(row[column])
        for row in rows
        if row.get(column) is not None and math.isfinite(float(row[column]))
    ]
    return _summary(values)


def _label_conditioned_summaries(
    rows: list[dict[str, float | int | str | None]],
    label: str,
) -> dict[str, object]:
    """Separate gradient geometry in batches with and without label positives."""
    prefix = label.lower()
    positive = [row for row in rows if int(row[f"{prefix}_positive_rows"]) > 0]
    negative_only = [row for row in rows if int(row[f"{prefix}_positive_rows"]) == 0]
    columns = (
        f"cosine_segmentation_vs_{prefix}",
        f"cosine_base_segmentation_vs_{prefix}",
        f"cosine_hard_empty_vs_{prefix}",
        f"{prefix}_to_segmentation_grad_norm_ratio",
    )

    def summarize_subset(
        subset: list[dict[str, float | int | str | None]],
    ) -> dict[str, object]:
        return {
            "batches": len(subset),
            "positive_rows": int(
                sum(int(row[f"{prefix}_positive_rows"]) for row in subset)
            ),
            "metrics": {
                column: _finite_summary_for_rows(subset, column)
                for column in columns
            },
        }

    return {
        "with_positive_rows": summarize_subset(positive),
        "negative_only": summarize_subset(negative_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("train", "calibration", "outer"), default="train"
    )
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to the checkpoint training batch size.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, help="Defaults to the checkpoint seed.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    if args.max_batches <= 0 or args.workers < 0:
        raise ValueError("max-batches must be positive and workers non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.precision == "bf16" and args.device != "cuda":
        raise ValueError("BF16 diagnostic precision requires CUDA")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    batch_size = int(
        args.batch_size if args.batch_size is not None else config["batch_size"]
    )
    seed = int(args.seed if args.seed is not None else config["seed"])
    manifest_path = args.manifest_path or Path(
        str(config.get("manifest_path", "Data/processed/ich_2p5d/slice_manifest.csv"))
    )
    _seed_everything(seed)
    loaders = create_segmentation_loaders(
        manifest_path,
        outer_fold=int(config["outer_fold"]),
        calibration_fold=int(config["calibration_fold"]),
        batch_size=batch_size,
        workers=args.workers,
        seed=seed,
        sampler_study_balance_power=float(
            config.get("sampler_study_balance_power", 0.0)
        ),
    )
    loader_by_split = {
        "train": loaders[0],
        "calibration": loaders[1],
        "outer": loaders[2],
    }
    train_frame = loaders[3]
    loader = loader_by_split[args.split]
    device = torch.device(args.device)

    model = build_segmentation_model(
        architecture=str(config["architecture"]),
        encoder_name=str(config["encoder_name"]),
        pretrained=False,
        dropout=float(config["dropout"]),
    ).to(device)
    load_segmentation_weights(model, args.checkpoint)
    model.train(args.split == "train")
    shared_parameters = tuple(
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    if not shared_parameters:
        raise ValueError("Model encoder has no trainable shared parameters")

    pos_weight = segmentation_classification_weights(
        train_frame, maximum=float(config.get("maximum_pos_weight", 20.0))
    ).to(device)
    segmentation_class_weights = segmentation_foreground_weights(
        train_frame,
        power=float(config.get("segmentation_class_weight_power", 0.0)),
        maximum=float(config.get("maximum_segmentation_class_weight", 8.0)),
        basis=str(config.get("segmentation_class_weight_basis", "slice")),
    ).to(device)
    loss_fn = ICH25DSegmentationLoss(
        classification_pos_weight=pos_weight,
        segmentation_class_weights=segmentation_class_weights,
        classification_weight=float(config.get("classification_loss_weight", 0.25)),
        classification_focal_gamma=float(config.get("classification_focal_gamma", 1.0)),
        background_weight=float(config.get("background_weight", 0.15)),
        empty_foreground_weight=float(config.get("empty_foreground_weight", 0.0)),
        empty_foreground_top_fraction=float(
            config.get("empty_foreground_top_fraction", 1.0)
        ),
    ).to(device)
    classification_weight = float(config.get("classification_loss_weight", 0.25))
    empty_weight = float(config.get("empty_foreground_weight", 0.0))

    requested_labels = [label for label in TARGET_CLASSIFICATION_LABELS if label in OUTPUT_LABELS]
    if len(requested_labels) != len(TARGET_CLASSIFICATION_LABELS):
        raise ValueError(
            f"Expected classification labels {TARGET_CLASSIFICATION_LABELS}, got {OUTPUT_LABELS}"
        )

    batch_rows: list[dict[str, float | int | str | None]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        segmentation_known = batch["segmentation_known"].to(device, non_blocking=True)
        classification_known = batch["classification_known"].to(device, non_blocking=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else nullcontext()
        )
        with autocast:
            mask_logits, class_logits = _unpack_outputs(model(images))
            components = loss_fn.components(
                mask_logits,
                class_logits,
                masks,
                targets,
                segmentation_known,
                classification_known,
            )

        hard_empty_loss = empty_weight * components["empty_foreground"]
        base_segmentation_loss = components["segmentation"] - hard_empty_loss
        segmentation_loss = components["segmentation"]
        segmentation_grad = _gradients(
            segmentation_loss, shared_parameters, retain_graph=True
        )
        base_grad = _gradients(
            base_segmentation_loss, shared_parameters, retain_graph=True
        )
        hard_grad = _gradients(hard_empty_loss, shared_parameters, retain_graph=True)

        row: dict[str, float | int | str | None] = {
            "batch": batch_index,
            "split": args.split,
            "batch_size": int(images.shape[0]),
            "segmentation_known_rows": int((segmentation_known > 0.5).sum().item()),
            "empty_spatial_rows": int(
                (
                    (segmentation_known > 0.5)
                    & ~(masks > 0).flatten(start_dim=1).any(dim=1)
                ).sum().item()
            ),
            "segmentation_loss": float(segmentation_loss.detach().cpu()),
            "base_segmentation_loss": float(base_segmentation_loss.detach().cpu()),
            "weighted_hard_empty_loss": float(hard_empty_loss.detach().cpu()),
        }
        hard_base_cosine, hard_norm, base_norm = _gradient_geometry(hard_grad, base_grad)
        row["cosine_hard_vs_base_segmentation"] = hard_base_cosine
        row["hard_empty_grad_norm"] = hard_norm
        row["base_segmentation_grad_norm"] = base_norm
        row["hard_to_base_grad_norm_ratio"] = (
            hard_norm / base_norm if base_norm > 0.0 else None
        )

        total_classification_loss = classification_weight * components["classification"]
        total_classification_grad = _gradients(
            total_classification_loss, shared_parameters, retain_graph=True
        )
        total_seg_cosine, total_seg_norm, total_class_norm = _gradient_geometry(
            segmentation_grad, total_classification_grad
        )
        total_base_cosine, _, _ = _gradient_geometry(
            base_grad, total_classification_grad
        )
        total_hard_cosine, _, _ = _gradient_geometry(
            hard_grad, total_classification_grad
        )
        row["weighted_classification_loss"] = float(
            total_classification_loss.detach().cpu()
        )
        row["cosine_segmentation_vs_classification_total"] = total_seg_cosine
        row["cosine_base_segmentation_vs_classification_total"] = total_base_cosine
        row["cosine_hard_empty_vs_classification_total"] = total_hard_cosine
        row["classification_total_grad_norm"] = total_class_norm
        row["classification_to_segmentation_grad_norm_ratio"] = (
            total_class_norm / total_seg_norm if total_seg_norm > 0.0 else None
        )

        for label_index_in_request, label in enumerate(requested_labels):
            label_index = OUTPUT_LABELS.index(label)
            class_loss = (
                classification_weight
                * _classification_loss_for_label(
                    class_logits,
                    targets,
                    classification_known,
                    label_index=label_index,
                    pos_weight=pos_weight,
                    focal_gamma=float(config.get("classification_focal_gamma", 1.0)),
                )
                / len(OUTPUT_LABELS)
            )
            class_grad = _gradients(
                class_loss,
                shared_parameters,
                retain_graph=label_index_in_request < len(requested_labels) - 1,
            )
            seg_cosine, seg_norm, class_norm = _gradient_geometry(
                segmentation_grad, class_grad
            )
            base_cosine, _, _ = _gradient_geometry(base_grad, class_grad)
            hard_cosine, _, _ = _gradient_geometry(hard_grad, class_grad)
            prefix = label.lower()
            row[f"weighted_{prefix}_classification_loss"] = float(
                class_loss.detach().cpu()
            )
            row[f"{prefix}_positive_rows"] = int(
                (
                    (classification_known > 0.5)
                    & (targets[:, label_index] > 0.5)
                ).sum().item()
            )
            row[f"cosine_segmentation_vs_{prefix}"] = seg_cosine
            row[f"cosine_base_segmentation_vs_{prefix}"] = base_cosine
            row[f"cosine_hard_empty_vs_{prefix}"] = hard_cosine
            row[f"{prefix}_classification_grad_norm"] = class_norm
            row[f"{prefix}_to_segmentation_grad_norm_ratio"] = (
                class_norm / seg_norm if seg_norm > 0.0 else None
            )
        batch_rows.append(row)
        del (
            segmentation_grad,
            base_grad,
            hard_grad,
            total_classification_grad,
            components,
            mask_logits,
            class_logits,
        )

    if not batch_rows:
        raise ValueError("No diagnostic batches were processed")
    cosine_columns = sorted(
        column
        for column in batch_rows[0]
        if column.startswith("cosine_")
    )
    ratio_columns = sorted(
        column
        for column in batch_rows[0]
        if column.endswith("_grad_norm_ratio")
    )
    summaries = {}
    for column in (*cosine_columns, *ratio_columns):
        summaries[column] = _finite_summary_for_rows(batch_rows, column)

    segmentation_known_rows = int(
        sum(int(row["segmentation_known_rows"]) for row in batch_rows)
    )
    empty_spatial_rows = int(
        sum(int(row["empty_spatial_rows"]) for row in batch_rows)
    )
    conditioned_summaries = {
        label: _label_conditioned_summaries(batch_rows, label)
        for label in requested_labels
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gradient_conflict_batches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    result = {
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "diagnostic_only_no_parameter_updates": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "split": args.split,
        "batches": len(batch_rows),
        "batch_size": batch_size,
        "precision": args.precision,
        "seed": seed,
        "shared_parameter_tensors": len(shared_parameters),
        "classification_weight": classification_weight,
        "empty_foreground_weight": empty_weight,
        "empty_foreground_top_fraction": float(
            config.get("empty_foreground_top_fraction", 1.0)
        ),
        "batch_composition": {
            "segmentation_known_rows": segmentation_known_rows,
            "empty_spatial_rows": empty_spatial_rows,
            "empty_spatial_fraction_of_segmentation_known": (
                empty_spatial_rows / segmentation_known_rows
                if segmentation_known_rows > 0
                else None
            ),
        },
        "summaries": summaries,
        "label_conditioned_summaries": conditioned_summaries,
    }
    (args.output_dir / "gradient_conflict_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
