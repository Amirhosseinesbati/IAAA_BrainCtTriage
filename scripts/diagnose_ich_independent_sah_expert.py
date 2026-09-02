"""Local-only separability probe for an independently trained SAH expert.

Exp86 deliberately does not fuse predictions into the incumbent mask.  It trains
an independent decoder/head on the patient-safe training folds, evaluates a
single fixed final epoch on the calibration fold, and writes only aggregate
histogram metrics.  The outer fold and row-level predictions remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_v2.operations import file_sha256, git_commit


BACKGROUND_CLASS_ID = 0
IPH_CLASS_ID = OUTPUT_LABELS.index("IPH")
SAH_CLASS_ID = OUTPUT_LABELS.index("SAH")
HISTOGRAM_BINS = 4096
PRIMARY_REGION = "background_or_iph"
PRIMARY_SCORE = "expert_gated"
BASELINE_SCORE = "incumbent_gated"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


class IndependentSahExpert(torch.nn.Module):
    """Learn a SAH-specific decoder while keeping the incumbent immutable."""

    def __init__(self, incumbent_model: torch.nn.Module) -> None:
        super().__init__()
        for name in (
            "encoder",
            "decoder",
            "segmentation_head",
            "classification_head",
        ):
            if not isinstance(getattr(incumbent_model, name, None), torch.nn.Module):
                raise ValueError(f"Incumbent model does not expose {name}")
        incumbent_head = incumbent_model.segmentation_head[0]
        if not isinstance(incumbent_head, torch.nn.Conv2d):
            raise TypeError("Exp86 requires a convolutional incumbent mask head")
        if incumbent_head.out_channels != len(OUTPUT_LABELS):
            raise ValueError("Incumbent head class count does not match OUTPUT_LABELS")

        self.incumbent_model = incumbent_model
        self.expert_decoder = copy.deepcopy(incumbent_model.decoder)
        self.expert_head = torch.nn.Conv2d(
            incumbent_head.in_channels,
            1,
            kernel_size=incumbent_head.kernel_size,
            stride=incumbent_head.stride,
            padding=incumbent_head.padding,
            dilation=incumbent_head.dilation,
            bias=incumbent_head.bias is not None,
        )
        with torch.no_grad():
            reference = 0.5 * (
                incumbent_head.weight[BACKGROUND_CLASS_ID]
                + incumbent_head.weight[IPH_CLASS_ID]
            )
            self.expert_head.weight[0].copy_(
                incumbent_head.weight[SAH_CLASS_ID] - reference
            )
            if incumbent_head.bias is not None and self.expert_head.bias is not None:
                reference_bias = 0.5 * (
                    incumbent_head.bias[BACKGROUND_CLASS_ID]
                    + incumbent_head.bias[IPH_CLASS_ID]
                )
                self.expert_head.bias[0].copy_(
                    incumbent_head.bias[SAH_CLASS_ID] - reference_bias
                )
        self.incumbent_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.incumbent_model.eval()
        return self

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        self.requires_grad_(False)
        self.expert_decoder.requires_grad_(True)
        self.expert_head.requires_grad_(True)
        parameters = [
            *self.expert_decoder.parameters(),
            *self.expert_head.parameters(),
        ]
        if not parameters:
            raise ValueError("Independent SAH expert has no trainable parameters")
        return parameters

    def set_training_mode(self) -> None:
        self.train()
        self.incumbent_model.eval()
        self.expert_decoder.train()
        self.expert_head.train()

    def forward_components(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            encoded = self.incumbent_model.encoder(images)
            if not isinstance(encoded, (list, tuple)) or not encoded:
                raise TypeError("Incumbent encoder must return a feature sequence")
            incumbent_features = list(encoded)
            expert_features = [feature.detach().clone() for feature in encoded]
            incumbent_decoded = self.incumbent_model.decoder(incumbent_features)
            incumbent_logits = self.incumbent_model.segmentation_head(
                incumbent_decoded
            )
            class_logits = self.incumbent_model.classification_head(encoded[-1])
        expert_decoded = self.expert_decoder(expert_features)
        expert_logit = self.expert_head(expert_decoded)
        if expert_logit.shape[-2:] != incumbent_logits.shape[-2:]:
            expert_logit = F.interpolate(
                expert_logit,
                size=incumbent_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return {
            "expert_logit": expert_logit,
            "incumbent_mask_logits": incumbent_logits,
            "incumbent_class_logits": class_logits,
        }


def masked_sah_expert_loss(
    expert_logit: torch.Tensor,
    incumbent_mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    known: torch.Tensor,
    *,
    positive_weight: float = 16.0,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.60,
    tversky_weight: float = 0.40,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize only recoverable background/IPH-to-SAH pixels."""

    if expert_logit.ndim != 4 or expert_logit.shape[1] != 1:
        raise ValueError("expert_logit must have shape [B, 1, H, W]")
    incumbent = incumbent_mask_logits.detach().argmax(dim=1)
    eligible = (incumbent == BACKGROUND_CLASS_ID) | (incumbent == IPH_CLASS_ID)
    supervised = eligible & (known[:, None, None] > 0.5)
    if not bool(supervised.any()):
        raise ValueError("Exp86 batch contains no supervised recoverable pixels")
    target = (target_mask == SAH_CLASS_ID).float()
    logits = expert_logit[:, 0].float()
    mask = supervised.float()
    positive_weight_tensor = torch.as_tensor(
        positive_weight, dtype=logits.dtype, device=logits.device
    )
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight_tensor,
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    focal = (bce * (1.0 - probability_true).pow(focal_gamma) * mask).sum()
    focal = focal / mask.sum().clamp_min(1.0)

    true_positive = (probability * target * mask).sum()
    false_positive = (probability * (1.0 - target) * mask).sum()
    false_negative = ((1.0 - probability) * target * mask).sum()
    tversky = (true_positive + 1.0) / (
        true_positive + 0.30 * false_positive + 0.70 * false_negative + 1.0
    )
    loss = focal_weight * focal + tversky_weight * (1.0 - tversky)
    statistics = {
        "loss": float(loss.detach().cpu()),
        "focal": float(focal.detach().cpu()),
        "tversky": float(tversky.detach().cpu()),
        "eligible_pixels": float(mask.sum().detach().cpu()),
        "positive_pixels": float((target * mask).sum().detach().cpu()),
    }
    return loss, statistics


class StreamingBinaryHistogram:
    """Threshold-free binary metrics without retaining pixel predictions."""

    def __init__(self, bins: int = HISTOGRAM_BINS) -> None:
        if bins < 32:
            raise ValueError("Histogram metric requires at least 32 bins")
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        selected_scores = scores.detach()[mask].float().clamp(0.0, 1.0)
        selected_targets = targets.detach()[mask].bool()
        if selected_scores.numel() == 0:
            return
        indices = torch.clamp(
            (selected_scores * (self.bins - 1)).long(), 0, self.bins - 1
        )
        positive = torch.bincount(
            indices[selected_targets], minlength=self.bins
        ).cpu().numpy()
        negative = torch.bincount(
            indices[~selected_targets], minlength=self.bins
        ).cpu().numpy()
        self.positive += positive
        self.negative += negative

    def summarize(self) -> dict[str, float | int | None]:
        positive_total = int(self.positive.sum())
        negative_total = int(self.negative.sum())
        if positive_total == 0 or negative_total == 0:
            return {
                "positive_pixels": positive_total,
                "negative_pixels": negative_total,
                "prevalence": None,
                "average_precision": None,
                "roc_auc": None,
                "precision_at_recall_0_10": None,
                "precision_at_recall_0_25": None,
            }

        positive_descending = self.positive[::-1]
        negative_descending = self.negative[::-1]
        true_positive = np.cumsum(positive_descending, dtype=np.float64)
        false_positive = np.cumsum(negative_descending, dtype=np.float64)
        precision = true_positive / np.maximum(1.0, true_positive + false_positive)
        recall = true_positive / positive_total
        recall_increment = np.diff(np.concatenate([[0.0], recall]))
        average_precision = float(np.sum(precision * recall_increment))

        negatives_below = np.cumsum(self.negative, dtype=np.float64) - self.negative
        concordance = np.sum(
            self.positive * (negatives_below + 0.5 * self.negative),
            dtype=np.float64,
        )
        roc_auc = float(concordance / (positive_total * negative_total))

        def precision_at_recall(target_recall: float) -> float:
            indices = np.flatnonzero(recall >= target_recall)
            return float(precision[indices[0]]) if len(indices) else 0.0

        return {
            "positive_pixels": positive_total,
            "negative_pixels": negative_total,
            "prevalence": positive_total / (positive_total + negative_total),
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "precision_at_recall_0_10": precision_at_recall(0.10),
            "precision_at_recall_0_25": precision_at_recall(0.25),
        }


def separability_gate(metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    primary = metrics[PRIMARY_REGION][PRIMARY_SCORE]
    baseline = metrics[PRIMARY_REGION][BASELINE_SCORE]
    near = metrics["near_incumbent_foreground"][PRIMARY_SCORE]
    near_baseline = metrics["near_incumbent_foreground"][BASELINE_SCORE]
    required = [
        primary["average_precision"],
        primary["roc_auc"],
        primary["precision_at_recall_0_10"],
        baseline["average_precision"],
        baseline["roc_auc"],
        near["average_precision"],
        near_baseline["average_precision"],
    ]
    finite = all(value is not None and math.isfinite(float(value)) for value in required)
    if not finite:
        checks = {"all_required_metrics_finite": False}
    else:
        ap_gain = float(primary["average_precision"] - baseline["average_precision"])
        ap_ratio = float(primary["average_precision"]) / max(
            1e-12, float(baseline["average_precision"])
        )
        near_ap_gain = float(
            near["average_precision"] - near_baseline["average_precision"]
        )
        prevalence = float(primary["prevalence"])
        checks = {
            "all_required_metrics_finite": True,
            "at_least_512_recoverable_sah_pixels": int(primary["positive_pixels"])
            >= 512,
            "primary_average_precision_gain_at_least_0_005": ap_gain >= 0.005,
            "primary_average_precision_ratio_at_least_1_25": ap_ratio >= 1.25,
            "primary_roc_auc_gain_at_least_0_01": float(primary["roc_auc"])
            >= float(baseline["roc_auc"]) + 0.01,
            "near_foreground_average_precision_gain_at_least_0_01": near_ap_gain
            >= 0.01,
            "precision_at_10pct_recall_at_least_2pct_or_20x_prevalence": float(
                primary["precision_at_recall_0_10"]
            )
            >= max(0.02, 20.0 * prevalence),
        }
    checks["all_passed"] = all(bool(value) for value in checks.values())
    return checks


def _build_expert(
    checkpoint: Path, source_config: dict[str, Any], device: torch.device
) -> IndependentSahExpert:
    incumbent = build_segmentation_model(
        architecture=str(source_config["architecture"]),
        encoder_name=str(source_config["encoder_name"]),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
    )
    load_segmentation_weights(incumbent, checkpoint)
    return IndependentSahExpert(incumbent).to(device)


def _evaluate(
    model: IndependentSahExpert,
    loader: Any,
    device: torch.device,
) -> dict[str, dict[str, dict[str, Any]]]:
    regions = ("background_or_iph", "near_incumbent_foreground", "iph_only")
    scores = ("incumbent_raw", "incumbent_gated", "expert_raw", "expert_gated")
    histograms = {
        region: {name: StreamingBinaryHistogram() for name in scores}
        for region in regions
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components = model.forward_components(images)
            incumbent_logits = components["incumbent_mask_logits"].float()
            incumbent_mask = incumbent_logits.argmax(dim=1)
            target = batch["mask"].to(device, non_blocking=True) == SAH_CLASS_ID
            known = batch["known"].to(device, non_blocking=True)[:, None, None] > 0.5
            recoverable = known & (
                (incumbent_mask == BACKGROUND_CLASS_ID)
                | (incumbent_mask == IPH_CLASS_ID)
            )
            foreground = (incumbent_mask != BACKGROUND_CLASS_ID).float()[:, None]
            near_foreground = F.max_pool2d(
                foreground, kernel_size=15, stride=1, padding=7
            )[:, 0] > 0.5
            region_masks = {
                "background_or_iph": recoverable,
                "near_incumbent_foreground": recoverable & near_foreground,
                "iph_only": known & (incumbent_mask == IPH_CLASS_ID),
            }
            incumbent_margin = incumbent_logits[:, SAH_CLASS_ID] - torch.logsumexp(
                incumbent_logits[:, [BACKGROUND_CLASS_ID, IPH_CLASS_ID]], dim=1
            )
            incumbent_probability = torch.sigmoid(incumbent_margin)
            expert_probability = torch.sigmoid(components["expert_logit"][:, 0].float())
            sah_slice_gate = torch.sigmoid(
                components["incumbent_class_logits"][:, SAH_CLASS_ID].float()
            )[:, None, None]
            score_tensors = {
                "incumbent_raw": incumbent_probability,
                "incumbent_gated": incumbent_probability * sah_slice_gate,
                "expert_raw": expert_probability,
                "expert_gated": expert_probability * sah_slice_gate,
            }
            for region, region_mask in region_masks.items():
                for name, score in score_tensors.items():
                    histograms[region][name].update(score, target, region_mask)
    return {
        region: {name: histogram.summarize() for name, histogram in values.items()}
        for region, values in histograms.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.epochs != 3
        or args.batch_size != 8
        or args.learning_rate != 1e-4
        or args.weight_decay != 1e-4
    ):
        raise ValueError("Exp86 is locked to epochs=3, batch=8, lr=1e-4, wd=1e-4")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp86 requires CUDA BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "aggregate_result.json"
    checkpoint_path = args.output_dir / "expert.pth"
    if summary_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite Exp86 output: {args.output_dir}")

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a standard training config")
    source_config = payload["config"]
    if int(source_config.get("outer_fold", -1)) != 2 or int(
        source_config.get("calibration_fold", -1)
    ) != 1:
        raise ValueError("Exp86 is locked to outer=2/calibration=1")
    seed = int(source_config.get("seed", 42))
    _seed_everything(seed)
    (
        train_loader,
        calibration_loader,
        _,
        train_frame,
        calibration_frame,
        outer_frame,
    ) = create_segmentation_loaders(
        args.manifest_path,
        outer_fold=2,
        calibration_fold=1,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=seed,
        sampler_study_balance_power=0.0,
    )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = _build_expert(args.checkpoint, source_config, device)
    parameters = model.trainable_parameters()
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.set_training_mode()
        epoch_statistics: dict[str, list[float]] = {
            name: []
            for name in (
                "loss",
                "focal",
                "tversky",
                "eligible_pixels",
                "positive_pixels",
            )
        }
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            target_mask = batch["mask"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components = model.forward_components(images)
                loss, statistics = masked_sah_expert_loss(
                    components["expert_logit"],
                    components["incumbent_mask_logits"],
                    target_mask,
                    known,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            for name, value in statistics.items():
                epoch_statistics[name].append(value)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate_used,
                **{
                    name: float(np.mean(values))
                    for name, values in epoch_statistics.items()
                },
            }
        )
        scheduler.step()

    metrics = _evaluate(model, calibration_loader, device)
    gate = separability_gate(metrics)
    decision = (
        "advance_to_preregistered_safe_fusion_screen"
        if gate["all_passed"]
        else "reject_independent_sah_expert_before_fusion_or_outer"
    )
    duration_s = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "analysis_kind": "independent_sah_expert_threshold_free_separability_probe",
        "experiment": "exp86_independent_sah_expert_separability_v1",
        "run_name": args.run_name,
        "decision": decision,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "git_commit": git_commit(),
        "seed": seed,
        "precision": "bf16",
        "outer_fold": 2,
        "calibration_fold": 1,
        "outer_evaluation_performed": False,
        "row_level_predictions_persisted": False,
        "external_reporting_enabled": False,
        "threshold_search_performed": False,
        "train_slices": int(len(train_frame)),
        "calibration_slices": int(len(calibration_frame)),
        "outer_slices_not_evaluated": int(len(outer_frame)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "trainable_parameter_count": trainable_parameter_count,
        "training_history": history,
        "calibration_histogram_bins": HISTOGRAM_BINS,
        "calibration_metrics": metrics,
        "preregistered_gate": gate,
        "duration_s": duration_s,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "expert_checkpoint_saved": bool(gate["all_passed"]),
    }
    if gate["all_passed"]:
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "schema_version": 1,
                "state_dict": model.state_dict(),
                "source_checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "experiment": result["experiment"],
                "training_scope": "independent_sah_decoder_and_binary_head",
                "calibration_metrics": metrics,
                "git_commit": git_commit(),
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        result["expert_checkpoint_sha256"] = file_sha256(checkpoint_path)
    _write_json(summary_path, result)
    _write_json(args.output_dir / "training_history.json", history)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
