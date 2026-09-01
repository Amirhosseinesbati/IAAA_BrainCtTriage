"""Diagnose gradient strength/conflict in the frozen SAH expansion adapter.

Only patient-safe training batches with spatially supervised SAH pixels are
used.  No optimizer step is taken, no outer loader is iterated, and only
aggregate gradient geometry is persisted or logged.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from scripts.diagnose_ich_multitask_gradient_conflict import (
    _gradient_geometry,
    _gradients,
    _suggested_auxiliary_weight_fields,
)
from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_classification_weights,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import ICH25DSegmentationLoss
from src.strategies.ich_2p5d.segmentation_model import (
    SahBackgroundExpansionAdapter,
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import (
    _unpack_outputs,
    configure_trainable_parameters,
    set_segmentation_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


ACTUAL_SAH_TVERSKY_WEIGHT = 0.03
TARGET_GRADIENT_RATIOS = (0.10, 0.25, 0.50, 1.00)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _finite_summary_for_rows(
    rows: list[dict[str, float | int | None]], column: str
) -> dict[str, float | int | None]:
    values = [
        float(row[column])
        for row in rows
        if row.get(column) is not None and np.isfinite(float(row[column]))
    ]
    return _summary(values)


def gradient_diagnostic_interpretation(
    *, weighted_ratio_median: float, cosine_median: float
) -> str:
    if weighted_ratio_median < 0.05 and cosine_median < -0.10:
        return "underweighted_and_conflicting_sah_signal"
    if weighted_ratio_median < 0.05:
        return "underweighted_sah_signal"
    if cosine_median < -0.25:
        return "meaningful_sah_gradient_but_conflicting_objectives"
    return "sah_gradient_is_material_failure_likely_cap_or_representation"


def _combine_gradients(
    base: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    *,
    weight: float,
) -> tuple[torch.Tensor | None, ...]:
    combined: list[torch.Tensor | None] = []
    for base_gradient, auxiliary_gradient in zip(base, auxiliary, strict=True):
        if base_gradient is None and auxiliary_gradient is None:
            combined.append(None)
        elif base_gradient is None:
            assert auxiliary_gradient is not None
            combined.append(weight * auxiliary_gradient)
        elif auxiliary_gradient is None:
            combined.append(base_gradient)
        else:
            combined.append(base_gradient + weight * auxiliary_gradient)
    return tuple(combined)


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if args.positive_batches < 1 or args.maximum_scanned_batches < args.positive_batches:
        raise ValueError("Invalid positive/scanned batch limits")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SAH adapter gradient diagnostic requires BF16 CUDA")
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    source = payload["config"]
    if int(source["outer_fold"]) != args.outer_fold:
        raise ValueError("Checkpoint outer fold mismatch")
    if int(source["calibration_fold"]) != args.calibration_fold:
        raise ValueError("Checkpoint calibration fold mismatch")
    if str(source.get("architecture")) != args.architecture:
        raise ValueError("Checkpoint architecture mismatch")
    if str(source.get("encoder_name")) != args.encoder_name:
        raise ValueError("Checkpoint encoder mismatch")
    _seed_everything(args.seed)

    train_loader, _, _, train_frame, _, _ = create_segmentation_loaders(
        args.manifest,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        sampler_study_balance_power=0.0,
        context_radius=1,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=False,
        sah_residual_adapter=True,
        sah_residual_hidden_channels=args.hidden_channels,
        sah_maximum_logit_residual=args.maximum_logit_residual,
    ).to(device)
    if not isinstance(model, SahBackgroundExpansionAdapter):
        raise TypeError("Expected the SAH background expansion adapter")
    load_segmentation_weights(model.base_model, args.checkpoint)
    parameters = tuple(
        configure_trainable_parameters(model, freeze_base_model=True)
    )
    if sum(parameter.numel() for parameter in parameters) != 3217:
        raise ValueError("Unexpected SAH adapter trainable parameter count")
    set_segmentation_training_mode(model, freeze_base_model=True)

    pos_weight = segmentation_classification_weights(
        train_frame, maximum=20.0
    ).to(device)
    class_weights = segmentation_foreground_weights(
        train_frame,
        power=1.0,
        maximum=8.0,
        basis="pixel",
    ).to(device)
    loss_fn = ICH25DSegmentationLoss(
        classification_pos_weight=pos_weight,
        segmentation_class_weights=class_weights,
        classification_weight=0.0,
        classification_focal_gamma=1.0,
        background_weight=0.15,
        empty_foreground_weight=0.05,
        empty_foreground_top_fraction=0.001,
        sah_tversky_loss_weight=1.0,
    ).to(device)

    rows: list[dict[str, float | int | None]] = []
    scanned = 0
    for batch in train_loader:
        scanned += 1
        masks = batch["mask"]
        known = batch["segmentation_known"] > 0.5
        positive_rows = known & (masks == 5).flatten(start_dim=1).any(dim=1)
        if not torch.any(positive_rows):
            if scanned >= args.maximum_scanned_batches:
                break
            continue
        images = batch["image"].to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        segmentation_known = batch["segmentation_known"].to(device, non_blocking=True)
        classification_known = batch["classification_known"].to(
            device, non_blocking=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mask_logits, class_logits = _unpack_outputs(model(images))
            components = loss_fn.components(
                mask_logits,
                class_logits,
                masks,
                targets,
                segmentation_known,
                classification_known,
            )
        segmentation_gradients = _gradients(
            components["segmentation"], parameters, retain_graph=True
        )
        sah_gradients = _gradients(
            components["sah_tversky"], parameters, retain_graph=False
        )
        cosine, segmentation_norm, sah_norm = _gradient_geometry(
            segmentation_gradients, sah_gradients
        )
        if cosine is None or segmentation_norm <= 0 or sah_norm <= 0:
            raise ValueError("Positive SAH batch produced a zero adapter gradient")
        combined = _combine_gradients(
            segmentation_gradients,
            sah_gradients,
            weight=ACTUAL_SAH_TVERSKY_WEIGHT,
        )
        combined_sah_cosine, combined_norm, _ = _gradient_geometry(
            combined, sah_gradients
        )
        suggestions = _suggested_auxiliary_weight_fields(
            prefix="sah_tversky",
            target_ratios=TARGET_GRADIENT_RATIOS,
            segmentation_norm=segmentation_norm,
            auxiliary_norm=sah_norm,
        )
        rows.append(
            {
                "batch": scanned,
                "sah_positive_rows": int(positive_rows.sum()),
                "sah_positive_pixels": int((masks == 5).sum()),
                "segmentation_loss": float(components["segmentation"].detach()),
                "sah_tversky_loss": float(components["sah_tversky"].detach()),
                "cosine_segmentation_vs_sah": cosine,
                "segmentation_grad_norm": segmentation_norm,
                "sah_grad_norm": sah_norm,
                "raw_sah_to_segmentation_grad_norm_ratio": sah_norm
                / segmentation_norm,
                "weighted_003_sah_to_segmentation_grad_norm_ratio": (
                    ACTUAL_SAH_TVERSKY_WEIGHT * sah_norm / segmentation_norm
                ),
                "combined_003_vs_sah_cosine": combined_sah_cosine,
                "combined_003_grad_norm": combined_norm,
                **suggestions,
            }
        )
        if len(rows) >= args.positive_batches or scanned >= args.maximum_scanned_batches:
            break
    if len(rows) < args.positive_batches:
        raise ValueError(
            f"Only {len(rows)} SAH-positive batches found in {scanned} scans"
        )

    numeric_columns = (
        "segmentation_loss",
        "sah_tversky_loss",
        "cosine_segmentation_vs_sah",
        "segmentation_grad_norm",
        "sah_grad_norm",
        "raw_sah_to_segmentation_grad_norm_ratio",
        "weighted_003_sah_to_segmentation_grad_norm_ratio",
        "combined_003_vs_sah_cosine",
        "combined_003_grad_norm",
        "suggested_sah_tversky_weight_10pct",
        "suggested_sah_tversky_weight_25pct",
        "suggested_sah_tversky_weight_50pct",
        "suggested_sah_tversky_weight_100pct",
    )
    summaries = {
        column: _finite_summary_for_rows(rows, column)
        for column in numeric_columns
    }
    for required in (
        "weighted_003_sah_to_segmentation_grad_norm_ratio",
        "cosine_segmentation_vs_sah",
        "suggested_sah_tversky_weight_25pct",
        "combined_003_vs_sah_cosine",
    ):
        if summaries[required]["median"] is None:
            raise ValueError(f"No finite values recorded for {required}")
    weighted_ratio_median = float(
        summaries["weighted_003_sah_to_segmentation_grad_norm_ratio"]["median"]
    )
    cosine_median = float(
        summaries["cosine_segmentation_vs_sah"]["median"]
    )
    result = {
        "analysis_kind": "train_split_sah_adapter_gradient_diagnostic",
        "decision": gradient_diagnostic_interpretation(
            weighted_ratio_median=weighted_ratio_median,
            cosine_median=cosine_median,
        ),
        "diagnostic_only_no_parameter_updates": True,
        "outer_evaluation_performed": False,
        "outer_fold_reserved": args.outer_fold,
        "calibration_fold_not_used_for_gradient_tuning": args.calibration_fold,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "scanned_batches": scanned,
        "sah_positive_batches": len(rows),
        "actual_sah_tversky_weight": ACTUAL_SAH_TVERSKY_WEIGHT,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "summaries": summaries,
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "sah_adapter_gradient_diagnostic",
                "evaluation_scope": "train_only_no_calibration_or_outer_selection",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "positive_batches": len(rows),
                "actual_sah_tversky_weight": ACTUAL_SAH_TVERSKY_WEIGHT,
                "maximum_logit_residual": args.maximum_logit_residual,
            }
        )
        mlflow.log_metrics(
            {
                "weighted_ratio_median": weighted_ratio_median,
                "cosine_median": cosine_median,
                "suggested_weight_25pct_median": float(
                    summaries["suggested_sah_tversky_weight_25pct"]["median"]
                ),
                "combined_003_vs_sah_cosine_median": float(
                    summaries["combined_003_vs_sah_cosine"]["median"]
                ),
                "duration_s": result["duration_s"],
                "peak_vram_gb": result["peak_vram_gb"],
            }
        )
        mlflow.log_artifact(str(args.output), artifact_path="ich_diagnostics")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.notify:
        notify_campaign(
            "info",
            "🧭 تحلیل گرادیان SAH مسابقه IAAA کامل شد. تحلیل کوتاه: نسبت و جهت "
            "واقعی loss اصلی و SAH-Tversky روی head ایزوله اندازه‌گیری شد؛ تصمیم "
            "آزمایش بعدی بر پایهٔ این نسبت است، نه افزایش حدسی وزن یا epoch.",
            experiment="exp65_postmortem_sah_adapter_gradients",
            decision=result["decision"],
            positive_batches=len(rows),
            weighted_ratio_median=f"{weighted_ratio_median:.4f}",
            cosine_median=f"{cosine_median:.4f}",
            suggested_weight_25pct=f"{float(summaries['suggested_sah_tversky_weight_25pct']['median']):.4f}",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--architecture", default="unetplusplus")
    parser.add_argument("--encoder-name", default="tu-efficientnetv2_rw_s")
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-batches", type=int, default=12)
    parser.add_argument("--maximum-scanned-batches", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--maximum-logit-residual", type=float, default=8.0)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    result = run_diagnostic(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
