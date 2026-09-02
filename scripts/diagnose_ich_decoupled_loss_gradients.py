"""Train-only gradient probe for the hierarchical ICH segmentation objective."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import torch

from scripts.diagnose_ich_multitask_gradient_conflict import _gradient_geometry
from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    HierarchicalForegroundSubtypeLoss,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import _unpack_outputs
from src.strategies.ich_v2.losses import MaskedDiceFocalLoss
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


SUBTYPE_NAMES = ("IVH", "IPH", "SDH", "EDH", "SAH")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def _target_attraction_by_class(
    gradient: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, tuple[float, int]]:
    result: dict[str, tuple[float, int]] = {}
    gradient_last = gradient.detach().float().movedim(1, -1)
    for class_id, name in enumerate(SUBTYPE_NAMES, start=1):
        selected = valid & (masks == class_id)
        count = int(selected.sum().item())
        if count:
            target_gradient = gradient_last[selected][:, class_id]
            result[name] = (float(target_gradient.abs().sum().cpu()), count)
        else:
            result[name] = (0.0, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.max_batches <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Batch counts and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.precision == "bf16" and args.device != "cuda":
        raise ValueError("BF16 requires CUDA")

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    loaders = create_segmentation_loaders(
        args.manifest_path,
        outer_fold=int(config["outer_fold"]),
        calibration_fold=int(config["calibration_fold"]),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=seed,
        sampler_study_balance_power=float(
            config.get("sampler_study_balance_power", 0.0)
        ),
    )
    train_loader, _, _, train_frame = loaders[:4]
    device = torch.device(args.device)
    model = build_segmentation_model(
        architecture=str(config["architecture"]),
        encoder_name=str(config["encoder_name"]),
        pretrained=False,
        dropout=float(config["dropout"]),
    ).to(device)
    load_segmentation_weights(model, args.checkpoint)
    model.eval()
    probe_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("decoder.") or name.startswith("segmentation_head."))
    )
    if not probe_parameters:
        raise ValueError("No decoder or segmentation-head parameters were found")

    class_weights = segmentation_foreground_weights(
        train_frame,
        power=float(config.get("segmentation_class_weight_power", 0.0)),
        maximum=float(config.get("maximum_segmentation_class_weight", 8.0)),
        basis=str(config.get("segmentation_class_weight_basis", "slice")),
    ).to(device)
    shared = {
        "foreground_weights": class_weights,
        "background_weight": float(config.get("background_weight", 0.15)),
        "empty_foreground_weight": float(
            config.get("empty_foreground_weight", 0.0)
        ),
        "empty_foreground_top_fraction": float(
            config.get("empty_foreground_top_fraction", 1.0)
        ),
    }
    incumbent_loss = MaskedDiceFocalLoss(
        num_classes=6,
        dice_weight=0.65,
        focal_weight=0.35,
        focal_gamma=2.0,
        **shared,
    ).to(device)
    candidate_loss = HierarchicalForegroundSubtypeLoss(
        foreground_class_weights=shared["foreground_weights"],
        background_weight=shared["background_weight"],
        empty_foreground_weight=shared["empty_foreground_weight"],
        empty_foreground_top_fraction=shared["empty_foreground_top_fraction"],
    ).to(device)

    decoder_cosines: list[float] = []
    incumbent_losses: list[float] = []
    candidate_losses: list[float] = []
    background_old_abs_sum = 0.0
    background_new_abs_sum = 0.0
    background_values = 0
    class_sums = {
        name: {"incumbent": 0.0, "candidate": 0.0, "pixels": 0}
        for name in SUBTYPE_NAMES
    }
    processed = 0
    for batch in train_loader:
        if processed >= args.max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        known_rows = batch["segmentation_known"].to(device, non_blocking=True) > 0.5
        if not torch.any(known_rows):
            continue
        supervision = known_rows[:, None, None].expand_as(masks)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else nullcontext()
        )
        with autocast:
            mask_logits, _ = _unpack_outputs(model(images))
            incumbent = incumbent_loss.components(mask_logits, masks, supervision)
            candidate = candidate_loss.components(mask_logits, masks, supervision)
        incumbent_parameter_grad = torch.autograd.grad(
            incumbent["loss"],
            probe_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        candidate_parameter_grad = torch.autograd.grad(
            candidate["loss"],
            probe_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        cosine, _, _ = _gradient_geometry(
            incumbent_parameter_grad, candidate_parameter_grad
        )
        if cosine is not None:
            decoder_cosines.append(cosine)
        incumbent_logit_grad = torch.autograd.grad(
            incumbent["loss"], mask_logits, retain_graph=True
        )[0]
        candidate_logit_grad = torch.autograd.grad(
            candidate["loss"], mask_logits
        )[0]
        valid = supervision > 0.5
        old_class = _target_attraction_by_class(incumbent_logit_grad, masks, valid)
        new_class = _target_attraction_by_class(candidate_logit_grad, masks, valid)
        for name in SUBTYPE_NAMES:
            class_sums[name]["incumbent"] += old_class[name][0]
            class_sums[name]["candidate"] += new_class[name][0]
            class_sums[name]["pixels"] += old_class[name][1]
        background = valid & (masks == 0)
        if torch.any(background):
            old_foreground_grad = incumbent_logit_grad[:, 1:].movedim(1, -1)[
                background
            ]
            new_foreground_grad = candidate_logit_grad[:, 1:].movedim(1, -1)[
                background
            ]
            background_old_abs_sum += float(old_foreground_grad.abs().sum().cpu())
            background_new_abs_sum += float(new_foreground_grad.abs().sum().cpu())
            background_values += int(old_foreground_grad.numel())
        incumbent_losses.append(float(incumbent["loss"].detach().cpu()))
        candidate_losses.append(float(candidate["loss"].detach().cpu()))
        processed += 1

    if processed == 0:
        raise ValueError("No train batches with spatial supervision were processed")
    subtype_summary = {}
    for name, values in class_sums.items():
        count = int(values["pixels"])
        incumbent_mean = values["incumbent"] / count if count else None
        candidate_mean = values["candidate"] / count if count else None
        subtype_summary[name] = {
            "pixels": count,
            "incumbent_mean_absolute_target_logit_gradient": incumbent_mean,
            "candidate_mean_absolute_target_logit_gradient": candidate_mean,
            "candidate_to_incumbent_ratio": _safe_ratio(
                candidate_mean, incumbent_mean
            ),
        }
    background_old_mean = (
        background_old_abs_sum / background_values if background_values else None
    )
    background_new_mean = (
        background_new_abs_sum / background_values if background_values else None
    )
    decoder_cosine_mean = _finite_mean(decoder_cosines)
    background_ratio = _safe_ratio(background_new_mean, background_old_mean)
    finite_values = [
        _finite_mean(incumbent_losses),
        _finite_mean(candidate_losses),
        decoder_cosine_mean,
        background_ratio,
        *(
            subtype_summary[name]["candidate_to_incumbent_ratio"]
            for name in SUBTYPE_NAMES
        ),
    ]
    gates = {
        "all_subtypes_at_least_100_pixels": all(
            subtype_summary[name]["pixels"] >= 100 for name in SUBTYPE_NAMES
        ),
        "edh_target_attraction_ratio_at_least_1_10": (
            subtype_summary["EDH"]["candidate_to_incumbent_ratio"] is not None
            and subtype_summary["EDH"]["candidate_to_incumbent_ratio"] >= 1.10
        ),
        "sah_target_attraction_ratio_at_least_1_25": (
            subtype_summary["SAH"]["candidate_to_incumbent_ratio"] is not None
            and subtype_summary["SAH"]["candidate_to_incumbent_ratio"] >= 1.25
        ),
        "background_gradient_ratio_at_most_1_50": (
            background_ratio is not None and background_ratio <= 1.50
        ),
        "decoder_gradient_cosine_at_least_0_10": (
            decoder_cosine_mean is not None and decoder_cosine_mean >= 0.10
        ),
        "all_aggregate_values_finite": all(
            value is not None and math.isfinite(float(value)) for value in finite_values
        ),
    }
    gates["all_passed"] = all(gates.values())
    result = {
        "analysis_kind": "train_only_decoupled_foreground_subtype_gradient_probe",
        "decision": (
            "authorize_preregistered_calibration_screen"
            if gates["all_passed"]
            else "reject_exact_loss_weighting_before_calibration_or_outer"
        ),
        "diagnostic_only_no_parameter_updates": True,
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "split": "train",
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": manifest_sha,
        "batches": processed,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "seed": seed,
        "probe_parameter_tensors": len(probe_parameters),
        "objective_weights": {
            "foreground_dice": 0.40,
            "foreground_focal": 0.20,
            "conditional_subtype": 0.30,
            "subtype_ovr": 0.10,
        },
        "mean_incumbent_segmentation_loss": _finite_mean(incumbent_losses),
        "mean_candidate_segmentation_loss": _finite_mean(candidate_losses),
        "mean_decoder_head_gradient_cosine": decoder_cosine_mean,
        "true_background_foreground_gradient": {
            "values": background_values,
            "incumbent_mean_absolute": background_old_mean,
            "candidate_mean_absolute": background_new_mean,
            "candidate_to_incumbent_ratio": background_ratio,
        },
        "subtypes": subtype_summary,
        "preregistered_gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "gradient_probe.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        result["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "train_only_gradient_diagnostic",
                "evaluation_scope": result["evaluation_scope"],
                "git_commit": result["git_commit"],
                "decision": result["decision"],
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "batches": processed,
                "batch_size": args.batch_size,
                "precision": args.precision,
                "seed": seed,
            }
        )
        metrics = {
            "decoder_head_gradient_cosine": float(decoder_cosine_mean or 0.0),
            "background_gradient_ratio": float(background_ratio or 0.0),
            **{
                f"{name.lower()}_target_attraction_ratio": float(
                    subtype_summary[name]["candidate_to_incumbent_ratio"] or 0.0
                )
                for name in SUBTYPE_NAMES
            },
        }
        mlflow.log_metrics(metrics)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path))
    notify_campaign(
        "completion",
        (
            "آزمایش گرادیانی سلسله‌مراتبی تمام شد. "
            f"نتیجه: {'عبور' if gates['all_passed'] else 'رد'}؛ "
            "تقویت EDH="
            f"{float(subtype_summary['EDH']['candidate_to_incumbent_ratio'] or float('nan')):.2f}×، "
            "SAH="
            f"{float(subtype_summary['SAH']['candidate_to_incumbent_ratio'] or float('nan')):.2f}×، "
            f"فشار پس‌زمینه={float(background_ratio or float('nan')):.2f}× و "
            f"cosine={float(decoder_cosine_mean or float('nan')):.3f}. "
            "تحلیل کوتاه: عبور فقط مجوز غربال calibration است؛ رد یعنی این وزن‌دهی "
            "بدون مصرف outer کنار گذاشته می‌شود."
        ),
        run=args.run_name,
        decision=result["decision"],
        mlflow=result.get("mlflow_run_id", "n/a"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
