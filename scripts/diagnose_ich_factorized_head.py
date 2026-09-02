"""Aggregate-only technical gate for the factorized ICH output architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch

from src.strategies.ich_2p5d.segmentation_model import (
    FactorizedForegroundSubtypeModel,
    build_segmentation_model,
    compose_factorized_mask_logits,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    configure_trainable_parameters,
    load_initial_segmentation_checkpoint,
    set_segmentation_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 32:
        raise ValueError("batch_size must be positive and image_size at least 32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    source = payload["config"]
    architecture = str(source["architecture"])
    encoder_name = str(source["encoder_name"])
    outer_fold = int(source["outer_fold"])
    calibration_fold = int(source["calibration_fold"])

    legacy = build_segmentation_model(
        architecture=architecture,
        encoder_name=encoder_name,
        pretrained=False,
        dropout=float(source.get("dropout", 0.2)),
    ).to(device)
    load_segmentation_weights(legacy, args.checkpoint)
    factorized = build_segmentation_model(
        architecture=architecture,
        encoder_name=encoder_name,
        pretrained=False,
        dropout=float(source.get("dropout", 0.2)),
        factorized_output_head=True,
    ).to(device)
    if not isinstance(factorized, FactorizedForegroundSubtypeModel):
        raise TypeError("Builder did not return the factorized model")
    config = ICH25DSegmentationTrainConfig(
        run_name=args.run_name,
        output_dir=str(args.output_dir),
        architecture=architecture,
        encoder_name=encoder_name,
        outer_fold=outer_fold,
        calibration_fold=calibration_fold,
        initial_checkpoint=str(args.checkpoint),
        factorized_output_head=True,
        pretrained=False,
        evaluate_outer=False,
    )
    load_initial_segmentation_checkpoint(factorized, args.checkpoint, config)
    legacy.eval()
    factorized.eval()
    images = torch.randn(
        (args.batch_size, 9, args.image_size, args.image_size), device=device
    )
    with torch.no_grad():
        legacy_masks, legacy_classes = legacy(images)
        components = factorized.forward_components(images)
        factorized_masks = components["mask_logits"]
        factorized_classes = components["class_logits"]
        legacy_probabilities = torch.softmax(legacy_masks.float(), dim=1)
        factorized_probabilities = torch.softmax(factorized_masks.float(), dim=1)
        probability_difference = (
            factorized_probabilities - legacy_probabilities
        ).abs()
        maximum_probability_difference = float(probability_difference.max().cpu())
        mean_probability_difference = float(probability_difference.mean().cpu())
        argmax_mismatch_fraction = float(
            (factorized_masks.argmax(dim=1) != legacy_masks.argmax(dim=1))
            .float()
            .mean()
            .cpu()
        )
        classification_maximum_difference = float(
            (factorized_classes - legacy_classes).abs().max().cpu()
        )
        foreground_residual_maximum = float(
            components["foreground_residual"].abs().max().cpu()
        )
        subtype_residual_maximum = float(
            components["subtype_residual"].abs().max().cpu()
        )

    foreground = torch.randn((2, 1, 3, 4), device=device, requires_grad=True)
    subtype = torch.randn((2, 5, 3, 4), device=device, requires_grad=True)
    logits = compose_factorized_mask_logits(foreground, subtype)
    foreground_objective = torch.softmax(logits, dim=1)[:, 1:].sum()
    foreground_gradient, subtype_cross_gradient = torch.autograd.grad(
        foreground_objective, (foreground, subtype), retain_graph=True
    )
    conditional_objective = -torch.log_softmax(logits[:, 1:], dim=1)[:, 2].mean()
    foreground_cross_gradient, subtype_gradient = torch.autograd.grad(
        conditional_objective, (foreground, subtype)
    )
    maximum_foreground_to_subtype_cross_gradient = float(
        subtype_cross_gradient.abs().max().detach().cpu()
    )
    maximum_subtype_to_foreground_cross_gradient = float(
        foreground_cross_gradient.abs().max().detach().cpu()
    )

    trainable = configure_trainable_parameters(
        factorized, freeze_base_model=False, classification_head_only=False
    )
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    encoder_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in factorized.base_model.encoder.parameters()
        if parameter.requires_grad
    )
    classification_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in factorized.base_model.classification_head.parameters()
        if parameter.requires_grad
    )
    set_segmentation_training_mode(factorized, freeze_base_model=False)
    training_modes_correct = bool(
        not factorized.base_model.encoder.training
        and not factorized.base_model.classification_head.training
        and factorized.base_model.decoder.training
        and factorized.base_model.segmentation_head.training
        and factorized.foreground_residual_head.training
        and factorized.subtype_residual_head.training
    )
    del foreground_gradient, subtype_gradient

    gates = {
        "maximum_probability_difference_at_most_2e_6": (
            maximum_probability_difference <= 2e-6
        ),
        "hard_argmax_mismatch_fraction_zero": argmax_mismatch_fraction == 0.0,
        "classification_maximum_difference_zero": (
            classification_maximum_difference == 0.0
        ),
        "zero_initialized_residual_outputs": (
            foreground_residual_maximum == 0.0 and subtype_residual_maximum == 0.0
        ),
        "foreground_to_subtype_cross_gradient_at_most_1e_6": (
            maximum_foreground_to_subtype_cross_gradient <= 1e-6
        ),
        "subtype_to_foreground_cross_gradient_at_most_1e_6": (
            maximum_subtype_to_foreground_cross_gradient <= 1e-6
        ),
        "encoder_and_classifier_frozen": (
            encoder_trainable_parameter_count == 0
            and classification_trainable_parameter_count == 0
        ),
        "decoder_and_spatial_heads_trainable": trainable_parameter_count > 0,
        "training_modes_correct": training_modes_correct,
    }
    gates["all_passed"] = all(gates.values())
    result = {
        "analysis_kind": "factorized_foreground_subtype_architecture_technical_gate",
        "decision": (
            "authorize_preregistered_calibration_smoke"
            if gates["all_passed"]
            else "reject_before_any_training_or_held_out_access"
        ),
        "aggregate_only_no_medical_predictions": True,
        "diagnostic_only_no_parameter_updates": True,
        "evaluation_scope": "synthetic_input_and_checkpoint_structure_only",
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "architecture": architecture,
        "encoder_name": encoder_name,
        "outer_fold": outer_fold,
        "calibration_fold": calibration_fold,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "maximum_probability_difference": maximum_probability_difference,
        "mean_probability_difference": mean_probability_difference,
        "hard_argmax_mismatch_fraction": argmax_mismatch_fraction,
        "classification_maximum_difference": classification_maximum_difference,
        "foreground_residual_maximum": foreground_residual_maximum,
        "subtype_residual_maximum": subtype_residual_maximum,
        "maximum_foreground_to_subtype_cross_gradient": (
            maximum_foreground_to_subtype_cross_gradient
        ),
        "maximum_subtype_to_foreground_cross_gradient": (
            maximum_subtype_to_foreground_cross_gradient
        ),
        "trainable_parameter_count": trainable_parameter_count,
        "encoder_trainable_parameter_count": encoder_trainable_parameter_count,
        "classification_trainable_parameter_count": (
            classification_trainable_parameter_count
        ),
        "preregistered_gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "technical_gate.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        result["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "factorized_architecture_technical_gate",
                "evaluation_scope": result["evaluation_scope"],
                "git_commit": result["git_commit"],
                "decision": result["decision"],
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "architecture": architecture,
                "encoder_name": encoder_name,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
            }
        )
        mlflow.log_metrics(
            {
                "maximum_probability_difference": maximum_probability_difference,
                "mean_probability_difference": mean_probability_difference,
                "hard_argmax_mismatch_fraction": argmax_mismatch_fraction,
                "classification_maximum_difference": classification_maximum_difference,
                "foreground_to_subtype_cross_gradient": (
                    maximum_foreground_to_subtype_cross_gradient
                ),
                "subtype_to_foreground_cross_gradient": (
                    maximum_subtype_to_foreground_cross_gradient
                ),
                "trainable_parameter_count": float(trainable_parameter_count),
            }
        )
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path))
    notify_campaign(
        "completion",
        (
            "گیت فنی معماری فاکتورگیری‌شده تمام شد. "
            f"نتیجه: {'عبور' if gates['all_passed'] else 'رد'}؛ "
            f"بیشینه اختلاف احتمال={maximum_probability_difference:.3e}، "
            f"mismatch ماسک={argmax_mismatch_fraction:.3e} و بیشینهٔ گرادیان "
            "متقاطع="
            f"{max(maximum_foreground_to_subtype_cross_gradient, maximum_subtype_to_foreground_cross_gradient):.3e}. "
            "تحلیل کوتاه: عبور فقط سلامت جبری و مهندسی warm-start را ثابت می‌کند؛ "
            "بهبود بالینی باید جداگانه روی calibration قفل‌شده اثبات شود."
        ),
        run=args.run_name,
        decision=result["decision"],
        mlflow=result.get("mlflow_run_id", "n/a"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gates["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
