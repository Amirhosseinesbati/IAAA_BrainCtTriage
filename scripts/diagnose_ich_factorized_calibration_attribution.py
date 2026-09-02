"""Calibration-only attribution of the four-step Exp79/80 factorized failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW

from scripts.diagnose_ich_factorized_loss_causality import (
    SUBTYPE_NAMES,
    VARIANT_COMPONENTS,
    _build_factorized_model,
    _build_loss,
    _clone_batch,
    _forward_loss_components,
    _seed_everything,
    _variant_objective,
)
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_model import (
    factorized_trainable_parameters,
    set_factorized_training_mode,
)
from src.strategies.ich_2p5d.segmentation_train import (
    _predict_slices,
    checkpoint_selection_score,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


SUMMARY_METRICS = (
    "selection_score",
    "mean_foreground_dice",
    "any_ich_study_auc",
    "macro_subtype_study_auc",
    "presence_f1_at_0_1ml",
    "normal_false_positive_rate_at_0_1ml",
    "total_volume_mae_ml",
    "total_volume_bias_ml",
)


def _summary_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    result = {
        metric: float(candidate[metric] - baseline[metric])
        for metric in SUMMARY_METRICS
    }
    result["checkpoint_score"] = float(
        checkpoint_selection_score(candidate, "fpr_volume_penalized")
        - checkpoint_selection_score(baseline, "fpr_volume_penalized")
    )
    result["subtypes"] = {
        name: {
            "dice_known_pixels": float(
                candidate["subtypes"][name]["dice_known_pixels"]
                - baseline["subtypes"][name]["dice_known_pixels"]
            ),
            "mae_ml": float(
                candidate["subtypes"][name]["mae_ml"]
                - baseline["subtypes"][name]["mae_ml"]
            ),
            "bias_ml": float(
                candidate["subtypes"][name]["bias_ml"]
                - baseline["subtypes"][name]["bias_ml"]
            ),
        }
        for name in SUBTYPE_NAMES
    }
    return result


def calibration_attribution_decision(
    deltas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def diffuse(variant: str) -> float:
        return float(
            np.mean(
                [
                    deltas[variant]["subtypes"][name]["dice_known_pixels"]
                    for name in ("SDH", "SAH")
                ]
            )
        )

    full_dice = float(deltas["full_exp80"]["mean_foreground_dice"])
    no_dice_rescue = float(
        deltas["without_conditional_dice"]["mean_foreground_dice"]
        - full_dice
    )
    no_focal_rescue = float(
        deltas["without_conditional_focal"]["mean_foreground_dice"]
        - full_dice
    )
    no_dice_diffuse_rescue = diffuse("without_conditional_dice") - diffuse(
        "full_exp80"
    )
    no_focal_diffuse_rescue = diffuse("without_conditional_focal") - diffuse(
        "full_exp80"
    )
    dice_pass = no_dice_rescue >= 0.01 and no_dice_diffuse_rescue >= 0.02
    focal_pass = no_focal_rescue >= 0.01 and no_focal_diffuse_rescue >= 0.02
    if full_dice > -0.01:
        decision = "four_step_calibration_failure_not_reproduced"
    elif dice_pass and focal_pass:
        decision = "conditional_loss_interaction_primary_suspect"
    elif dice_pass and no_dice_rescue >= no_focal_rescue + 0.005:
        decision = "conditional_dice_primary_suspect"
    elif focal_pass and no_focal_rescue >= no_dice_rescue + 0.005:
        decision = "conditional_focal_primary_suspect"
    elif deltas["foreground_only"]["mean_foreground_dice"] <= -0.01:
        decision = "shared_foreground_or_decoder_pressure_primary_suspect"
    else:
        decision = "inconclusive_calibration_attribution"
    return {
        "decision": decision,
        "full_exp80_mean_dice_delta": full_dice,
        "full_exp80_diffuse_mean_dice_delta": diffuse("full_exp80"),
        "rescue_without_conditional_dice_mean_dice": no_dice_rescue,
        "rescue_without_conditional_dice_diffuse_mean_dice": no_dice_diffuse_rescue,
        "rescue_without_conditional_focal_mean_dice": no_focal_rescue,
        "rescue_without_conditional_focal_diffuse_mean_dice": no_focal_diffuse_rescue,
        "foreground_only_mean_dice_delta": float(
            deltas["foreground_only"]["mean_foreground_dice"]
        ),
        "failure_reproduction_threshold": -0.01,
        "mean_dice_rescue_threshold": 0.01,
        "diffuse_mean_dice_rescue_threshold": 0.02,
        "primary_separation_threshold": 0.005,
    }


def _evaluate_calibration(
    model: torch.nn.Module,
    calibration_loader: Any,
    truth: Any,
    device: torch.device,
) -> dict[str, Any]:
    predictions = _predict_slices(model, calibration_loader, device=device)
    _, summary = summarize_segmentation_predictions(predictions, truth)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--update-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    args = parser.parse_args()
    if args.update_steps != 4 or args.batch_size != 16:
        raise ValueError("Exp82 is locked to four updates and batch size 16")
    if args.learning_rate != 5e-5:
        raise ValueError("Exp82 is locked to learning rate 5e-5")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp82 requires CUDA BF16")

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    if int(config.get("outer_fold", -1)) != 2 or int(config.get("calibration_fold", -1)) != 1:
        raise ValueError("Exp82 is locked to outer=2/calibration=1")
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    (
        train_loader,
        calibration_loader,
        _,
        train_frame,
        _,
        _,
    ) = create_segmentation_loaders(
        args.manifest_path,
        outer_fold=2,
        calibration_fold=1,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=seed,
        sampler_study_balance_power=0.0,
    )
    update_batches = []
    for batch in train_loader:
        update_batches.append(_clone_batch(batch))
        if len(update_batches) == args.update_steps:
            break
    if len(update_batches) != args.update_steps:
        raise ValueError("Could not collect the locked four update batches")

    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    device = torch.device("cuda")
    loss_fn = _build_loss(train_frame, config, device)
    baseline_model = _build_factorized_model(args.checkpoint, config, device)
    baseline_summary = _evaluate_calibration(
        baseline_model, calibration_loader, truth, device
    )
    del baseline_model
    torch.cuda.empty_cache()

    variant_summaries: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for variant in VARIANT_COMPONENTS:
        _seed_everything(seed)
        model = _build_factorized_model(args.checkpoint, config, device)
        parameters = factorized_trainable_parameters(model)
        optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
        set_factorized_training_mode(model)
        update_losses = []
        for batch in update_batches:
            optimizer.zero_grad(set_to_none=True)
            _, _, _, components = _forward_loss_components(
                model, loss_fn, batch, device, args.precision
            )
            objective = _variant_objective(components, variant)
            objective.backward()
            optimizer.step()
            update_losses.append(float(objective.detach().cpu()))
        summary = _evaluate_calibration(model, calibration_loader, truth, device)
        summary["checkpoint_score"] = checkpoint_selection_score(
            summary, "fpr_volume_penalized"
        )
        summary["mean_update_objective"] = float(np.mean(update_losses))
        variant_summaries[variant] = summary
        deltas[variant] = _summary_delta(baseline_summary, summary)
        del model, optimizer
        torch.cuda.empty_cache()

    baseline_summary["checkpoint_score"] = checkpoint_selection_score(
        baseline_summary, "fpr_volume_penalized"
    )
    attribution = calibration_attribution_decision(deltas)
    result = {
        "analysis_kind": "factorized_exp80_four_step_calibration_attribution",
        "decision": attribution["decision"],
        "diagnostic_only_no_model_or_checkpoint_promotion": True,
        "evaluation_scope": "ich_only_calibration_fold1_outer_fold2_untouched",
        "aggregate_only_no_row_level_medical_predictions": True,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": manifest_sha,
        "metadata_source": str(metadata_source),
        "seed": seed,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "update_steps": args.update_steps,
        "learning_rate": args.learning_rate,
        "baseline_summary": baseline_summary,
        "variant_summaries": variant_summaries,
        "deltas_vs_baseline": deltas,
        "causal_attribution": attribution,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "calibration_attribution.json"
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        result["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "factorized_loss_calibration_attribution",
                "evaluation_scope": result["evaluation_scope"],
                "git_commit": result["git_commit"],
                "decision": result["decision"],
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "batch_size": args.batch_size,
                "update_steps": args.update_steps,
                "learning_rate": args.learning_rate,
                "precision": args.precision,
                "seed": seed,
            }
        )
        mlflow.log_metrics(
            {
                "full_mean_dice_delta": attribution["full_exp80_mean_dice_delta"],
                "full_diffuse_mean_dice_delta": attribution[
                    "full_exp80_diffuse_mean_dice_delta"
                ],
                "rescue_no_conditional_dice": attribution[
                    "rescue_without_conditional_dice_mean_dice"
                ],
                "rescue_no_conditional_focal": attribution[
                    "rescue_without_conditional_focal_mean_dice"
                ],
            }
        )
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path))
    notify_campaign(
        "completion",
        (
            "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
            "🧪 نسبت‌دادن افت calibration در چهار update تمام شد. "
            f"نتیجه: {result['decision']}؛ افت Dice مسیر کامل="
            f"{attribution['full_exp80_mean_dice_delta']:+.4f}، نجات با حذف Dice شرطی="
            f"{attribution['rescue_without_conditional_dice_mean_dice']:+.4f} و "
            "نجات با حذف focal شرطی="
            f"{attribution['rescue_without_conditional_focal_mean_dice']:+.4f}.\n\n"
            "تحلیل کاربردی: outer دست‌نخورده است و هیچ checkpointی ساخته نشده؛ "
            "این نتیجه فقط تعیین می‌کند recipe بعدی کدام جزء را باید حذف یا مهار کند."
        ),
        run=args.run_name,
        decision=result["decision"],
        mlflow=result.get("mlflow_run_id", "n/a"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
