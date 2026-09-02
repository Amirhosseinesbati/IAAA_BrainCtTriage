"""Four-step local-only calibration gate for factorized residual heads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from scripts.diagnose_ich_factorized_calibration_attribution import (
    _evaluate_calibration,
    _summary_delta,
)
from scripts.diagnose_ich_factorized_loss_causality import (
    _build_factorized_model,
    _build_loss,
    _clone_batch,
    _forward_loss_components,
    _seed_everything,
    _variant_objective,
)
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_model import (
    factorized_residual_head_parameters,
    set_factorized_residual_head_training_mode,
)
from src.strategies.ich_2p5d.segmentation_train import checkpoint_selection_score
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import file_sha256, git_commit


def residual_head_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    required = [
        baseline["mean_foreground_dice"],
        candidate["mean_foreground_dice"],
        baseline["normal_false_positive_rate_at_0_1ml"],
        candidate["normal_false_positive_rate_at_0_1ml"],
        baseline["presence_f1_at_0_1ml"],
        candidate["presence_f1_at_0_1ml"],
        baseline["total_volume_mae_ml"],
        candidate["total_volume_mae_ml"],
        baseline["total_volume_bias_ml"],
        candidate["total_volume_bias_ml"],
        *(
            summary["subtypes"][name]["dice_known_pixels"]
            for summary in (baseline, candidate)
            for name in ("SDH", "EDH", "SAH")
        ),
    ]
    gates = {
        "all_required_aggregates_finite": all(
            value is not None and math.isfinite(float(value)) for value in required
        ),
        "mean_dice_drop_at_most_0_005": (
            candidate["mean_foreground_dice"]
            >= baseline["mean_foreground_dice"] - 0.005
        ),
        "sdh_dice_drop_at_most_0_005": (
            candidate["subtypes"]["SDH"]["dice_known_pixels"]
            >= baseline["subtypes"]["SDH"]["dice_known_pixels"] - 0.005
        ),
        "edh_dice_drop_at_most_0_005": (
            candidate["subtypes"]["EDH"]["dice_known_pixels"]
            >= baseline["subtypes"]["EDH"]["dice_known_pixels"] - 0.005
        ),
        "sah_dice_drop_at_most_0_005": (
            candidate["subtypes"]["SAH"]["dice_known_pixels"]
            >= baseline["subtypes"]["SAH"]["dice_known_pixels"] - 0.005
        ),
        "normal_fpr_noninferior": (
            candidate["normal_false_positive_rate_at_0_1ml"]
            <= baseline["normal_false_positive_rate_at_0_1ml"]
        ),
        "presence_f1_noninferior": (
            candidate["presence_f1_at_0_1ml"]
            >= baseline["presence_f1_at_0_1ml"]
        ),
        "volume_mae_increase_at_most_0_5ml": (
            candidate["total_volume_mae_ml"]
            <= baseline["total_volume_mae_ml"] + 0.5
        ),
        "absolute_volume_bias_increase_at_most_0_5ml": (
            abs(candidate["total_volume_bias_ml"])
            <= abs(baseline["total_volume_bias_ml"]) + 0.5
        ),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--update-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    if args.update_steps != 4 or args.batch_size != 16:
        raise ValueError("Exp83 is locked to four updates and batch size 16")
    if args.learning_rate != 5e-5:
        raise ValueError("Exp83 is locked to learning rate 5e-5")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp83 requires CUDA BF16")

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    if int(config.get("outer_fold", -1)) != 2 or int(config.get("calibration_fold", -1)) != 1:
        raise ValueError("Exp83 is locked to outer=2/calibration=1")
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
    baseline = _evaluate_calibration(
        baseline_model, calibration_loader, truth, device
    )
    del baseline_model
    torch.cuda.empty_cache()

    _seed_everything(seed)
    model = _build_factorized_model(args.checkpoint, config, device)
    parameters = factorized_residual_head_parameters(model)
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    set_factorized_residual_head_training_mode(model)
    losses = []
    for batch in update_batches:
        optimizer.zero_grad(set_to_none=True)
        _, _, _, components = _forward_loss_components(
            model, loss_fn, batch, device, "bf16"
        )
        objective = _variant_objective(components, "full_exp80")
        objective.backward()
        optimizer.step()
        losses.append(float(objective.detach().cpu()))
    candidate = _evaluate_calibration(model, calibration_loader, truth, device)
    del model, optimizer
    torch.cuda.empty_cache()

    baseline["checkpoint_score"] = checkpoint_selection_score(
        baseline, "fpr_volume_penalized"
    )
    candidate["checkpoint_score"] = checkpoint_selection_score(
        candidate, "fpr_volume_penalized"
    )
    candidate["mean_update_objective"] = float(np.mean(losses))
    gates = residual_head_gate(baseline, candidate)
    result = {
        "analysis_kind": "factorized_residual_heads_four_step_calibration_gate",
        "decision": (
            "authorize_residual_head_only_three_epoch_calibration_screen"
            if gates["all_passed"]
            else "reject_residual_head_only_before_full_calibration_or_outer"
        ),
        "diagnostic_only_no_model_or_checkpoint_promotion": True,
        "evaluation_scope": "ich_only_calibration_fold1_outer_fold2_untouched",
        "aggregate_only_no_row_level_medical_predictions": True,
        "external_reporting_enabled": False,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": manifest_sha,
        "metadata_source": str(metadata_source),
        "seed": seed,
        "precision": "bf16",
        "batch_size": args.batch_size,
        "update_steps": args.update_steps,
        "learning_rate": args.learning_rate,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "baseline_summary": baseline,
        "candidate_summary": candidate,
        "deltas_vs_baseline": _summary_delta(baseline, candidate),
        "preregistered_gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "residual_head_gate.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
