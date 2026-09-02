"""Local-only three-epoch screen of factorized ICH residual heads.

This deliberately avoids MLflow, Telegram, row-level prediction artifacts and
outer-fold inference.  It is the full-calibration follow-up authorized by the
four-update Exp83 safety gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.diagnose_ich_factorized_calibration_attribution import (
    _evaluate_calibration,
    _summary_delta,
)
from scripts.diagnose_ich_factorized_loss_causality import (
    _build_factorized_model,
    _build_loss,
    _forward_loss_components,
    _seed_everything,
    _variant_objective,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_model import (
    factorized_residual_head_parameters,
    set_factorized_residual_head_training_mode,
)
from src.strategies.ich_2p5d.segmentation_train import (
    CHECKPOINT_SELECTION_METRICS,
    checkpoint_selection_score,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import file_sha256, git_commit


def residual_head_screen_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Locked Exp84 replication gate; every condition is conjunctive."""

    tolerance = 1e-12
    required = [
        baseline["checkpoint_score"],
        candidate["checkpoint_score"],
        *(
            summary[name]
            for summary in (baseline, candidate)
            for name in (
                "selection_score",
                "mean_foreground_dice",
                "any_ich_study_auc",
                "macro_subtype_study_auc",
                "presence_f1_at_0_1ml",
                "normal_false_positive_rate_at_0_1ml",
                "total_volume_mae_ml",
                "total_volume_bias_ml",
            )
        ),
        *(
            summary["subtypes"][name]["dice_known_pixels"]
            for summary in (baseline, candidate)
            for name in ("IVH", "IPH", "SDH", "EDH", "SAH")
        ),
    ]
    gates = {
        "all_required_aggregates_finite": all(
            value is not None and math.isfinite(float(value)) for value in required
        ),
        "checkpoint_score_gain_at_least_0_003": (
            candidate["checkpoint_score"] + tolerance
            >= baseline["checkpoint_score"] + 0.003
        ),
        "selection_gain_at_least_0_003": (
            candidate["selection_score"] + tolerance
            >= baseline["selection_score"] + 0.003
        ),
        "mean_dice_gain_at_least_0_005": (
            candidate["mean_foreground_dice"] + tolerance
            >= baseline["mean_foreground_dice"] + 0.005
        ),
        "sah_dice_gain_at_least_0_010": (
            candidate["subtypes"]["SAH"]["dice_known_pixels"] + tolerance
            >= baseline["subtypes"]["SAH"]["dice_known_pixels"] + 0.010
        ),
        "sdh_dice_gain_at_least_0_005": (
            candidate["subtypes"]["SDH"]["dice_known_pixels"] + tolerance
            >= baseline["subtypes"]["SDH"]["dice_known_pixels"] + 0.005
        ),
        "ivh_dice_drop_at_most_0_005": (
            candidate["subtypes"]["IVH"]["dice_known_pixels"] + tolerance
            >= baseline["subtypes"]["IVH"]["dice_known_pixels"] - 0.005
        ),
        "iph_dice_drop_at_most_0_005": (
            candidate["subtypes"]["IPH"]["dice_known_pixels"] + tolerance
            >= baseline["subtypes"]["IPH"]["dice_known_pixels"] - 0.005
        ),
        "edh_dice_drop_at_most_0_005": (
            candidate["subtypes"]["EDH"]["dice_known_pixels"] + tolerance
            >= baseline["subtypes"]["EDH"]["dice_known_pixels"] - 0.005
        ),
        "normal_fpr_noninferior": (
            candidate["normal_false_positive_rate_at_0_1ml"]
            <= baseline["normal_false_positive_rate_at_0_1ml"] + tolerance
        ),
        "presence_f1_noninferior": (
            candidate["presence_f1_at_0_1ml"] + tolerance
            >= baseline["presence_f1_at_0_1ml"]
        ),
        "volume_mae_noninferior": (
            candidate["total_volume_mae_ml"]
            <= baseline["total_volume_mae_ml"] + tolerance
        ),
        "absolute_volume_bias_noninferior": (
            abs(candidate["total_volume_bias_ml"])
            <= abs(baseline["total_volume_bias_ml"]) + tolerance
        ),
        "any_auc_unchanged": abs(
            candidate["any_ich_study_auc"] - baseline["any_ich_study_auc"]
        ) <= tolerance,
        "macro_subtype_auc_unchanged": abs(
            candidate["macro_subtype_study_auc"]
            - baseline["macro_subtype_study_auc"]
        ) <= tolerance,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _candidate_config(
    source: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Return a standard training config that can rebuild the saved wrapper."""

    result = dict(source)
    result.update(
        {
            "run_name": args.run_name,
            "output_dir": str(args.output_dir),
            "manifest_path": str(args.manifest_path),
            "outer_fold": 2,
            "calibration_fold": 1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "classification_loss_weight": 0.0,
            "checkpoint_selection_strategy": "fpr_volume_penalized",
            "initial_checkpoint": str(args.checkpoint),
            "pretrained": False,
            "factorized_output_head": True,
            "freeze_base_model": False,
            "classification_head_only": False,
            "segmentation_objective": "hierarchical_foreground_subtype",
            "foreground_dice_weight": 0.325,
            "foreground_focal_weight": 0.175,
            "conditional_subtype_weight": 0.175,
            "conditional_subtype_dice_weight": 0.325,
            "conditional_subtype_focal_gamma": 2.0,
            "subtype_ovr_weight": 0.0,
            "conditional_subtype_mode": "cross_entropy",
            "foreground_gradient_mode": "probability_weighted",
            "empty_foreground_weight": 0.05,
            "empty_foreground_top_fraction": 0.001,
            "sampler_study_balance_power": 0.0,
            "max_train_steps": None,
            "evaluate_outer": False,
        }
    )
    return result


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _save_candidate(
    path: Path,
    model: torch.nn.Module,
    *,
    config: dict[str, Any],
    epoch: int,
    calibration_summary: dict[str, Any],
    manifest_sha256: str,
    initial_checkpoint_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 1,
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "output_labels": OUTPUT_LABELS,
            "segmentation_classes": 6,
            "input_channels": 9,
            "selection_metric": CHECKPOINT_SELECTION_METRICS[
                "fpr_volume_penalized"
            ],
            "calibration_summary": calibration_summary,
            "manifest_sha256": manifest_sha256,
            "hard_negative_manifest_sha256": None,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "training_scope": "factorized_residual_heads_only_870_parameters",
            "external_reporting_enabled": False,
            "git_commit": git_commit(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.epochs != 3
        or args.batch_size != 16
        or args.learning_rate != 5e-5
        or args.weight_decay != 1e-4
    ):
        raise ValueError(
            "Exp84 is locked to epochs=3, batch=16, lr=5e-5, weight_decay=1e-4"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp84 requires CUDA BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pth"
    summary_path = args.output_dir / "run_summary.json"
    if checkpoint_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Exp84 run in {args.output_dir}"
        )

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a standard training config")
    source_config = payload["config"]
    if int(source_config.get("outer_fold", -1)) != 2 or int(
        source_config.get("calibration_fold", -1)
    ) != 1:
        raise ValueError("Exp84 is locked to outer=2/calibration=1")
    seed = int(source_config.get("seed", 42))
    candidate_config = _candidate_config(source_config, args)
    _write_json(args.output_dir / "resolved_config.json", candidate_config)

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
    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    loss_fn = _build_loss(train_frame, candidate_config, device)
    model = _build_factorized_model(args.checkpoint, source_config, device)
    parameters = factorized_residual_head_parameters(model)
    parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    if trainable_parameter_count != 870:
        raise ValueError(
            f"Exp84 expected exactly 870 trainable parameters, got {trainable_parameter_count}"
        )

    baseline = _evaluate_calibration(model, calibration_loader, truth, device)
    baseline["checkpoint_score"] = checkpoint_selection_score(
        baseline, "fpr_volume_penalized"
    )
    optimizer = AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "learning_rate": args.learning_rate,
            "calibration_summary": baseline,
            "delta_vs_baseline": _summary_delta(baseline, baseline),
            "screen_gate": residual_head_screen_gate(baseline, baseline),
        }
    ]
    best_epoch = 0
    best_score = float(baseline["checkpoint_score"])
    best_summary = baseline
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        set_factorized_residual_head_training_mode(model)
        component_history: dict[str, list[float]] = {
            name: []
            for name in (
                "foreground_support",
                "conditional_focal",
                "conditional_dice",
                "full_exp80",
            )
        }
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            _, _, _, components = _forward_loss_components(
                model, loss_fn, batch, device, "bf16"
            )
            objective = _variant_objective(components, "full_exp80")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            for name in component_history:
                component_history[name].append(
                    float(components[name].detach().cpu())
                )
            if step % 50 == 0:
                print(
                    f"epoch={epoch}/3 step={step}/{len(train_loader)} "
                    f"loss={np.mean(component_history['full_exp80'][-50:]):.6f}",
                    flush=True,
                )
        scheduler.step()
        candidate = _evaluate_calibration(model, calibration_loader, truth, device)
        candidate["checkpoint_score"] = checkpoint_selection_score(
            candidate, "fpr_volume_penalized"
        )
        delta = _summary_delta(baseline, candidate)
        gate = residual_head_screen_gate(baseline, candidate)
        epoch_record = {
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": float(scheduler.get_last_lr()[0]),
            "mean_train_components": {
                name: float(np.mean(values))
                for name, values in component_history.items()
            },
            "calibration_summary": candidate,
            "delta_vs_baseline": delta,
            "screen_gate": gate,
        }
        history.append(epoch_record)
        _write_json(args.output_dir / "history.json", history)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        score = float(candidate["checkpoint_score"])
        if score > best_score + 1e-6:
            best_epoch = epoch
            best_score = score
            best_summary = candidate
            _save_candidate(
                checkpoint_path,
                model,
                config=candidate_config,
                epoch=epoch,
                calibration_summary=candidate,
                manifest_sha256=manifest_sha,
                initial_checkpoint_sha256=checkpoint_sha,
            )
            _write_json(
                args.output_dir / "best_calibration_summary.json", candidate
            )

    duration = time.perf_counter() - started
    best_gate = residual_head_screen_gate(baseline, best_summary)
    candidate_saved = checkpoint_path.is_file()
    if best_gate["all_passed"]:
        decision = "authorize_multiseed_replication_before_outer"
    elif candidate_saved:
        decision = "retain_local_experimental_candidate_no_outer"
    else:
        decision = "reject_residual_head_only_three_epoch_screen"
    result = {
        "analysis_kind": "factorized_residual_heads_three_epoch_calibration_screen",
        "decision": decision,
        "evaluation_scope": "ich_only_calibration_fold1_outer_fold2_untouched",
        "aggregate_only_no_row_level_medical_predictions": True,
        "external_reporting_enabled": False,
        "outer_evaluation_performed": False,
        "checkpoint_promotion_performed": False,
        "candidate_checkpoint_saved": candidate_saved,
        "candidate_checkpoint": str(checkpoint_path) if candidate_saved else None,
        "candidate_checkpoint_sha256": (
            file_sha256(checkpoint_path) if candidate_saved else None
        ),
        "initial_checkpoint": str(args.checkpoint),
        "initial_checkpoint_sha256": checkpoint_sha,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": manifest_sha,
        "metadata_source": str(metadata_source),
        "git_commit": git_commit(),
        "seed": seed,
        "precision": "bf16",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_slices": len(train_frame),
        "calibration_slices": len(calibration_frame),
        "outer_slices_not_evaluated": len(outer_frame),
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_names": parameter_names,
        "best_epoch": best_epoch,
        "baseline_summary": baseline,
        "best_summary": best_summary,
        "best_delta_vs_baseline": _summary_delta(baseline, best_summary),
        "preregistered_replication_gate": best_gate,
        "duration_s": duration,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "history_file": str(args.output_dir / "history.json"),
    }
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
