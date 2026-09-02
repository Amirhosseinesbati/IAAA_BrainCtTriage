"""Local-only early trajectory and parameter-scope attribution after Exp84."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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
    _forward_loss_components,
    _seed_everything,
    _variant_objective,
)
from scripts.train_ich_factorized_residual_heads_local import (
    _candidate_config,
    _write_json,
    residual_head_screen_gate,
)
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_model import (
    FactorizedForegroundSubtypeModel,
    factorized_residual_head_parameters,
)
from src.strategies.ich_2p5d.segmentation_train import checkpoint_selection_score
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import file_sha256, git_commit


MILESTONES = (4, 16, 32, 64, 128, 192, 303)
SCOPES = ("both", "foreground_only", "subtype_only")
REPRODUCTION_METRICS = (
    "checkpoint_score",
    "selection_score",
    "mean_foreground_dice",
    "normal_false_positive_rate_at_0_1ml",
    "presence_f1_at_0_1ml",
    "total_volume_mae_ml",
    "total_volume_bias_ml",
)


def _scope_parameters(
    model: FactorizedForegroundSubtypeModel, scope: str
) -> list[torch.nn.Parameter]:
    if scope == "both":
        return factorized_residual_head_parameters(model)
    model.requires_grad_(False)
    if scope == "foreground_only":
        head = model.foreground_residual_head
    elif scope == "subtype_only":
        head = model.subtype_residual_head
    else:
        raise ValueError(f"Unknown residual parameter scope: {scope}")
    head.requires_grad_(True)
    parameters = list(head.parameters())
    if not parameters:
        raise ValueError(f"No parameters found for scope {scope}")
    return parameters


def _scope_training_mode(
    model: FactorizedForegroundSubtypeModel, scope: str
) -> None:
    model.eval()
    if scope in ("both", "foreground_only"):
        model.foreground_residual_head.train()
    if scope in ("both", "subtype_only"):
        model.subtype_residual_head.train()


def _batch_identity(batch: dict[str, Any]) -> str:
    payload = [
        [str(study), int(index)]
        for study, index in zip(batch["study_id"], batch["slice_index"])
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _summary_difference(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    differences = {
        name: float(candidate[name] - reference[name])
        for name in REPRODUCTION_METRICS
    }
    differences["subtypes"] = {
        name: float(
            candidate["subtypes"][name]["dice_known_pixels"]
            - reference["subtypes"][name]["dice_known_pixels"]
        )
        for name in ("IVH", "IPH", "SDH", "EDH", "SAH")
    }
    absolute = [abs(value) for value in differences.values() if isinstance(value, float)]
    absolute.extend(abs(value) for value in differences["subtypes"].values())
    differences["maximum_absolute_difference"] = max(absolute)
    return differences


def _reference_summary(path: Path, *, epoch: int | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if epoch is None:
        return payload["candidate_summary"]
    record = next(item for item in payload if int(item["epoch"]) == epoch)
    return record["calibration_summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--exp83-result", required=True, type=Path)
    parser.add_argument("--exp84-history", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.batch_size != 16
        or args.learning_rate != 5e-5
        or args.weight_decay != 1e-4
    ):
        raise ValueError("Exp85 is locked to batch=16, lr=5e-5, wd=1e-4")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp85 requires CUDA BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "scope_trajectory.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint_payload, dict) or not isinstance(
        checkpoint_payload.get("config"), dict
    ):
        raise ValueError("Checkpoint must contain a standard config")
    source_config = checkpoint_payload["config"]
    if int(source_config.get("outer_fold", -1)) != 2 or int(
        source_config.get("calibration_fold", -1)
    ) != 1:
        raise ValueError("Exp85 is locked to outer=2/calibration=1")
    seed = int(source_config.get("seed", 42))
    locked = argparse.Namespace(
        run_name="ich-exp85-factorized-residual-scope-trajectory-v1",
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        checkpoint=args.checkpoint,
        epochs=1,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_config = _candidate_config(source_config, locked)
    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    _seed_everything(seed)
    (
        _,
        baseline_loader,
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
    loss_fn = _build_loss(train_frame, loss_config, device)
    baseline_model = _build_factorized_model(args.checkpoint, source_config, device)
    baseline = _evaluate_calibration(baseline_model, baseline_loader, truth, device)
    baseline["checkpoint_score"] = checkpoint_selection_score(
        baseline, "fpr_volume_penalized"
    )
    del baseline_model, baseline_loader
    torch.cuda.empty_cache()

    trajectories: dict[str, list[dict[str, Any]]] = {}
    scope_parameter_counts: dict[str, int] = {}
    batch_identities: dict[str, dict[str, str]] = {}
    started = time.perf_counter()
    for scope in SCOPES:
        _seed_everything(seed)
        train_loader, calibration_loader, _, _, _, _ = create_segmentation_loaders(
            args.manifest_path,
            outer_fold=2,
            calibration_fold=1,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=seed,
            sampler_study_balance_power=0.0,
        )
        model = _build_factorized_model(args.checkpoint, source_config, device)
        parameters = _scope_parameters(model, scope)
        scope_parameter_counts[scope] = sum(p.numel() for p in parameters)
        optimizer = AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        records: list[dict[str, Any]] = []
        identities: dict[str, str] = {}
        objective_values: list[float] = []
        _scope_training_mode(model, scope)
        for step, batch in enumerate(train_loader, start=1):
            if step in MILESTONES or step == 1:
                identities[str(step)] = _batch_identity(batch)
            optimizer.zero_grad(set_to_none=True)
            _, _, _, components = _forward_loss_components(
                model, loss_fn, batch, device, "bf16"
            )
            objective = _variant_objective(components, "full_exp80")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            objective_values.append(float(objective.detach().cpu()))
            if step in MILESTONES:
                summary = _evaluate_calibration(
                    model, calibration_loader, truth, device
                )
                summary["checkpoint_score"] = checkpoint_selection_score(
                    summary, "fpr_volume_penalized"
                )
                record = {
                    "step": step,
                    "mean_objective_to_step": float(np.mean(objective_values)),
                    "calibration_summary": summary,
                    "delta_vs_baseline": _summary_delta(baseline, summary),
                    "screen_gate": residual_head_screen_gate(baseline, summary),
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "scope": scope,
                            "step": step,
                            "score": summary["checkpoint_score"],
                            "mean_dice": summary["mean_foreground_dice"],
                            "sdh": summary["subtypes"]["SDH"]["dice_known_pixels"],
                            "sah": summary["subtypes"]["SAH"]["dice_known_pixels"],
                            "mae": summary["total_volume_mae_ml"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                _scope_training_mode(model, scope)
            if step >= MILESTONES[-1]:
                break
        if len(records) != len(MILESTONES):
            raise ValueError(
                f"Scope {scope} yielded {len(records)} milestones, expected {len(MILESTONES)}"
            )
        trajectories[scope] = records
        batch_identities[scope] = identities
        del model, optimizer, train_loader, calibration_loader
        torch.cuda.empty_cache()

    exp83_reference = _reference_summary(args.exp83_result)
    exp84_epoch1_reference = _reference_summary(args.exp84_history, epoch=1)
    both_by_step = {item["step"]: item for item in trajectories["both"]}
    reproduction = {
        "exp83_step4": _summary_difference(
            both_by_step[4]["calibration_summary"], exp83_reference
        ),
        "exp84_epoch1_step303": _summary_difference(
            both_by_step[303]["calibration_summary"], exp84_epoch1_reference
        ),
    }
    identities_match = all(
        batch_identities[scope] == batch_identities["both"] for scope in SCOPES
    )
    reproduction_passed = identities_match and all(
        item["maximum_absolute_difference"] <= 1e-6
        for item in reproduction.values()
    )
    candidates = [
        {"scope": scope, **record}
        for scope, records in trajectories.items()
        for record in records
    ]
    best = max(
        candidates,
        key=lambda item: float(item["calibration_summary"]["checkpoint_score"]),
    )
    passing = [item for item in candidates if item["screen_gate"]["all_passed"]]
    if not reproduction_passed:
        decision = "reproduction_failed_no_scientific_interpretation"
    elif passing:
        decision = "early_scope_candidate_exists_preregister_replication"
    elif float(best["calibration_summary"]["checkpoint_score"]) > float(
        baseline["checkpoint_score"]
    ):
        decision = "score_only_early_scope_signal_requires_new_gate"
    else:
        decision = "no_early_or_isolated_scope_candidate"
    result = {
        "analysis_kind": "factorized_residual_early_trajectory_and_scope_attribution",
        "decision": decision,
        "diagnostic_only_no_model_or_checkpoint": True,
        "evaluation_scope": "ich_only_calibration_fold1_outer_fold2_untouched",
        "aggregate_only_no_row_level_medical_predictions": True,
        "external_reporting_enabled": False,
        "outer_evaluation_performed": False,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": file_sha256(args.manifest_path),
        "exp83_result_sha256": file_sha256(args.exp83_result),
        "exp84_history_sha256": file_sha256(args.exp84_history),
        "metadata_source": str(metadata_source),
        "seed": seed,
        "precision": "bf16",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "milestones": list(MILESTONES),
        "scopes": list(SCOPES),
        "scope_parameter_counts": scope_parameter_counts,
        "batch_identities": batch_identities,
        "batch_identities_match_across_scopes": identities_match,
        "reproduction": reproduction,
        "reproduction_passed": reproduction_passed,
        "baseline_summary": baseline,
        "trajectories": trajectories,
        "best_observed": best,
        "passing_candidate_count": len(passing),
        "train_slices": len(train_frame),
        "calibration_slices": len(calibration_frame),
        "outer_slices_not_evaluated": len(outer_frame),
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    if not all(
        math.isfinite(float(item["calibration_summary"]["checkpoint_score"]))
        for item in candidates
    ):
        raise ValueError("Non-finite checkpoint score in Exp85")
    _write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
