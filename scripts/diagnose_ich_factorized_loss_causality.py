"""Train-only causal probe for the factorized Exp80 ICH objective."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW

from scripts.diagnose_ich_multitask_gradient_conflict import _gradient_geometry
from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_foreground_counts,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    HierarchicalForegroundSubtypeLoss,
)
from src.strategies.ich_2p5d.segmentation_model import (
    FactorizedForegroundSubtypeModel,
    base_segmentation_model,
    build_segmentation_model,
    factorized_trainable_parameters,
    load_segmentation_weights,
    set_factorized_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


SUBTYPE_NAMES = ("IVH", "IPH", "SDH", "EDH", "SAH")
COMPONENT_NAMES = (
    "foreground_support",
    "conditional_focal",
    "conditional_dice",
    "full_exp80",
)
VARIANT_COMPONENTS = {
    "full_exp80": ("foreground_support", "conditional_focal", "conditional_dice"),
    "without_conditional_dice": ("foreground_support", "conditional_focal"),
    "without_conditional_focal": ("foreground_support", "conditional_dice"),
    "foreground_only": ("foreground_support",),
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _objective_components(
    raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    foreground = (
        0.325 * raw["foreground_dice"]
        + 0.175 * raw["foreground_focal"]
        + 0.05 * raw["empty_foreground"]
    )
    conditional_focal = 0.175 * raw["conditional_subtype"]
    conditional_dice = 0.325 * raw["conditional_subtype_dice"]
    return {
        "foreground_support": foreground,
        "conditional_focal": conditional_focal,
        "conditional_dice": conditional_dice,
        "full_exp80": foreground + conditional_focal + conditional_dice,
    }


def _variant_objective(
    components: dict[str, torch.Tensor], variant: str
) -> torch.Tensor:
    names = VARIANT_COMPONENTS[variant]
    return sum((components[name] for name in names), start=components[names[0]] * 0.0)


def _parameter_group(name: str) -> str:
    if name.startswith("base_model.decoder."):
        return "decoder"
    if name.startswith("base_model.segmentation_head."):
        return "legacy_segmentation_head"
    if name.startswith("foreground_residual_head."):
        return "foreground_residual_head"
    if name.startswith("subtype_residual_head."):
        return "subtype_residual_head"
    raise ValueError(f"Unexpected trainable factorized parameter: {name}")


def _gradient_norm(
    gradients: tuple[torch.Tensor | None, ...], indices: list[int]
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for index in indices:
        gradient = gradients[index]
        if gradient is not None:
            total += gradient.detach().double().square().sum().cpu()
    return float(total.sqrt())


def _margin_attraction(
    gradient: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, tuple[float, int]]:
    subtype_gradient = gradient.detach().float()[:, 1:].movedim(1, -1)
    result: dict[str, tuple[float, int]] = {}
    for class_id, name in enumerate(SUBTYPE_NAMES, start=1):
        selected = valid & (masks == class_id)
        count = int(selected.sum().item())
        if not count:
            result[name] = (0.0, 0)
            continue
        values = subtype_gradient[selected]
        target_index = class_id - 1
        target = values[:, target_index]
        competitors = (values.sum(dim=1) - target) / 4.0
        result[name] = (float((competitors - target).sum().cpu()), count)
    return result


def _clone_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].detach().cpu().clone()
        for key in ("image", "mask", "segmentation_known")
    }


def _collect_batches(
    train_loader: Any,
    *,
    gradient_batches: int,
    update_steps: int,
    probe_batches: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    required = gradient_batches + update_steps + probe_batches
    selected: list[dict[str, torch.Tensor]] = []
    for batch in train_loader:
        known = batch["segmentation_known"] > 0.5
        foreground = (batch["mask"] > 0) & known[:, None, None]
        if torch.any(foreground):
            selected.append(_clone_batch(batch))
        if len(selected) >= required:
            break
    if len(selected) < required:
        raise ValueError(
            f"Only {len(selected)} foreground-containing batches found; {required} required"
        )
    return (
        selected[:gradient_batches],
        selected[gradient_batches : gradient_batches + update_steps],
        selected[gradient_batches + update_steps : required],
    )


def _build_factorized_model(
    checkpoint: Path,
    config: dict[str, Any],
    device: torch.device,
) -> FactorizedForegroundSubtypeModel:
    model = build_segmentation_model(
        architecture=str(config["architecture"]),
        encoder_name=str(config["encoder_name"]),
        pretrained=False,
        dropout=float(config["dropout"]),
        factorized_output_head=True,
    )
    if not isinstance(model, FactorizedForegroundSubtypeModel):
        raise TypeError("Builder did not return a factorized model")
    load_segmentation_weights(base_segmentation_model(model), checkpoint)
    factorized_trainable_parameters(model)
    return model.to(device)


def _build_loss(
    train_frame: Any,
    config: dict[str, Any],
    device: torch.device,
) -> HierarchicalForegroundSubtypeLoss:
    weights = segmentation_foreground_weights(
        train_frame,
        power=float(config.get("segmentation_class_weight_power", 1.0)),
        maximum=float(config.get("maximum_segmentation_class_weight", 8.0)),
        basis=str(config.get("segmentation_class_weight_basis", "pixel")),
    ).to(device)
    counts = segmentation_foreground_counts(
        train_frame,
        basis=str(config.get("segmentation_class_weight_basis", "pixel")),
    ).to(device)
    return HierarchicalForegroundSubtypeLoss(
        foreground_class_weights=weights,
        foreground_class_counts=counts,
        conditional_subtype_mode="cross_entropy",
        foreground_gradient_mode="probability_weighted",
        foreground_dice_weight=0.325,
        foreground_focal_weight=0.175,
        conditional_subtype_weight=0.175,
        conditional_subtype_dice_weight=0.325,
        conditional_subtype_focal_gamma=2.0,
        subtype_ovr_weight=0.0,
        background_weight=0.15,
        empty_foreground_weight=0.05,
        empty_foreground_top_fraction=0.001,
    ).to(device)


def _forward_loss_components(
    model: FactorizedForegroundSubtypeModel,
    loss_fn: HierarchicalForegroundSubtypeLoss,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    known = batch["segmentation_known"].to(device, non_blocking=True) > 0.5
    supervision = known[:, None, None].expand_as(masks)
    with _autocast(device, precision):
        logits = model.forward_components(images)["mask_logits"]
        raw = loss_fn.components(logits, masks, supervision)
        components = _objective_components(raw)
    return logits, masks, supervision, components


def _gradient_probe(
    model: FactorizedForegroundSubtypeModel,
    loss_fn: HierarchicalForegroundSubtypeLoss,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    set_factorized_training_mode(model)
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = tuple(parameter for _, parameter in named_parameters)
    groups: dict[str, list[int]] = {}
    for index, (name, _) in enumerate(named_parameters):
        groups.setdefault(_parameter_group(name), []).append(index)
    losses = {name: [] for name in COMPONENT_NAMES}
    norms = {
        name: {group: [] for group in groups}
        for name in COMPONENT_NAMES
    }
    cosines = {
        f"{left}__vs__{right}": []
        for left, right in combinations(COMPONENT_NAMES[:-1], 2)
    }
    attraction = {
        component: {
            subtype: {"sum": 0.0, "pixels": 0}
            for subtype in SUBTYPE_NAMES
        }
        for component in COMPONENT_NAMES
    }
    for batch in batches:
        logits, masks, supervision, components = _forward_loss_components(
            model, loss_fn, batch, device, precision
        )
        component_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for index, component_name in enumerate(COMPONENT_NAMES):
            gradients = torch.autograd.grad(
                components[component_name],
                (*parameters, logits),
                retain_graph=index < len(COMPONENT_NAMES) - 1,
                allow_unused=True,
            )
            parameter_gradients = gradients[:-1]
            logit_gradient = gradients[-1]
            if logit_gradient is None:
                raise RuntimeError(f"No logit gradient for {component_name}")
            component_gradients[component_name] = parameter_gradients
            losses[component_name].append(float(components[component_name].detach().cpu()))
            for group, indices in groups.items():
                norms[component_name][group].append(
                    _gradient_norm(parameter_gradients, indices)
                )
            per_class = _margin_attraction(
                logit_gradient, masks, supervision > 0.5
            )
            for subtype, (value, pixels) in per_class.items():
                attraction[component_name][subtype]["sum"] += value
                attraction[component_name][subtype]["pixels"] += pixels
        for left, right in combinations(COMPONENT_NAMES[:-1], 2):
            cosine, _, _ = _gradient_geometry(
                component_gradients[left], component_gradients[right]
            )
            if cosine is not None:
                cosines[f"{left}__vs__{right}"].append(cosine)
    return {
        "batches": len(batches),
        "mean_weighted_component_losses": {
            name: _finite_mean(values) for name, values in losses.items()
        },
        "mean_parameter_gradient_norms": {
            name: {
                group: _finite_mean(values)
                for group, values in group_values.items()
            }
            for name, group_values in norms.items()
        },
        "mean_pairwise_parameter_gradient_cosines": {
            name: _finite_mean(values) for name, values in cosines.items()
        },
        "mean_subtype_margin_attraction": {
            component: {
                subtype: (
                    values["sum"] / values["pixels"]
                    if values["pixels"]
                    else None
                )
                for subtype, values in subtype_values.items()
            }
            for component, subtype_values in attraction.items()
        },
        "subtype_supervised_pixels": {
            subtype: attraction["full_exp80"][subtype]["pixels"]
            for subtype in SUBTYPE_NAMES
        },
    }


@torch.no_grad()
def _probe_metrics(
    model: FactorizedForegroundSubtypeModel,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    totals = {
        name: {
            "observed": 0.0,
            "hard_predicted": 0.0,
            "hard_intersection": 0.0,
            "soft_predicted": 0.0,
            "soft_intersection": 0.0,
            "target_probability": 0.0,
            "conditional_target_probability": 0.0,
            "margin": 0.0,
        }
        for name in SUBTYPE_NAMES
    }
    background_fg_sum = 0.0
    background_pixels = 0
    true_fg_sum = 0.0
    true_fg_pixels = 0
    for batch in batches:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        known = batch["segmentation_known"].to(device, non_blocking=True) > 0.5
        valid = known[:, None, None].expand_as(masks)
        with _autocast(device, precision):
            logits = model.forward_components(images)["mask_logits"]
        logits = logits.float()
        probabilities = torch.softmax(logits, dim=1)
        conditional = torch.softmax(logits[:, 1:], dim=1)
        hard = logits.argmax(dim=1)
        foreground_probability = probabilities[:, 1:].sum(dim=1)
        background = valid & (masks == 0)
        true_foreground = valid & (masks > 0)
        background_fg_sum += float(foreground_probability[background].sum().cpu())
        background_pixels += int(background.sum().item())
        true_fg_sum += float(foreground_probability[true_foreground].sum().cpu())
        true_fg_pixels += int(true_foreground.sum().item())
        for class_id, name in enumerate(SUBTYPE_NAMES, start=1):
            target = valid & (masks == class_id)
            predicted = valid & (hard == class_id)
            observed = int(target.sum().item())
            values = totals[name]
            values["observed"] += observed
            values["hard_predicted"] += int(predicted.sum().item())
            values["hard_intersection"] += int((target & predicted).sum().item())
            values["soft_predicted"] += float(
                probabilities[:, class_id][valid].sum().cpu()
            )
            values["soft_intersection"] += float(
                probabilities[:, class_id][target].sum().cpu()
            )
            if observed:
                values["target_probability"] += float(
                    probabilities[:, class_id][target].sum().cpu()
                )
                values["conditional_target_probability"] += float(
                    conditional[:, class_id - 1][target].sum().cpu()
                )
                subtype_logits = logits[:, 1:].movedim(1, -1)[target]
                target_index = class_id - 1
                target_logits = subtype_logits[:, target_index]
                competitors = subtype_logits.clone()
                competitors[:, target_index] = -torch.inf
                values["margin"] += float(
                    (target_logits - competitors.max(dim=1).values).sum().cpu()
                )
    result: dict[str, Any] = {"subtypes": {}}
    for name, values in totals.items():
        observed = values["observed"]
        hard_denominator = observed + values["hard_predicted"]
        soft_denominator = observed + values["soft_predicted"]
        result["subtypes"][name] = {
            "observed_pixels": int(observed),
            "hard_dice": (
                2.0 * values["hard_intersection"] / hard_denominator
                if hard_denominator
                else None
            ),
            "soft_dice": (
                2.0 * values["soft_intersection"] / soft_denominator
                if soft_denominator
                else None
            ),
            "mean_target_probability": (
                values["target_probability"] / observed if observed else None
            ),
            "mean_conditional_target_probability": (
                values["conditional_target_probability"] / observed
                if observed
                else None
            ),
            "mean_target_vs_best_other_margin": (
                values["margin"] / observed if observed else None
            ),
            "predicted_soft_pixels": values["soft_predicted"],
        }
    result["mean_foreground_probability_on_true_background"] = (
        background_fg_sum / background_pixels if background_pixels else None
    )
    result["mean_foreground_probability_on_true_foreground"] = (
        true_fg_sum / true_fg_pixels if true_fg_pixels else None
    )
    result["background_pixels"] = background_pixels
    result["true_foreground_pixels"] = true_fg_pixels
    return result


def _metric_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"subtypes": {}}
    for name in SUBTYPE_NAMES:
        result["subtypes"][name] = {}
        for metric in (
            "hard_dice",
            "soft_dice",
            "mean_target_probability",
            "mean_conditional_target_probability",
            "mean_target_vs_best_other_margin",
            "predicted_soft_pixels",
        ):
            before = baseline["subtypes"][name][metric]
            after = candidate["subtypes"][name][metric]
            result["subtypes"][name][metric] = (
                float(after - before)
                if before is not None and after is not None
                else None
            )
    for metric in (
        "mean_foreground_probability_on_true_background",
        "mean_foreground_probability_on_true_foreground",
    ):
        before = baseline[metric]
        after = candidate[metric]
        result[metric] = (
            float(after - before)
            if before is not None and after is not None
            else None
        )
    return result


def causal_decision(update_deltas: dict[str, Any]) -> dict[str, Any]:
    def diffuse(variant: str) -> float:
        values = [
            update_deltas[variant]["subtypes"][name]["soft_dice"]
            for name in ("SDH", "SAH")
        ]
        if any(value is None for value in values):
            raise ValueError("SDH and SAH soft-Dice deltas are required")
        return float(np.mean(values))

    full = diffuse("full_exp80")
    without_dice = diffuse("without_conditional_dice")
    without_focal = diffuse("without_conditional_focal")
    rescue_without_dice = without_dice - full
    rescue_without_focal = without_focal - full
    if full >= -0.001:
        decision = "short_horizon_drift_not_reproduced"
    elif rescue_without_dice >= 0.005 and rescue_without_focal >= 0.005:
        decision = "conditional_loss_interaction_suspected"
    elif (
        rescue_without_dice >= 0.005
        and rescue_without_dice >= rescue_without_focal + 0.002
    ):
        decision = "conditional_dice_primary_suspect"
    elif (
        rescue_without_focal >= 0.005
        and rescue_without_focal >= rescue_without_dice + 0.002
    ):
        decision = "conditional_focal_primary_suspect"
    else:
        foreground = diffuse("foreground_only")
        decision = (
            "shared_foreground_or_decoder_pressure_suspected"
            if foreground < -0.001
            else "inconclusive_component_attribution"
        )
    return {
        "decision": decision,
        "full_exp80_diffuse_mean_soft_dice_delta": full,
        "without_conditional_dice_diffuse_mean_soft_dice_delta": without_dice,
        "without_conditional_focal_diffuse_mean_soft_dice_delta": without_focal,
        "foreground_only_diffuse_mean_soft_dice_delta": diffuse("foreground_only"),
        "rescue_without_conditional_dice": rescue_without_dice,
        "rescue_without_conditional_focal": rescue_without_focal,
        "minimum_rescue_threshold": 0.005,
        "minimum_separation_threshold": 0.002,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gradient-batches", type=int, default=24)
    parser.add_argument("--update-steps", type=int, default=8)
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if min(args.gradient_batches, args.update_steps, args.probe_batches, args.batch_size) <= 0:
        raise ValueError("Batch and step counts must be positive")
    if args.workers < 0 or args.learning_rate <= 0:
        raise ValueError("Workers and learning rate are invalid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.precision == "bf16" and args.device != "cuda":
        raise ValueError("BF16 requires CUDA")

    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    if int(config.get("outer_fold", -1)) != 2 or int(config.get("calibration_fold", -1)) != 1:
        raise ValueError("Exp81 is locked to outer=2/calibration=1")
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    loaders = create_segmentation_loaders(
        args.manifest_path,
        outer_fold=2,
        calibration_fold=1,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=seed,
        sampler_study_balance_power=0.0,
    )
    train_loader, _, _, train_frame = loaders[:4]
    gradient_batches, update_batches, probe_batches = _collect_batches(
        train_loader,
        gradient_batches=args.gradient_batches,
        update_steps=args.update_steps,
        probe_batches=args.probe_batches,
    )
    device = torch.device(args.device)
    loss_fn = _build_loss(train_frame, config, device)

    gradient_model = _build_factorized_model(args.checkpoint, config, device)
    gradient_summary = _gradient_probe(
        gradient_model, loss_fn, gradient_batches, device, args.precision
    )
    del gradient_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_model = _build_factorized_model(args.checkpoint, config, device)
    baseline_metrics = _probe_metrics(
        baseline_model, probe_batches, device, args.precision
    )
    del baseline_model
    update_results: dict[str, Any] = {}
    update_deltas: dict[str, Any] = {}
    for variant in VARIANT_COMPONENTS:
        _seed_everything(seed)
        model = _build_factorized_model(args.checkpoint, config, device)
        parameters = factorized_trainable_parameters(model)
        optimizer = AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        set_factorized_training_mode(model)
        losses: list[float] = []
        for batch in update_batches:
            optimizer.zero_grad(set_to_none=True)
            _, _, _, components = _forward_loss_components(
                model, loss_fn, batch, device, args.precision
            )
            objective = _variant_objective(components, variant)
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach().cpu()))
        metrics = _probe_metrics(model, probe_batches, device, args.precision)
        update_results[variant] = {
            "mean_update_objective": _finite_mean(losses),
            "probe_metrics": metrics,
        }
        update_deltas[variant] = _metric_deltas(baseline_metrics, metrics)
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attribution = causal_decision(update_deltas)
    result = {
        "analysis_kind": "factorized_exp80_loss_causality_train_only_probe",
        "decision": attribution["decision"],
        "diagnostic_only_no_model_promotion": True,
        "evaluation_scope": "ich_only_train_no_calibration_no_outer_no_oof",
        "aggregate_only_no_row_level_medical_predictions": True,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": manifest_sha,
        "seed": seed,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "gradient_batches": args.gradient_batches,
        "update_steps": args.update_steps,
        "probe_batches": args.probe_batches,
        "learning_rate": args.learning_rate,
        "gradient_summary": gradient_summary,
        "baseline_probe_metrics": baseline_metrics,
        "update_results": update_results,
        "update_deltas": update_deltas,
        "causal_attribution": attribution,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "causality_probe.json"
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        result["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "factorized_loss_causality_train_probe",
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
                "gradient_batches": args.gradient_batches,
                "update_steps": args.update_steps,
                "probe_batches": args.probe_batches,
                "learning_rate": args.learning_rate,
                "precision": args.precision,
                "seed": seed,
            }
        )
        mlflow.log_metrics(
            {
                "full_diffuse_soft_dice_delta": attribution[
                    "full_exp80_diffuse_mean_soft_dice_delta"
                ],
                "rescue_without_conditional_dice": attribution[
                    "rescue_without_conditional_dice"
                ],
                "rescue_without_conditional_focal": attribution[
                    "rescue_without_conditional_focal"
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
            "🔬 کالبدشکافی علت افت Exp80 تمام شد. "
            f"نتیجه: {result['decision']}؛ تغییر soft-Dice میانگین SAH/SDH "
            f"در objective کامل={attribution['full_exp80_diffuse_mean_soft_dice_delta']:+.4f}، "
            f"نجات با حذف Dice شرطی={attribution['rescue_without_conditional_dice']:+.4f} و "
            f"نجات با حذف focal شرطی={attribution['rescue_without_conditional_focal']:+.4f}.\n\n"
            "تحلیل کاربردی: این خروجی فقط علت مکانیکی updateهای کوتاه را مشخص "
            "می‌کند؛ تا قبل از recipe قفل‌شده و calibration مستقل، مجوز ارتقای مدل نیست."
        ),
        run=args.run_name,
        decision=result["decision"],
        mlflow=result.get("mlflow_run_id", "n/a"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
