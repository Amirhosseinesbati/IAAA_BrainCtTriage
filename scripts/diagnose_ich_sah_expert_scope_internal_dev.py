"""Patient-safe internal-dev attribution of SAH expert trainable scope.

Exp87 does not access calibration or outer images.  It compares head-only,
final-decoder-block and full-decoder scopes on the same fold-0/3 training
sequence and evaluates threshold-free separability on internal dev fold 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.diagnose_ich_independent_sah_expert import (
    BACKGROUND_CLASS_ID,
    HISTOGRAM_BINS,
    IPH_CLASS_ID,
    SAH_CLASS_ID,
    IndependentSahExpert,
    StreamingBinaryHistogram,
    _build_expert,
    _seed_everything,
    _write_json,
    masked_sah_expert_loss,
)
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    load_segmentation_manifest,
    subtype_aware_sampler,
)
from src.strategies.ich_v2.operations import file_sha256, git_commit


SCOPES = ("head_only", "final_decoder_block", "full_decoder")
INTERNAL_TRAIN_FOLDS = (0, 3)
INTERNAL_DEV_FOLD = 4
CALIBRATION_FOLD_UNTOUCHED = 1
OUTER_FOLD_UNTOUCHED = 2


def _patient_safe_internal_frames(
    manifest_path: Path,
) -> tuple[Any, Any, dict[str, int]]:
    frame = load_segmentation_manifest(manifest_path)
    train = frame.loc[
        frame["fold"].isin(INTERNAL_TRAIN_FOLDS)
        & (frame["classification_known"] == 1)
    ].copy()
    dev = frame.loc[frame["fold"] == INTERNAL_DEV_FOLD].copy()
    calibration = frame.loc[frame["fold"] == CALIBRATION_FOLD_UNTOUCHED]
    outer = frame.loc[frame["fold"] == OUTER_FOLD_UNTOUCHED]
    parts = (train, dev, calibration, outer)
    patient_sets = [set(part["patient_id"].astype(str)) for part in parts]
    for index, left in enumerate(patient_sets):
        for right in patient_sets[index + 1 :]:
            if left & right:
                raise ValueError("Patient leakage detected in Exp87 fold partition")
    if not len(train) or not len(dev):
        raise ValueError("Exp87 internal train/dev split is empty")
    sort_columns = ["study_id", "slice_index"]
    counts = {
        "internal_train_slices": int(len(train)),
        "internal_dev_slices": int(len(dev)),
        "calibration_slices_not_inferred": int(len(calibration)),
        "outer_slices_not_inferred": int(len(outer)),
    }
    return (
        train.sort_values(sort_columns).reset_index(drop=True),
        dev.sort_values(sort_columns).reset_index(drop=True),
        counts,
    )


def _make_loaders(
    train_frame: Any,
    dev_frame: Any,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ICHAdjacentSegmentationDataset(train_frame, augment=True, context_radius=1),
        batch_size=batch_size,
        sampler=subtype_aware_sampler(train_frame, seed=seed),
        **common,
    )
    dev_loader = DataLoader(
        ICHAdjacentSegmentationDataset(dev_frame, context_radius=1),
        batch_size=max(batch_size, min(batch_size * 2, 16)),
        shuffle=False,
        **common,
    )
    return train_loader, dev_loader


def _select_scope(
    model: IndependentSahExpert, scope: str
) -> list[torch.nn.Parameter]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown Exp87 scope: {scope}")
    model.requires_grad_(False)
    modules: tuple[torch.nn.Module, ...]
    if scope == "head_only":
        modules = (model.expert_head,)
    elif scope == "final_decoder_block":
        blocks = getattr(model.expert_decoder, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleDict) or "x_0_4" not in blocks:
            raise ValueError("Exp87 requires Unet++ final decoder block x_0_4")
        modules = (blocks["x_0_4"], model.expert_head)
    else:
        modules = (model.expert_decoder, model.expert_head)
    for module in modules:
        module.requires_grad_(True)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    if not parameters:
        raise ValueError("Selected Exp87 scope has no trainable parameters")
    # Deliberately keep the complete model in eval mode so normalization running
    # statistics stay frozen. Autograd still updates the selected affine/conv weights.
    model.eval()
    model.incumbent_model.eval()
    return parameters


def _evaluate_raw_separability(
    model: IndependentSahExpert,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, dict[str, Any]]]:
    regions = ("background_or_iph", "near_incumbent_foreground", "iph_only")
    histograms = {
        region: {
            "incumbent_raw": StreamingBinaryHistogram(HISTOGRAM_BINS),
            "expert_raw": StreamingBinaryHistogram(HISTOGRAM_BINS),
        }
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
            masks = {
                "background_or_iph": recoverable,
                "near_incumbent_foreground": recoverable & near_foreground,
                "iph_only": known & (incumbent_mask == IPH_CLASS_ID),
            }
            incumbent_margin = incumbent_logits[:, SAH_CLASS_ID] - torch.logsumexp(
                incumbent_logits[:, [BACKGROUND_CLASS_ID, IPH_CLASS_ID]], dim=1
            )
            scores = {
                "incumbent_raw": torch.sigmoid(incumbent_margin),
                "expert_raw": torch.sigmoid(components["expert_logit"][:, 0].float()),
            }
            for region, mask in masks.items():
                for name, score in scores.items():
                    histograms[region][name].update(score, target, mask)
    return {
        region: {name: histogram.summarize() for name, histogram in values.items()}
        for region, values in histograms.items()
    }


def _scope_gate(metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, bool]:
    primary = metrics["background_or_iph"]
    near = metrics["near_incumbent_foreground"]
    expert = primary["expert_raw"]
    incumbent = primary["incumbent_raw"]
    values = [
        expert["average_precision"],
        incumbent["average_precision"],
        expert["roc_auc"],
        incumbent["roc_auc"],
        near["expert_raw"]["average_precision"],
        near["incumbent_raw"]["average_precision"],
    ]
    finite = all(value is not None and math.isfinite(float(value)) for value in values)
    checks = {"all_required_metrics_finite": finite}
    if finite:
        checks.update(
            {
                "expert_ap_gain_at_least_0_005": float(expert["average_precision"])
                >= float(incumbent["average_precision"]) + 0.005,
                "expert_auc_gain_at_least_0_01": float(expert["roc_auc"])
                >= float(incumbent["roc_auc"]) + 0.01,
                "near_foreground_ap_gain_at_least_0_01": float(
                    near["expert_raw"]["average_precision"]
                )
                >= float(near["incumbent_raw"]["average_precision"]) + 0.01,
            }
        )
    checks["all_passed"] = all(checks.values())
    return checks


def _metric_delta(
    before: dict[str, dict[str, dict[str, Any]]],
    after: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for region in before:
        for score in before[region]:
            for metric in ("average_precision", "roc_auc", "precision_at_recall_0_10"):
                left = before[region][score][metric]
                right = after[region][score][metric]
                if left is not None and right is not None:
                    result[f"{region}.{score}.{metric}"] = float(right) - float(left)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.epochs != 1
        or args.batch_size != 8
        or args.learning_rate != 1e-4
        or args.weight_decay != 1e-4
    ):
        raise ValueError("Exp87 is locked to one epoch, batch=8, lr=1e-4, wd=1e-4")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp87 requires CUDA BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "aggregate_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite Exp87 output: {result_path}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a standard training config")
    source_config = payload["config"]
    seed = int(source_config.get("seed", 42))
    train_frame, dev_frame, split_counts = _patient_safe_internal_frames(
        args.manifest_path
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    scope_results: dict[str, Any] = {}
    started = time.perf_counter()
    for scope in SCOPES:
        _seed_everything(seed)
        train_loader, dev_loader = _make_loaders(
            train_frame,
            dev_frame,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=seed,
        )
        model = _build_expert(args.checkpoint, source_config, device)
        parameters = _select_scope(model, scope)
        before = _evaluate_raw_separability(model, dev_loader, device)
        optimizer = AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        batch_digest = hashlib.sha256()
        losses: list[float] = []
        tversky: list[float] = []
        positives: list[float] = []
        for batch in train_loader:
            for study_id, slice_index in zip(
                batch["study_id"], batch["slice_index"].tolist(), strict=True
            ):
                batch_digest.update(f"{study_id}:{int(slice_index)}\n".encode())
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
            losses.append(statistics["loss"])
            tversky.append(statistics["tversky"])
            positives.append(statistics["positive_pixels"])
        after = _evaluate_raw_separability(model, dev_loader, device)
        scope_results[scope] = {
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "batch_identity_sha256": batch_digest.hexdigest(),
            "mean_train_loss": float(np.mean(losses)),
            "mean_train_tversky": float(np.mean(tversky)),
            "mean_positive_pixels_per_batch": float(np.mean(positives)),
            "before_internal_dev": before,
            "after_internal_dev": after,
            "after_minus_before": _metric_delta(before, after),
            "scope_gate": _scope_gate(after),
        }
        del model, optimizer, train_loader, dev_loader
        torch.cuda.empty_cache()

    batch_hashes = {
        value["batch_identity_sha256"] for value in scope_results.values()
    }
    passing = [
        scope
        for scope, value in scope_results.items()
        if value["scope_gate"]["all_passed"]
    ]
    selected_scope = (
        max(
            passing,
            key=lambda scope: scope_results[scope]["after_internal_dev"]
            ["background_or_iph"]["expert_raw"]["average_precision"],
        )
        if passing
        else None
    )
    decision = (
        "advance_selected_scope_to_fixed_calibration_screen"
        if selected_scope is not None and len(batch_hashes) == 1
        else "close_independent_sah_expert_scope_branch"
    )
    result = {
        "schema_version": 1,
        "analysis_kind": "patient_safe_internal_dev_sah_expert_scope_attribution",
        "experiment": "exp87_sah_expert_scope_internal_dev_v1",
        "run_name": args.run_name,
        "decision": decision,
        "selected_scope": selected_scope,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest_path),
        "git_commit": git_commit(),
        "internal_train_folds": list(INTERNAL_TRAIN_FOLDS),
        "internal_dev_fold": INTERNAL_DEV_FOLD,
        "calibration_fold_not_inferred": CALIBRATION_FOLD_UNTOUCHED,
        "outer_fold_not_inferred": OUTER_FOLD_UNTOUCHED,
        "calibration_evaluation_performed": False,
        "outer_evaluation_performed": False,
        "row_level_predictions_persisted": False,
        "external_reporting_enabled": False,
        "normalization_running_statistics_frozen": True,
        "batch_identities_match_across_scopes": len(batch_hashes) == 1,
        **split_counts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scopes": scope_results,
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "checkpoint_saved": False,
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
