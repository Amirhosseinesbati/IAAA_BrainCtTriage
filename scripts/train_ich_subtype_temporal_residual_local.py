"""Local-only calibration screen of an Any-invariant temporal subtype head."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.train_ich_temporal_residual_head import (
    _ensure_feature_cache,
    _evaluation_summary,
    _predict,
    _study_pos_weight,
)
from src.strategies.ich_2p5d.segmentation_data import (
    load_segmentation_manifest,
    segmentation_classification_weights,
    split_segmentation_slices,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.temporal_head import (
    ICHSequenceFeatureDataset,
    SubtypeTemporalResidualHead,
    auc_summary,
    collate_ich_sequences,
    temporal_classification_loss,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import file_sha256, git_commit


GATE = {
    "minimum_selection_proxy_delta": 0.0015,
    "minimum_macro_subtype_auc_delta": 0.01,
    "minimum_subtype_auc_delta": -0.01,
    "minimum_improved_subtypes": 3,
    "any_ich_absolute_tolerance": 1e-12,
}


@dataclass(frozen=True)
class SubtypeTemporalConfig:
    run_name: str
    output_dir: str
    cache_dir: str
    manifest_path: str
    base_checkpoint: str
    baseline_summary: str
    outer_fold: int = 2
    calibration_fold: int = 1
    projection_dim: int = 64
    hidden_dim: int = 32
    dropout: float = 0.2
    study_loss_weight: float = 0.5
    focal_gamma: float = 1.0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    epochs: int = 20
    patience: int = 4
    study_batch_size: int = 8
    extraction_batch_size: int = 16
    workers: int = 4
    maximum_pos_weight: float = 20.0
    seed: int = 42


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subtype_temporal_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    any_delta = float(candidate["any_ich_auc"]) - float(baseline["any_ich_auc"])
    subtype = {
        label: float(candidate["subtype_auc"][label])
        - float(baseline["subtype_auc"][label])
        for label in baseline["subtype_auc"]
    }
    macro_delta = float(candidate["macro_subtype_auc"]) - float(
        baseline["macro_subtype_auc"]
    )
    return {
        "any_ich_auc": any_delta,
        "macro_subtype_auc": macro_delta,
        "selection_proxy": 0.30 * any_delta + 0.15 * macro_delta,
        "subtype_auc": subtype,
    }


def subtype_temporal_promotion_decision(
    delta: dict[str, Any]
) -> dict[str, Any]:
    subtype_values = [float(value) for value in delta["subtype_auc"].values()]
    checks = {
        "any_ich_auc_exact": abs(float(delta["any_ich_auc"]))
        <= GATE["any_ich_absolute_tolerance"],
        "selection_proxy": float(delta["selection_proxy"])
        >= GATE["minimum_selection_proxy_delta"],
        "macro_subtype_auc": float(delta["macro_subtype_auc"])
        >= GATE["minimum_macro_subtype_auc_delta"],
        "subtype_safety": min(subtype_values) >= GATE["minimum_subtype_auc_delta"],
        "at_least_three_subtypes_improve": sum(value > 0 for value in subtype_values)
        >= GATE["minimum_improved_subtypes"],
    }
    return {
        "criteria": GATE,
        "checks": checks,
        "promotion_allowed": all(checks.values()),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def run(config: SubtypeTemporalConfig) -> dict[str, Any]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("Outer and calibration folds must differ")
    if (
        config.projection_dim != 64
        or config.hidden_dim != 32
        or config.dropout != 0.2
        or config.study_loss_weight != 0.5
        or config.focal_gamma != 1.0
        or config.learning_rate != 5e-4
        or config.weight_decay != 1e-3
        or config.epochs != 20
        or config.patience != 4
        or config.study_batch_size != 8
        or config.extraction_batch_size != 16
        or config.maximum_pos_weight != 20.0
    ):
        raise ValueError("Exp88 hyperparameters are locked to the Exp53 recipe")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp88 requires CUDA BF16")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_subtype_temporal_head.pth"
    summary_path = output / "run_summary.json"
    if checkpoint_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite Exp88 output: {output}")
    _write_json(output / "resolved_config.json", asdict(config))

    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    base_payload = torch.load(
        config.base_checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(base_payload, dict) or not isinstance(
        base_payload.get("config"), dict
    ):
        raise ValueError("Base checkpoint must contain a standard config")
    source_config = base_payload["config"]
    if (
        int(source_config.get("outer_fold", -1)) != config.outer_fold
        or int(source_config.get("calibration_fold", -1))
        != config.calibration_fold
    ):
        raise ValueError("Base checkpoint does not match Exp88 held-out folds")
    model = build_segmentation_model(
        architecture=str(source_config["architecture"]),
        encoder_name=str(source_config["encoder_name"]),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
    ).to(device)
    load_segmentation_weights(model, config.base_checkpoint)
    model.requires_grad_(False).eval()

    manifest = load_segmentation_manifest(config.manifest_path)
    training, calibration, outer = split_segmentation_slices(
        manifest,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
    )
    truth, truth_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    checkpoint_sha = file_sha256(config.base_checkpoint)
    manifest_sha = file_sha256(config.manifest_path)
    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = f"f{config.outer_fold}c{config.calibration_fold}_{checkpoint_sha[:12]}"
    train_cache = cache_root / f"train_{identity}.npz"
    calibration_cache = cache_root / f"calibration_{identity}.npz"
    train_cache_meta = _ensure_feature_cache(
        model,
        training,
        train_cache,
        checkpoint_sha256=checkpoint_sha,
        manifest_sha256=manifest_sha,
        split_name="training",
        device=device,
        batch_size=config.extraction_batch_size,
        workers=config.workers,
    )
    calibration_cache_meta = _ensure_feature_cache(
        model,
        calibration,
        calibration_cache,
        checkpoint_sha256=checkpoint_sha,
        manifest_sha256=manifest_sha,
        split_name="calibration",
        device=device,
        batch_size=config.extraction_batch_size,
        workers=config.workers,
    )
    del model
    torch.cuda.empty_cache()

    train_dataset = ICHSequenceFeatureDataset(train_cache, truth)
    calibration_dataset = ICHSequenceFeatureDataset(calibration_cache, truth)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.study_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_ich_sequences,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=config.study_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ich_sequences,
    )
    feature_dim = int(train_dataset.embeddings.shape[1])
    temporal = SubtypeTemporalResidualHead(
        feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    parameters = list(temporal.parameters())
    optimizer = AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    slice_pos_weight = segmentation_classification_weights(
        training, maximum=config.maximum_pos_weight
    ).to(device)[1:]
    study_pos_weight = _study_pos_weight(
        truth, train_dataset.study_ids, config.maximum_pos_weight
    ).to(device)[1:]

    _, calibration_truth, baseline_scores, epoch0_scores = _predict(
        temporal, calibration_loader, device=device
    )
    if not np.array_equal(baseline_scores, epoch0_scores):
        raise RuntimeError("Zero-initialized subtype temporal head is not identity")
    baseline_metrics = auc_summary(calibration_truth, baseline_scores)
    baseline_payload = json.loads(Path(config.baseline_summary).read_text("utf-8"))
    baseline_reference = _evaluation_summary(baseline_payload)
    if (
        abs(
            float(baseline_metrics["any_ich_auc"])
            - float(baseline_reference["any_ich_study_auc"])
        )
        > 1e-12
        or abs(
            float(baseline_metrics["macro_subtype_auc"])
            - float(baseline_reference["macro_subtype_study_auc"])
        )
        > 1e-12
    ):
        raise RuntimeError("Exp88 cache baseline does not match locked v4 metrics")
    baseline_selection = float(baseline_reference["selection_score"])
    best_proxy = baseline_selection
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    torch.save(
        {
            "state_dict": temporal.state_dict(),
            "epoch": 0,
            "config": asdict(config),
            "feature_dim": feature_dim,
            "output_labels": list(baseline_metrics["subtype_auc"].keys()),
            "base_checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "any_ich_logit_invariant": True,
        },
        checkpoint_path,
    )
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        temporal.train()
        component_history = {"loss": [], "slice": [], "study": []}
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = temporal(features, base_logits, lengths)
            components = temporal_classification_loss(
                logits[:, :, 1:],
                batch["slice_targets"].to(device, non_blocking=True)[:, :, 1:],
                batch["classification_known"].to(device, non_blocking=True),
                batch["study_target"].to(device, non_blocking=True)[:, 1:],
                lengths,
                slice_pos_weight=slice_pos_weight,
                study_pos_weight=study_pos_weight,
                study_loss_weight=config.study_loss_weight,
                focal_gamma=config.focal_gamma,
            )
            components["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            for name, value in components.items():
                component_history[name].append(float(value.detach().cpu()))

        _, observed_truth, _, candidate_scores = _predict(
            temporal, calibration_loader, device=device
        )
        if not np.array_equal(observed_truth, calibration_truth):
            raise RuntimeError("Exp88 calibration study order changed")
        candidate_metrics = auc_summary(calibration_truth, candidate_scores)
        delta = subtype_temporal_delta(baseline_metrics, candidate_metrics)
        if abs(float(delta["any_ich_auc"])) > GATE["any_ich_absolute_tolerance"]:
            raise RuntimeError("Exp88 changed Any-ICH despite architectural lock")
        selection_proxy = baseline_selection + float(delta["selection_proxy"])
        row = {
            "epoch": epoch,
            "selection_proxy": selection_proxy,
            "any_ich_auc": candidate_metrics["any_ich_auc"],
            "macro_subtype_auc": candidate_metrics["macro_subtype_auc"],
            "delta_selection_proxy": delta["selection_proxy"],
            "delta_any_ich_auc": delta["any_ich_auc"],
            "delta_macro_subtype_auc": delta["macro_subtype_auc"],
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in component_history.items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps(row, sort_keys=True), flush=True)
        if selection_proxy > best_proxy + 1e-6:
            best_proxy = selection_proxy
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": temporal.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "feature_dim": feature_dim,
                    "output_labels": list(candidate_metrics["subtype_auc"].keys()),
                    "base_checkpoint_sha256": checkpoint_sha,
                    "manifest_sha256": manifest_sha,
                    "calibration_metrics": candidate_metrics,
                    "calibration_delta": delta,
                    "any_ich_logit_invariant": True,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    best_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    temporal.load_state_dict(best_payload["state_dict"], strict=True)
    _, _, _, best_scores = _predict(temporal, calibration_loader, device=device)
    candidate_metrics = auc_summary(calibration_truth, best_scores)
    delta = subtype_temporal_delta(baseline_metrics, candidate_metrics)
    decision = subtype_temporal_promotion_decision(delta)
    result = {
        "schema_version": 1,
        "analysis_kind": "any_invariant_frozen_encoder_temporal_subtype_head",
        "experiment": "exp88_subtype_temporal_residual_calibration_v1",
        "run_name": config.run_name,
        "decision": (
            "advance_to_patient_disjoint_five_fold_development_oof"
            if decision["promotion_allowed"]
            else "reject_subtype_temporal_head_before_oof"
        ),
        "best_epoch": best_epoch,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "promotion_decision": decision,
        "baseline_selection_score_unchanged_spatial": baseline_selection,
        "candidate_selection_proxy_unchanged_spatial": baseline_selection
        + float(delta["selection_proxy"]),
        "any_ich_logits_exact_by_architecture": True,
        "spatial_volume_metrics_unchanged_by_design": True,
        "outer_evaluation_performed": False,
        "outer_slices_not_inferred": int(len(outer)),
        "row_level_predictions_persisted": False,
        "external_reporting_enabled": False,
        "training_studies": len(train_dataset),
        "calibration_studies": len(calibration_dataset),
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "base_checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "temporal_checkpoint_sha256": file_sha256(checkpoint_path),
        "feature_cache": {
            "training": train_cache_meta,
            "calibration": calibration_cache_meta,
        },
        "truth_source": str(truth_source),
        "git_commit": git_commit(),
    }
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--study-loss-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--study-batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    config = SubtypeTemporalConfig(**vars(parser.parse_args()))
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
