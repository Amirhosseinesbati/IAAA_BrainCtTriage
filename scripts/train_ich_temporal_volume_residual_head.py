"""Train a leakage-gated temporal residual head for official ICH volumes."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    load_segmentation_manifest,
    split_segmentation_slices,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.temporal_volume_head import (
    ICHSequenceVolumeDataset,
    SUBTYPE_LABELS,
    TemporalVolumeResidualHead,
    collate_ich_volume_sequences,
    extract_frozen_encoder_volume_features,
    temporal_volume_loss,
    volume_summary,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


VOLUME_GATE = {
    "minimum_total_mae_improvement_ml": 0.5,
    "minimum_absolute_bias_improvement_ml": 0.5,
    "maximum_fpr_delta": 0.02,
    "minimum_presence_f1_delta": -0.01,
    "minimum_critical_trigger_macro_f1_delta": -0.02,
    "maximum_subtype_mae_increase_ml": 0.5,
}
GATE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TemporalVolumeTrainConfig:
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
    maximum_log_residual: float = 4.0
    study_loss_weight: float = 0.75
    total_loss_weight: float = 0.25
    huber_beta: float = 0.25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    epochs: int = 20
    patience: int = 4
    study_batch_size: int = 8
    extraction_batch_size: int = 16
    workers: int = 4
    maximum_pos_weight: float = 8.0
    seed: int = 42
    max_train_steps: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluation_summary(payload: dict[str, object]) -> dict[str, object]:
    rescored = payload.get("rescored_summary")
    if isinstance(rescored, dict) and "total_volume_mae_ml" in rescored:
        return rescored
    if "total_volume_mae_ml" in payload:
        return payload
    raise ValueError("Baseline JSON does not contain a volume summary")


def _positive_weights(
    targets: np.ndarray,
    *,
    maximum: float,
) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != len(SUBTYPE_LABELS):
        raise ValueError("Volume positive-weight targets have an unexpected shape")
    if maximum < 1 or not np.isfinite(maximum):
        raise ValueError("Maximum positive weight must be finite and at least one")
    positives = (targets > 0).sum(axis=0).astype(np.float64)
    negatives = len(targets) - positives
    if np.any(positives <= 0):
        missing = [
            label for label, count in zip(SUBTYPE_LABELS, positives, strict=True)
            if count <= 0
        ]
        raise ValueError(f"Volume training split lacks positive targets: {missing}")
    return torch.as_tensor(
        np.clip(np.sqrt(negatives / positives), 1.0, maximum),
        dtype=torch.float32,
    )


def _cache_metadata(
    cache_path: Path,
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    split_name: str,
    frame: pd.DataFrame,
    extraction_batch_size: int,
) -> dict[str, object]:
    return {
        "cache_path": str(cache_path),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "split": split_name,
        "slices": int(len(frame)),
        "studies": int(frame["study_id"].nunique()),
        "context_radius": 1,
        "augmentation": False,
        "extraction_batch_size": int(extraction_batch_size),
        "cache_schema": "ich_temporal_volume_v1",
    }


def _ensure_volume_cache(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    cache_path: Path,
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    split_name: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    metadata_path = cache_path.with_suffix(".json")
    expected = _cache_metadata(
        cache_path,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        split_name=split_name,
        frame=frame,
        extraction_batch_size=batch_size,
    )
    if cache_path.exists() and metadata_path.exists():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {key: observed.get(key) for key in expected}
        if comparable == expected and observed.get("cache_sha256") == file_sha256(
            cache_path
        ):
            return observed
        raise ValueError(f"Refusing to reuse stale temporal volume cache: {cache_path}")
    extracted = extract_frozen_encoder_volume_features(
        model,
        frame,
        cache_path,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    metadata = {
        **expected,
        **extracted,
        "cache_sha256": file_sha256(cache_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _predict(
    model: TemporalVolumeResidualHead,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    study_ids: list[str] = []
    truths: list[np.ndarray] = []
    baseline_volumes: list[np.ndarray] = []
    candidate_volumes: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            base_slices = batch["base_slice_volumes"].to(
                device, non_blocking=True
            )
            lengths = batch["lengths"].to(device, non_blocking=True)
            candidate_slices = model(features, base_logits, base_slices, lengths)
            steps = torch.arange(candidate_slices.shape[1], device=device)[None, :]
            padding = steps < lengths[:, None]
            baseline_study = (base_slices * padding[:, :, None]).sum(dim=1)
            candidate_study = (
                candidate_slices * padding[:, :, None]
            ).sum(dim=1)
            study_ids.extend(batch["study_id"])
            truths.append(batch["study_target_volumes"].numpy())
            baseline_volumes.append(baseline_study.cpu().numpy())
            candidate_volumes.append(candidate_study.cpu().numpy())
    return (
        study_ids,
        np.concatenate(truths),
        np.concatenate(baseline_volumes),
        np.concatenate(candidate_volumes),
    )


def volume_deltas(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    base_critical = baseline.get("critical_trigger_macro_f1")
    candidate_critical = candidate.get("critical_trigger_macro_f1")
    critical_delta = (
        None
        if base_critical is None or candidate_critical is None
        else float(candidate_critical) - float(base_critical)
    )
    return {
        "total_volume_mae_ml": float(candidate["total_volume_mae_ml"])
        - float(baseline["total_volume_mae_ml"]),
        "absolute_total_volume_bias_ml": abs(
            float(candidate["total_volume_bias_ml"])
        ) - abs(float(baseline["total_volume_bias_ml"])),
        "normal_false_positive_rate_at_0_1ml": float(
            candidate["normal_false_positive_rate_at_0_1ml"]
        ) - float(baseline["normal_false_positive_rate_at_0_1ml"]),
        "presence_f1_at_0_1ml": float(candidate["presence_f1_at_0_1ml"])
        - float(baseline["presence_f1_at_0_1ml"]),
        "presence_sensitivity_at_0_1ml": float(
            candidate["presence_sensitivity_at_0_1ml"]
        ) - float(baseline["presence_sensitivity_at_0_1ml"]),
        "critical_trigger_macro_f1": critical_delta,
        "subtype_mae_ml": {
            label: float(candidate["subtypes"][label]["mae_ml"])
            - float(baseline["subtypes"][label]["mae_ml"])
            for label in SUBTYPE_LABELS
        },
    }


def temporal_volume_promotion_decision(
    delta: dict[str, object],
) -> dict[str, object]:
    critical = delta["critical_trigger_macro_f1"]
    checks = {
        "total_volume_mae": float(delta["total_volume_mae_ml"])
        <= -VOLUME_GATE["minimum_total_mae_improvement_ml"] + GATE_TOLERANCE,
        "absolute_total_volume_bias": float(
            delta["absolute_total_volume_bias_ml"]
        ) <= -VOLUME_GATE["minimum_absolute_bias_improvement_ml"] + GATE_TOLERANCE,
        "normal_fpr_safety": float(
            delta["normal_false_positive_rate_at_0_1ml"]
        ) <= VOLUME_GATE["maximum_fpr_delta"] + GATE_TOLERANCE,
        "presence_f1_safety": float(delta["presence_f1_at_0_1ml"])
        >= VOLUME_GATE["minimum_presence_f1_delta"] - GATE_TOLERANCE,
        "critical_trigger_safety": critical is None
        or float(critical)
        >= VOLUME_GATE["minimum_critical_trigger_macro_f1_delta"]
        - GATE_TOLERANCE,
        "subtype_mae_safety": max(
            float(value) for value in delta["subtype_mae_ml"].values()
        ) <= VOLUME_GATE["maximum_subtype_mae_increase_ml"] + GATE_TOLERANCE,
    }
    return {
        "criteria": VOLUME_GATE,
        "checks": checks,
        "promotion_allowed": bool(all(checks.values())),
    }


def _checkpoint_score(
    summary: dict[str, object], baseline: dict[str, object]
) -> float | None:
    fpr_delta = float(summary["normal_false_positive_rate_at_0_1ml"]) - float(
        baseline["normal_false_positive_rate_at_0_1ml"]
    )
    f1_delta = float(summary["presence_f1_at_0_1ml"]) - float(
        baseline["presence_f1_at_0_1ml"]
    )
    if (
        fpr_delta > VOLUME_GATE["maximum_fpr_delta"] + GATE_TOLERANCE
        or f1_delta
        < VOLUME_GATE["minimum_presence_f1_delta"] - GATE_TOLERANCE
    ):
        return None
    return -float(summary["total_volume_mae_ml"])


def _validate_locked_baseline(
    observed: dict[str, object], locked: dict[str, object]
) -> None:
    if int(observed["studies"]) != int(locked["studies"]):
        raise RuntimeError(
            "Extracted volume baseline does not match locked study count: "
            f"{observed['studies']} vs {locked['studies']}"
        )
    exact_keys = (
        "normal_false_positive_rate_at_0_1ml",
        "presence_f1_at_0_1ml",
    )
    for key in exact_keys:
        if abs(float(observed[key]) - float(locked[key])) > 1e-12:
            raise RuntimeError(
                f"Extracted volume baseline does not match locked {key}: "
                f"{observed[key]} vs {locked[key]}"
            )
    numeric_keys = ("total_volume_mae_ml", "total_volume_bias_ml")
    for key in numeric_keys:
        if abs(float(observed[key]) - float(locked[key])) > 1e-5:
            raise RuntimeError(
                f"Extracted volume baseline does not match locked {key}: "
                f"{observed[key]} vs {locked[key]}"
            )
    for label in SUBTYPE_LABELS:
        observed_mae = float(observed["subtypes"][label]["mae_ml"])
        locked_mae = float(locked["subtypes"][label]["mae_ml"])
        if abs(observed_mae - locked_mae) > 1e-5:
            raise RuntimeError(
                "Extracted volume baseline does not match locked subtype MAE: "
                f"{label} {observed_mae} vs {locked_mae}"
            )


def run(config: TemporalVolumeTrainConfig) -> dict[str, object]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("Outer and calibration folds must differ")
    if min(config.epochs, config.patience, config.study_batch_size) < 1:
        raise ValueError("Temporal volume epochs, patience and batch size must be positive")
    if config.max_train_steps is not None and config.max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if min(config.learning_rate, config.huber_beta, config.maximum_log_residual) <= 0:
        raise ValueError("Temporal volume learning rate and scales must be positive")
    if min(config.weight_decay, config.study_loss_weight, config.total_loss_weight) < 0:
        raise ValueError("Temporal volume regularization and loss weights cannot be negative")
    if config.maximum_pos_weight < 1:
        raise ValueError("maximum_pos_weight must be at least one")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Temporal volume training requires CUDA BF16 support")

    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_temporal_volume_head.pth"
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    base_payload = torch.load(
        Path(config.base_checkpoint), map_location="cpu", weights_only=True
    )
    if not isinstance(base_payload, dict):
        raise TypeError("Base segmentation checkpoint must be a dictionary")
    source_config = base_payload.get("config", {})
    if not isinstance(source_config, dict):
        raise TypeError("Base segmentation checkpoint config must be a dictionary")
    if (
        int(source_config.get("outer_fold", -1)) != config.outer_fold
        or int(source_config.get("calibration_fold", -1)) != config.calibration_fold
    ):
        raise ValueError("Base checkpoint does not match temporal volume held-out folds")
    if tuple(base_payload.get("output_labels", ())) != tuple(OUTPUT_LABELS):
        raise ValueError("Base checkpoint labels do not match temporal volume labels")
    model = build_segmentation_model(
        architecture=str(source_config.get("architecture", "unetplusplus")),
        encoder_name=str(source_config.get("encoder_name", "efficientnet-b2")),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
    ).to(device)
    load_segmentation_weights(model, config.base_checkpoint)
    model.requires_grad_(False).eval()

    manifest = load_segmentation_manifest(config.manifest_path)
    training, calibration, _ = split_segmentation_slices(
        manifest,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
    )
    truth, truth_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    allowed_studies = set(training["study_id"].astype(str)) | set(
        calibration["study_id"].astype(str)
    )
    truth = truth.loc[truth["study_id"].astype(str).isin(allowed_studies)].copy()
    checkpoint_sha = file_sha256(config.base_checkpoint)
    manifest_sha = file_sha256(config.manifest_path)
    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = f"f{config.outer_fold}c{config.calibration_fold}_{checkpoint_sha[:12]}"
    train_cache = cache_root / f"train_volume_{identity}.npz"
    calibration_cache = cache_root / f"calibration_volume_{identity}.npz"
    train_cache_meta = _ensure_volume_cache(
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
    calibration_cache_meta = _ensure_volume_cache(
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

    train_dataset = ICHSequenceVolumeDataset(train_cache, truth)
    calibration_dataset = ICHSequenceVolumeDataset(calibration_cache, truth)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.study_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_ich_volume_sequences,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=config.study_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ich_volume_sequences,
    )
    feature_dim = int(train_dataset.embeddings.shape[1])
    temporal = TemporalVolumeResidualHead(
        feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        maximum_log_residual=config.maximum_log_residual,
    ).to(device)
    trainable_parameters = list(temporal.parameters())
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    active_slices = train_dataset.spatial_known > 0.5
    slice_pos_weight = _positive_weights(
        train_dataset.target_slice_volumes[active_slices].numpy(),
        maximum=config.maximum_pos_weight,
    ).to(device)
    study_targets_train = torch.stack(train_dataset.study_target_volumes).numpy()
    study_pos_weight = _positive_weights(
        study_targets_train,
        maximum=config.maximum_pos_weight,
    ).to(device)

    _, calibration_truth, baseline_volumes, epoch0_volumes = _predict(
        temporal, calibration_loader, device=device
    )
    if not np.array_equal(baseline_volumes, epoch0_volumes):
        raise RuntimeError("Zero-initialized temporal volume head is not exact identity")
    baseline_metrics = volume_summary(calibration_truth, baseline_volumes)
    baseline_payload = json.loads(
        Path(config.baseline_summary).read_text(encoding="utf-8")
    )
    baseline_reference = _evaluation_summary(baseline_payload)
    _validate_locked_baseline(baseline_metrics, baseline_reference)
    best_score = _checkpoint_score(baseline_metrics, baseline_metrics)
    if best_score is None:
        raise RuntimeError("Epoch-zero baseline unexpectedly failed its own safety gate")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    history.append({
        "epoch": 0,
        "checkpoint_score": best_score,
        "total_volume_mae_ml": baseline_metrics["total_volume_mae_ml"],
        "total_volume_bias_ml": baseline_metrics["total_volume_bias_ml"],
        "normal_fpr": baseline_metrics["normal_false_positive_rate_at_0_1ml"],
        "presence_f1": baseline_metrics["presence_f1_at_0_1ml"],
        "critical_trigger_macro_f1": baseline_metrics[
            "critical_trigger_macro_f1"
        ],
    })
    torch.save({
        "state_dict": temporal.state_dict(),
        "epoch": 0,
        "config": asdict(config),
        "feature_dim": feature_dim,
        "output_labels": SUBTYPE_LABELS,
        "base_checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
    }, checkpoint_path)

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    run_kind = "smoke" if config.max_train_steps else "calibration_screen"
    if run_kind != "smoke":
        notify_campaign(
            "start",
            "غربال temporal area/volume head مدل ICH آغاز شد. حجم epoch صفر دقیقاً "
            "برابر incumbent است و فقط head کم‌ظرفیت روی featureهای frozen آموزش "
            "می‌بیند. تحلیل کوتاه: این بار هدف مستقیم خروجی رسمی پنج حجم، MAE و bias "
            "است؛ mask فضایی دست‌نخورده و outer ممنوع است. اقدام بعدی: انتخاب epoch "
            "فقط تحت قید FPR/F1 و رد کامل در صورت نرسیدن به بهبود حجمی معنادار.",
            run=config.run_name,
            kind=run_kind,
            train_studies=len(train_dataset),
            val_studies=len(calibration_dataset),
            detail=f"feature_dim={feature_dim}; zero-init exact identity",
        )
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as mlflow_run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "frozen_encoder_temporal_volume_residual_head",
                "run_kind": run_kind,
                "outer_evaluation": "forbidden",
                "git_commit": git_commit(),
            })
            mlflow.log_params({
                **asdict(config),
                "feature_dim": feature_dim,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "training_studies": len(train_dataset),
                "calibration_studies": len(calibration_dataset),
                "base_checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "train_cache_sha256": train_cache_meta["cache_sha256"],
                "calibration_cache_sha256": calibration_cache_meta["cache_sha256"],
            })
            mlflow.log_metrics({
                "calibration_total_volume_mae_ml": float(
                    baseline_metrics["total_volume_mae_ml"]
                ),
                "calibration_total_volume_bias_ml": float(
                    baseline_metrics["total_volume_bias_ml"]
                ),
                "calibration_normal_fpr": float(
                    baseline_metrics["normal_false_positive_rate_at_0_1ml"]
                ),
                "calibration_presence_f1": float(
                    baseline_metrics["presence_f1_at_0_1ml"]
                ),
            }, step=0)

            for epoch in range(1, config.epochs + 1):
                temporal.train()
                component_history = {
                    "loss": [], "slice": [], "study": [], "total": []
                }
                for step, batch in enumerate(train_loader, start=1):
                    features = batch["features"].to(device, non_blocking=True)
                    base_logits = batch["base_logits"].to(device, non_blocking=True)
                    base_slices = batch["base_slice_volumes"].to(
                        device, non_blocking=True
                    )
                    lengths = batch["lengths"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    candidate_slices = temporal(
                        features, base_logits, base_slices, lengths
                    )
                    components = temporal_volume_loss(
                        candidate_slices,
                        batch["target_slice_volumes"].to(
                            device, non_blocking=True
                        ),
                        batch["spatial_known"].to(device, non_blocking=True),
                        batch["study_target_volumes"].to(
                            device, non_blocking=True
                        ),
                        lengths,
                        slice_pos_weight=slice_pos_weight,
                        study_pos_weight=study_pos_weight,
                        study_loss_weight=config.study_loss_weight,
                        total_loss_weight=config.total_loss_weight,
                        huber_beta=config.huber_beta,
                    )
                    components["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
                    optimizer.step()
                    for name, value in components.items():
                        component_history[name].append(float(value.detach().cpu()))
                    if config.max_train_steps and step >= config.max_train_steps:
                        break

                _, observed_truth, _, candidate_volumes = _predict(
                    temporal, calibration_loader, device=device
                )
                if not np.array_equal(observed_truth, calibration_truth):
                    raise RuntimeError("Calibration study order changed during training")
                candidate_metrics = volume_summary(
                    calibration_truth, candidate_volumes
                )
                delta = volume_deltas(baseline_metrics, candidate_metrics)
                checkpoint_score = _checkpoint_score(
                    candidate_metrics, baseline_metrics
                )
                epoch_payload = {
                    "epoch": epoch,
                    "checkpoint_score": checkpoint_score,
                    "checkpoint_eligible": checkpoint_score is not None,
                    "total_volume_mae_ml": candidate_metrics[
                        "total_volume_mae_ml"
                    ],
                    "total_volume_bias_ml": candidate_metrics[
                        "total_volume_bias_ml"
                    ],
                    "normal_fpr": candidate_metrics[
                        "normal_false_positive_rate_at_0_1ml"
                    ],
                    "presence_f1": candidate_metrics["presence_f1_at_0_1ml"],
                    "critical_trigger_macro_f1": candidate_metrics[
                        "critical_trigger_macro_f1"
                    ],
                    "delta_total_volume_mae_ml": delta["total_volume_mae_ml"],
                    "delta_absolute_total_volume_bias_ml": delta[
                        "absolute_total_volume_bias_ml"
                    ],
                    **{
                        f"train_{name}": float(np.mean(values))
                        for name, values in component_history.items()
                    },
                }
                history.append(epoch_payload)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics({
                    key: float(value)
                    for key, value in epoch_payload.items()
                    if key != "epoch" and value is not None
                }, step=epoch)
                print(json.dumps(epoch_payload, sort_keys=True), flush=True)
                if (
                    checkpoint_score is not None
                    and checkpoint_score > best_score + 1e-6
                ):
                    best_score = checkpoint_score
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({
                        "state_dict": temporal.state_dict(),
                        "epoch": epoch,
                        "config": asdict(config),
                        "feature_dim": feature_dim,
                        "output_labels": SUBTYPE_LABELS,
                        "base_checkpoint_sha256": checkpoint_sha,
                        "manifest_sha256": manifest_sha,
                        "calibration_metrics": candidate_metrics,
                        "calibration_delta": delta,
                    }, checkpoint_path)
                else:
                    stale_epochs += 1
                if config.max_train_steps or stale_epochs >= config.patience:
                    break

            best_payload = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            temporal.load_state_dict(best_payload["state_dict"], strict=True)
            _, _, _, best_volumes = _predict(
                temporal, calibration_loader, device=device
            )
            candidate_metrics = volume_summary(calibration_truth, best_volumes)
            delta = volume_deltas(baseline_metrics, candidate_metrics)
            decision = temporal_volume_promotion_decision(delta)
            duration = time.perf_counter() - started
            summary: dict[str, object] = {
                "analysis_kind": "ich_frozen_encoder_temporal_volume_residual_head",
                "run_name": config.run_name,
                "run_id": mlflow_run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
                "best_checkpoint_score": best_score,
                "duration_s": duration,
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "training_studies": len(train_dataset),
                "calibration_studies": len(calibration_dataset),
                "outer_evaluation_performed": False,
                "base_checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": delta,
                "promotion_decision": decision,
                "spatial_masks_unchanged_by_design": True,
                "official_volume_outputs_changed": best_epoch > 0,
                "zero_initialization_exact_identity": True,
                "feature_cache": {
                    "training": train_cache_meta,
                    "calibration": calibration_cache_meta,
                    "persisted_in_git": False,
                },
                "truth_source": str(truth_source),
                "git_commit": git_commit(),
            }
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            mlflow.log_metrics({
                "best_epoch": float(best_epoch),
                "delta_total_volume_mae_ml": float(delta["total_volume_mae_ml"]),
                "delta_absolute_total_volume_bias_ml": float(
                    delta["absolute_total_volume_bias_ml"]
                ),
                "delta_normal_fpr": float(
                    delta["normal_false_positive_rate_at_0_1ml"]
                ),
                "delta_presence_f1": float(delta["presence_f1_at_0_1ml"]),
                "promotion_allowed": float(decision["promotion_allowed"]),
                "duration_s": duration,
            })
            mlflow.log_artifacts(
                str(output), artifact_path="ich_temporal_volume_residual_head"
            )

        if run_kind != "smoke":
            event = "success" if decision["promotion_allowed"] else "warning"
            notify_campaign(
                event,
                f"غربال temporal area/volume head مدل ICH تمام شد. delta MAE="
                f"{float(delta['total_volume_mae_ml']):+.3f}mL، delta |bias|="
                f"{float(delta['absolute_total_volume_bias_ml']):+.3f}mL، delta FPR="
                f"{float(delta['normal_false_positive_rate_at_0_1ml']):+.3f} و "
                f"delta F1={float(delta['presence_f1_at_0_1ml']):+.3f} است؛ "
                f"promotion={decision['promotion_allowed']}. تحلیل کوتاه: این اعداد "
                "مستقیماً کیفیت پنج حجم رسمی را می‌سنجند، mask ثابت و outer خوانده "
                "نشده است. اقدام بعدی: در صورت عبور، replication قفل‌شده روی outer0؛ "
                "در غیر این صورت بستن head بدون sweep پس‌نگر.",
                run=config.run_name,
                kind=run_kind,
                best_epoch=best_epoch,
                detail=f"MLflow {summary['run_id']}; outer=false",
            )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "اجرای temporal area/volume head مدل ICH متوقف شد. تحلیل کوتاه: این خطا "
            "نتیجهٔ کیفیتی نیست و outer باز نشده است. اقدام بعدی: اصلاح کوچک‌ترین علت "
            "فنی و تکرار smoke.",
            run=config.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise


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
    parser.add_argument("--maximum-log-residual", type=float, default=4.0)
    parser.add_argument("--study-loss-weight", type=float, default=0.75)
    parser.add_argument("--total-loss-weight", type=float, default=0.25)
    parser.add_argument("--huber-beta", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--study-batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-pos-weight", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int)
    config = TemporalVolumeTrainConfig(**vars(parser.parse_args()))
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
