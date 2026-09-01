"""Five-fold cross-fitted temporal volume residual head for ICH development OOF."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from scripts.train_ich_temporal_volume_residual_head import (
    VOLUME_GATE,
    _checkpoint_score,
    _ensure_volume_cache,
    _evaluation_summary,
    _positive_weights,
    _predict,
    _validate_locked_baseline,
    temporal_volume_promotion_decision,
    volume_deltas,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import load_segmentation_manifest
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.temporal_volume_head import (
    ICHSequenceVolumeDataset,
    SUBTYPE_LABELS,
    TemporalVolumeResidualHead,
    collate_ich_volume_sequences,
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


EXPECTED_FOLDS = tuple(range(5))
MANIFEST_FOLD_COLUMN = "fold"
CROSSFIT_GATE = {
    **VOLUME_GATE,
    "minimum_mae_bootstrap_probability": 0.95,
    "maximum_mae_delta_ci95_upper_ml": 0.0,
}


@dataclass(frozen=True)
class CrossfitTemporalVolumeConfig:
    run_name: str
    output_dir: str
    cache_dir: str
    manifest_path: str
    locked_oof_summary: str
    fold_checkpoints: tuple[str, ...]
    heldout_folds: tuple[int, ...] = EXPECTED_FOLDS
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
    bootstrap_samples: int = 5000
    seed: int = 42
    max_train_steps: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fold_checkpoints(values: tuple[str, ...]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Fold checkpoints must use FOLD=PATH syntax")
        fold_text, path_text = value.split("=", 1)
        fold = int(fold_text)
        if fold in mapping:
            raise ValueError(f"Duplicate fold checkpoint: {fold}")
        mapping[fold] = Path(path_text)
    if set(mapping) != set(EXPECTED_FOLDS):
        raise ValueError(
            f"Crossfit needs checkpoints for folds 0..4, got {sorted(mapping)}"
        )
    return mapping


def inner_validation_fold(heldout_fold: int) -> int:
    """Use a fixed inner fold outside the historically dominant cal1/cal2 folds."""
    if heldout_fold not in EXPECTED_FOLDS:
        raise ValueError(f"Unexpected heldout fold: {heldout_fold}")
    return next(fold for fold in (3, 4, 0) if fold != heldout_fold)


def _load_oof_model(
    checkpoint: Path,
    *,
    expected_fold: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Fold {expected_fold} checkpoint must be a dictionary")
    source_config = payload.get("config", {})
    if not isinstance(source_config, dict):
        raise TypeError(f"Fold {expected_fold} checkpoint config must be a dictionary")
    if int(source_config.get("outer_fold", -1)) != expected_fold:
        raise ValueError(
            f"Checkpoint assigned to fold {expected_fold} declares outer fold "
            f"{source_config.get('outer_fold')}"
        )
    if tuple(payload.get("output_labels", ())) != tuple(OUTPUT_LABELS):
        raise ValueError(f"Fold {expected_fold} checkpoint labels do not match")
    if int(source_config.get("context_radius", 1)) != 1:
        raise ValueError("Crossfit volume extraction requires three-slice context")
    model = build_segmentation_model(
        architecture=str(source_config.get("architecture", "unetplusplus")),
        encoder_name=str(source_config.get("encoder_name", "efficientnet-b2")),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
        horizontal_symmetry_adapter=bool(
            source_config.get("horizontal_symmetry_adapter", False)
        ),
        five_slice_context_adapter=bool(
            source_config.get("five_slice_context_adapter", False)
        ),
    ).to(device)
    load_segmentation_weights(model, checkpoint)
    model.requires_grad_(False).eval()
    return model, {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "outer_fold": expected_fold,
        "checkpoint_calibration_fold": int(
            source_config.get("calibration_fold", -1)
        ),
        "architecture": str(source_config.get("architecture", "unetplusplus")),
        "encoder_name": str(source_config.get("encoder_name", "efficientnet-b2")),
    }


def _fold_dataloader(
    dataset: ICHSequenceVolumeDataset,
    *,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ich_volume_sequences,
    )


def _training_weights(
    datasets: list[ICHSequenceVolumeDataset],
    *,
    maximum: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_slices = torch.cat(
        [dataset.target_slice_volumes for dataset in datasets], dim=0
    )
    spatial_known = torch.cat(
        [dataset.spatial_known for dataset in datasets], dim=0
    )
    slice_weights = _positive_weights(
        target_slices[spatial_known > 0.5].numpy(), maximum=maximum
    ).to(device)
    study_targets = torch.cat(
        [torch.stack(dataset.study_target_volumes) for dataset in datasets],
        dim=0,
    ).numpy()
    study_weights = _positive_weights(study_targets, maximum=maximum).to(device)
    return slice_weights, study_weights


def _train_meta_fold(
    config: CrossfitTemporalVolumeConfig,
    *,
    heldout_fold: int,
    datasets: dict[int, ICHSequenceVolumeDataset],
    checkpoint_provenance: dict[int, dict[str, object]],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    inner_fold = inner_validation_fold(heldout_fold)
    training_folds = tuple(
        fold
        for fold in EXPECTED_FOLDS
        if fold not in {heldout_fold, inner_fold}
    )
    training_datasets = [datasets[fold] for fold in training_folds]
    training_dataset = ConcatDataset(training_datasets)
    generator = torch.Generator().manual_seed(config.seed + 1009 * heldout_fold)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.study_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_ich_volume_sequences,
    )
    inner_dataset = datasets[inner_fold]
    heldout_dataset = datasets[heldout_fold]
    inner_loader = _fold_dataloader(
        inner_dataset, batch_size=config.study_batch_size
    )
    heldout_loader = _fold_dataloader(
        heldout_dataset, batch_size=config.study_batch_size
    )
    feature_dims = {
        int(dataset.embeddings.shape[1]) for dataset in training_datasets
    } | {
        int(inner_dataset.embeddings.shape[1]),
        int(heldout_dataset.embeddings.shape[1]),
    }
    if len(feature_dims) != 1:
        raise ValueError(f"Fold {heldout_fold} feature dimensions differ")
    feature_dim = feature_dims.pop()

    _seed_everything(config.seed + heldout_fold)
    model = TemporalVolumeResidualHead(
        feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        maximum_log_residual=config.maximum_log_residual,
    ).to(device)
    parameters = list(model.parameters())
    optimizer = AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    slice_pos_weight, study_pos_weight = _training_weights(
        training_datasets,
        maximum=config.maximum_pos_weight,
        device=device,
    )

    inner_studies, inner_truth, inner_base, inner_epoch0 = _predict(
        model, inner_loader, device=device
    )
    if inner_studies != inner_dataset.study_ids:
        raise RuntimeError("Inner-validation study order changed")
    if not np.array_equal(inner_base, inner_epoch0):
        raise RuntimeError("Crossfit temporal head is not exact identity at epoch zero")
    inner_baseline = volume_summary(inner_truth, inner_base)
    best_score = _checkpoint_score(inner_baseline, inner_baseline)
    if best_score is None:
        raise RuntimeError("Inner baseline failed its own checkpoint safety gate")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output / f"head_meta_fold{heldout_fold}.pth"
    history_path = output / f"history_meta_fold{heldout_fold}.csv"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": 0,
            "config": asdict(config),
            "feature_dim": feature_dim,
            "output_labels": SUBTYPE_LABELS,
            "heldout_fold": heldout_fold,
            "inner_validation_fold": inner_fold,
            "training_folds": training_folds,
            "checkpoint_provenance": checkpoint_provenance,
        },
        checkpoint_path,
    )
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "checkpoint_score": best_score,
            "checkpoint_eligible": True,
            "inner_total_volume_mae_ml": inner_baseline[
                "total_volume_mae_ml"
            ],
            "inner_total_volume_bias_ml": inner_baseline[
                "total_volume_bias_ml"
            ],
            "inner_normal_fpr": inner_baseline[
                "normal_false_positive_rate_at_0_1ml"
            ],
            "inner_presence_f1": inner_baseline["presence_f1_at_0_1ml"],
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        components_history = {
            "loss": [],
            "slice": [],
            "study": [],
            "total": [],
        }
        for step, batch in enumerate(training_loader, start=1):
            features = batch["features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            base_slices = batch["base_slice_volumes"].to(
                device, non_blocking=True
            )
            lengths = batch["lengths"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            candidate_slices = model(
                features, base_logits, base_slices, lengths
            )
            components = temporal_volume_loss(
                candidate_slices,
                batch["target_slice_volumes"].to(device, non_blocking=True),
                batch["spatial_known"].to(device, non_blocking=True),
                batch["study_target_volumes"].to(device, non_blocking=True),
                lengths,
                slice_pos_weight=slice_pos_weight,
                study_pos_weight=study_pos_weight,
                study_loss_weight=config.study_loss_weight,
                total_loss_weight=config.total_loss_weight,
                huber_beta=config.huber_beta,
            )
            components["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            for name, value in components.items():
                components_history[name].append(float(value.detach().cpu()))
            if config.max_train_steps and step >= config.max_train_steps:
                break

        observed_studies, observed_truth, _, inner_candidate = _predict(
            model, inner_loader, device=device
        )
        if observed_studies != inner_studies or not np.array_equal(
            observed_truth, inner_truth
        ):
            raise RuntimeError("Inner-validation order changed during training")
        inner_metrics = volume_summary(inner_truth, inner_candidate)
        checkpoint_score = _checkpoint_score(inner_metrics, inner_baseline)
        epoch_payload = {
            "epoch": epoch,
            "checkpoint_score": checkpoint_score,
            "checkpoint_eligible": checkpoint_score is not None,
            "inner_total_volume_mae_ml": inner_metrics[
                "total_volume_mae_ml"
            ],
            "inner_total_volume_bias_ml": inner_metrics[
                "total_volume_bias_ml"
            ],
            "inner_normal_fpr": inner_metrics[
                "normal_false_positive_rate_at_0_1ml"
            ],
            "inner_presence_f1": inner_metrics["presence_f1_at_0_1ml"],
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in components_history.items()
            },
        }
        history.append(epoch_payload)
        pd.DataFrame(history).to_csv(history_path, index=False)
        mlflow.log_metrics(
            {
                f"meta_fold{heldout_fold}_{key}": float(value)
                for key, value in epoch_payload.items()
                if key != "epoch" and value is not None
            },
            step=epoch,
        )
        print(
            json.dumps(
                {"meta_heldout_fold": heldout_fold, **epoch_payload},
                sort_keys=True,
            ),
            flush=True,
        )
        if checkpoint_score is not None and checkpoint_score > best_score + 1e-6:
            best_score = checkpoint_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "feature_dim": feature_dim,
                    "output_labels": SUBTYPE_LABELS,
                    "heldout_fold": heldout_fold,
                    "inner_validation_fold": inner_fold,
                    "training_folds": training_folds,
                    "checkpoint_provenance": checkpoint_provenance,
                    "inner_validation_metrics": inner_metrics,
                    "inner_validation_delta": volume_deltas(
                        inner_baseline, inner_metrics
                    ),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if config.max_train_steps or stale_epochs >= config.patience:
            break

    best_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    _, _, _, best_inner_volumes = _predict(model, inner_loader, device=device)
    best_inner_metrics = volume_summary(inner_truth, best_inner_volumes)
    inner_delta = volume_deltas(inner_baseline, best_inner_metrics)
    inner_decision = temporal_volume_promotion_decision(inner_delta)

    heldout_studies, heldout_truth, heldout_base, heldout_candidate = _predict(
        model, heldout_loader, device=device
    )
    if heldout_studies != heldout_dataset.study_ids:
        raise RuntimeError("Meta-heldout study order changed")
    heldout_baseline = volume_summary(heldout_truth, heldout_base)
    heldout_metrics = volume_summary(heldout_truth, heldout_candidate)
    heldout_delta = volume_deltas(heldout_baseline, heldout_metrics)
    result = {
        "heldout_fold": heldout_fold,
        "inner_validation_fold": inner_fold,
        "training_folds": training_folds,
        "training_studies": int(len(training_dataset)),
        "inner_validation_studies": len(inner_dataset),
        "heldout_studies": len(heldout_dataset),
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "inner_baseline": inner_baseline,
        "inner_candidate": best_inner_metrics,
        "inner_delta": inner_delta,
        "inner_promotion_decision": inner_decision,
        "heldout_baseline": heldout_baseline,
        "heldout_candidate": heldout_metrics,
        "heldout_delta": heldout_delta,
        "heldout_was_not_used_for_checkpoint_selection": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return {
        "summary": result,
        "study_ids": heldout_studies,
        "patient_ids": heldout_dataset.patient_ids,
        "truth": heldout_truth,
        "baseline": heldout_base,
        "candidate": heldout_candidate,
    }


def _compact_volume_metrics(
    truth: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    truth_total = truth.sum(axis=1)
    predicted_total = predicted.sum(axis=1)
    target_present = truth_total > 0.0
    predicted_present = predicted_total >= 0.1
    normal = ~target_present
    return {
        "total_volume_mae_ml": float(
            np.mean(np.abs(predicted_total - truth_total))
        ),
        "absolute_total_volume_bias_ml": abs(
            float(np.mean(predicted_total - truth_total))
        ),
        "normal_false_positive_rate_at_0_1ml": float(
            np.mean(predicted_present[normal]) if np.any(normal) else 0.0
        ),
        "presence_f1_at_0_1ml": float(
            f1_score(target_present, predicted_present, zero_division=0)
        ),
    }


def paired_patient_volume_bootstrap(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    patient_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    if not (
        len(truth) == len(baseline) == len(candidate) == len(patient_ids)
    ):
        raise ValueError("Bootstrap arrays have different lengths")
    unique_patients = np.unique(patient_ids.astype(str))
    patient_rows = {
        patient: np.flatnonzero(patient_ids.astype(str) == patient)
        for patient in unique_patients
    }
    directions = {
        "total_volume_mae_ml": False,
        "absolute_total_volume_bias_ml": False,
        "normal_false_positive_rate_at_0_1ml": False,
        "presence_f1_at_0_1ml": True,
    }
    baseline_point = _compact_volume_metrics(truth, baseline)
    candidate_point = _compact_volume_metrics(truth, candidate)
    deltas = {name: [] for name in directions}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        selected_patients = rng.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        indices = np.concatenate(
            [patient_rows[str(patient)] for patient in selected_patients]
        )
        baseline_metrics = _compact_volume_metrics(
            truth[indices], baseline[indices]
        )
        candidate_metrics = _compact_volume_metrics(
            truth[indices], candidate[indices]
        )
        for name in directions:
            deltas[name].append(candidate_metrics[name] - baseline_metrics[name])
    result: dict[str, object] = {
        "resampling_unit": "patient",
        "samples": samples,
        "seed": seed,
        "patients": int(len(unique_patients)),
        "metrics": {},
    }
    for name, higher_is_better in directions.items():
        values = np.asarray(deltas[name], dtype=np.float64)
        probability = float(
            np.mean(values > 0) + 0.5 * np.mean(values == 0)
            if higher_is_better
            else np.mean(values < 0) + 0.5 * np.mean(values == 0)
        )
        result["metrics"][name] = {
            "baseline": baseline_point[name],
            "candidate": candidate_point[name],
            "candidate_minus_baseline": candidate_point[name]
            - baseline_point[name],
            "delta_ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "higher_is_better": higher_is_better,
            "bootstrap_probability_candidate_better": probability,
        }
    return result


def crossfit_volume_promotion_decision(
    delta: dict[str, object], bootstrap: dict[str, object]
) -> dict[str, object]:
    volume_decision = temporal_volume_promotion_decision(delta)
    mae_bootstrap = bootstrap["metrics"]["total_volume_mae_ml"]
    bootstrap_checks = {
        "mae_bootstrap_probability": float(
            mae_bootstrap["bootstrap_probability_candidate_better"]
        ) >= CROSSFIT_GATE["minimum_mae_bootstrap_probability"],
        "mae_ci95_upper": float(mae_bootstrap["delta_ci95"][1])
        <= CROSSFIT_GATE["maximum_mae_delta_ci95_upper_ml"],
    }
    return {
        "criteria": CROSSFIT_GATE,
        "volume_gate_checks": volume_decision["checks"],
        "bootstrap_checks": bootstrap_checks,
        "promotion_allowed": bool(
            volume_decision["promotion_allowed"]
            and all(bootstrap_checks.values())
        ),
    }


def run(config: CrossfitTemporalVolumeConfig) -> dict[str, object]:
    checkpoint_paths = parse_fold_checkpoints(config.fold_checkpoints)
    heldout_folds = tuple(sorted(set(config.heldout_folds)))
    if not heldout_folds or not set(heldout_folds).issubset(EXPECTED_FOLDS):
        raise ValueError("heldout_folds must be a non-empty subset of 0..4")
    if min(config.epochs, config.patience, config.study_batch_size) < 1:
        raise ValueError("Crossfit epochs, patience and batch size must be positive")
    if config.max_train_steps is not None and config.max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Crossfit volume training requires CUDA BF16 support")
    if tuple(VOLUME_KEYS) != tuple(f"V_{label}" for label in SUBTYPE_LABELS):
        raise RuntimeError("ICH volume key order does not match subtype order")

    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **asdict(config),
        "heldout_folds": heldout_folds,
        "inner_validation_policy": "first available of [3, 4, 0]",
        "development_status": "adaptive_oof_not_confirmatory",
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    run_kind = (
        "smoke"
        if config.max_train_steps is not None or heldout_folds != EXPECTED_FOLDS
        else "adaptive_development_oof"
    )
    device = torch.device("cuda")
    _seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats(device)
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    if run_kind != "smoke":
        notify_campaign(
            "start",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی\n\n🚀 meta-OOF حجمی exp55 آغاز شد. "
            "هر fold فقط با checkpointی feature می‌گیرد که همان fold را در training "
            "ندیده است؛ head هر fold نیز روی یک inner fold جدا انتخاب و سپس heldout را "
            "فقط یک‌بار می‌بیند.\n\n🔎 تحلیل: این طراحی خطای in-sample کشف‌شده در "
            "exp54 را حذف می‌کند. نتیجه توسعه‌ای و adaptive است؛ معیار عبور هم‌زمان "
            "بهبود MAE/bias، ایمنی FPR/F1/مرزهای triage و bootstrap بیمارمحور است.",
            run=config.run_name,
            kind=run_kind,
            detail=f"heldout_folds={heldout_folds}; final confirmation=leaderboard",
        )
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as mlflow_run:
            mlflow.set_tags(
                {
                    "task": "ich_segmentation_volume",
                    "stage": "crossfitted_temporal_volume_residual_head",
                    "run_kind": run_kind,
                    "evaluation_status": "adaptive_development_oof",
                    "leaderboard_confirmation_required": "true",
                    "git_commit": git_commit(),
                }
            )
            manifest = load_segmentation_manifest(config.manifest_path)
            manifest_folds = tuple(
                sorted(
                    manifest[MANIFEST_FOLD_COLUMN]
                    .astype(int)
                    .unique()
                    .tolist()
                )
            )
            if manifest_folds != EXPECTED_FOLDS:
                raise ValueError(
                    f"Crossfit manifest folds are {manifest_folds}, expected 0..4"
                )
            manifest_sha = file_sha256(config.manifest_path)
            truth, truth_source = ground_truth_ich_context()
            truth = truth.loc[
                :, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]
            ]
            cache_root = Path(config.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_metadata: dict[int, dict[str, object]] = {}
            checkpoint_provenance: dict[int, dict[str, object]] = {}
            datasets: dict[int, ICHSequenceVolumeDataset] = {}
            for fold in EXPECTED_FOLDS:
                checkpoint = checkpoint_paths[fold]
                model, provenance = _load_oof_model(
                    checkpoint, expected_fold=fold, device=device
                )
                checkpoint_provenance[fold] = provenance
                fold_frame = manifest.loc[
                    manifest[MANIFEST_FOLD_COLUMN].astype(int) == fold
                ].copy()
                cache_path = cache_root / (
                    f"oof_volume_fold{fold}_{str(provenance['sha256'])[:12]}.npz"
                )
                cache_metadata[fold] = _ensure_volume_cache(
                    model,
                    fold_frame,
                    cache_path,
                    checkpoint_sha256=str(provenance["sha256"]),
                    manifest_sha256=manifest_sha,
                    split_name=f"adaptive_oof_fold_{fold}",
                    device=device,
                    batch_size=config.extraction_batch_size,
                    workers=config.workers,
                )
                del model
                torch.cuda.empty_cache()
                datasets[fold] = ICHSequenceVolumeDataset(cache_path, truth)

            feature_dims = {
                int(dataset.embeddings.shape[1]) for dataset in datasets.values()
            }
            if len(feature_dims) != 1:
                raise ValueError("OOF feature caches have different dimensions")
            mlflow.log_params(
                {
                    **{
                        key: value
                        for key, value in asdict(config).items()
                        if key not in {"fold_checkpoints", "heldout_folds"}
                    },
                    "heldout_folds": ",".join(map(str, heldout_folds)),
                    "feature_dim": feature_dims.pop(),
                    "manifest_sha256": manifest_sha,
                    "fold_count": len(EXPECTED_FOLDS),
                    "meta_heldout_not_used_for_selection": True,
                }
            )
            for fold, provenance in checkpoint_provenance.items():
                mlflow.log_params(
                    {
                        f"fold{fold}_checkpoint_sha256": provenance["sha256"],
                        f"fold{fold}_checkpoint_calibration_fold": provenance[
                            "checkpoint_calibration_fold"
                        ],
                        f"fold{fold}_cache_sha256": cache_metadata[fold][
                            "cache_sha256"
                        ],
                    }
                )

            fold_results: list[dict[str, object]] = []
            all_study_ids: list[str] = []
            all_patient_ids: list[str] = []
            all_folds: list[int] = []
            all_truth: list[np.ndarray] = []
            all_baseline: list[np.ndarray] = []
            all_candidate: list[np.ndarray] = []
            for heldout_fold in heldout_folds:
                fold_result = _train_meta_fold(
                    config,
                    heldout_fold=heldout_fold,
                    datasets=datasets,
                    checkpoint_provenance=checkpoint_provenance,
                    output=output,
                    device=device,
                )
                fold_results.append(fold_result["summary"])
                count = len(fold_result["study_ids"])
                all_study_ids.extend(fold_result["study_ids"])
                all_patient_ids.extend(fold_result["patient_ids"])
                all_folds.extend([heldout_fold] * count)
                all_truth.append(fold_result["truth"])
                all_baseline.append(fold_result["baseline"])
                all_candidate.append(fold_result["candidate"])

            truth_array = np.concatenate(all_truth)
            baseline_array = np.concatenate(all_baseline)
            candidate_array = np.concatenate(all_candidate)
            prediction_frame = pd.DataFrame(
                {
                    "study_id": all_study_ids,
                    "patient_id": all_patient_ids,
                    "meta_heldout_fold": all_folds,
                }
            )
            for index, volume_key in enumerate(VOLUME_KEYS):
                prediction_frame[f"gt_{volume_key}"] = truth_array[:, index]
                prediction_frame[f"base_{volume_key}"] = baseline_array[:, index]
                prediction_frame[f"candidate_{volume_key}"] = candidate_array[:, index]
            if prediction_frame["study_id"].duplicated().any():
                raise RuntimeError("A study occurs in multiple meta-heldout folds")
            patient_fold_count = prediction_frame.groupby("patient_id")[
                "meta_heldout_fold"
            ].nunique()
            if int(patient_fold_count.max()) != 1:
                raise RuntimeError("A patient occurs in multiple meta-heldout folds")
            prediction_frame = prediction_frame.sort_values("study_id").reset_index(
                drop=True
            )
            prediction_frame.to_csv(
                output / "oof_study_volume_predictions.csv", index=False
            )

            baseline_metrics = volume_summary(truth_array, baseline_array)
            candidate_metrics = volume_summary(truth_array, candidate_array)
            delta = volume_deltas(baseline_metrics, candidate_metrics)
            bootstrap: dict[str, object] | None = None
            decision: dict[str, object] | None = None
            if heldout_folds == EXPECTED_FOLDS and config.max_train_steps is None:
                locked_payload = json.loads(
                    Path(config.locked_oof_summary).read_text(encoding="utf-8")
                )
                _validate_locked_baseline(
                    baseline_metrics, _evaluation_summary(locked_payload)
                )
                bootstrap = paired_patient_volume_bootstrap(
                    truth_array,
                    baseline_array,
                    candidate_array,
                    np.asarray(all_patient_ids),
                    samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                decision = crossfit_volume_promotion_decision(delta, bootstrap)

            duration = time.perf_counter() - started
            summary: dict[str, object] = {
                "analysis_kind": "ich_crossfitted_temporal_volume_residual_head",
                "run_name": config.run_name,
                "run_id": mlflow_run.info.run_id,
                "run_kind": run_kind,
                "development_status": "adaptive_oof_not_confirmatory",
                "heldout_folds": heldout_folds,
                "studies": int(len(prediction_frame)),
                "patients": int(prediction_frame["patient_id"].nunique()),
                "duration_s": duration,
                "peak_vram_gb": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "manifest_sha256": manifest_sha,
                "locked_oof_summary": config.locked_oof_summary,
                "checkpoint_provenance": checkpoint_provenance,
                "feature_cache": cache_metadata,
                "fold_results": fold_results,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": delta,
                "paired_patient_bootstrap": bootstrap,
                "promotion_decision": decision,
                "truth_source": str(truth_source),
                "git_commit": git_commit(),
                "spatial_masks_unchanged_by_design": True,
                "leaderboard_confirmation_required": True,
            }
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            mlflow.log_metrics(
                {
                    "oof_delta_total_volume_mae_ml": float(
                        delta["total_volume_mae_ml"]
                    ),
                    "oof_delta_absolute_total_volume_bias_ml": float(
                        delta["absolute_total_volume_bias_ml"]
                    ),
                    "oof_delta_normal_fpr": float(
                        delta["normal_false_positive_rate_at_0_1ml"]
                    ),
                    "oof_delta_presence_f1": float(
                        delta["presence_f1_at_0_1ml"]
                    ),
                    "promotion_allowed": float(
                        decision["promotion_allowed"] if decision else 0.0
                    ),
                    "duration_s": duration,
                }
            )
            mlflow.log_artifacts(
                str(output), artifact_path="ich_crossfit_temporal_volume"
            )

        if run_kind != "smoke":
            event = "success" if decision and decision["promotion_allowed"] else "warning"
            mae_probability = (
                float(
                    bootstrap["metrics"]["total_volume_mae_ml"][
                        "bootstrap_probability_candidate_better"
                    ]
                )
                if bootstrap
                else 0.0
            )
            notify_campaign(
                event,
                "🧠 مسابقه IAAA 2026 | مدل خونریزی\n\n📊 meta-OOF پنج‌fold "
                f"exp55 تمام شد: delta MAE={float(delta['total_volume_mae_ml']):+.3f}mL، "
                f"delta |bias|={float(delta['absolute_total_volume_bias_ml']):+.3f}mL، "
                f"delta FPR={float(delta['normal_false_positive_rate_at_0_1ml']):+.3f} "
                f"و delta F1={float(delta['presence_f1_at_0_1ml']):+.3f}؛ "
                f"P(MAE بهتر)={mae_probability:.3f} و promotion="
                f"{bool(decision and decision['promotion_allowed'])}.\n\n🔎 تحلیل: هر heldout "
                "برای انتخاب epoch استفاده نشده است و bootstrap در سطح بیمار انجام شد؛ "
                "بااین‌حال OOF تاریخی adaptive است. اقدام بعدی فقط در صورت عبور همه "
                "gateها ساخت headهای deployable است؛ تأیید قطعی فقط leaderboard واقعی.",
                run=config.run_name,
                kind=run_kind,
                detail=f"MLflow {summary['run_id']}; adaptive OOF",
            )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی\n\n⚠️ اجرای meta-OOF حجمی "
            "exp55 با خطای فنی متوقف شد.\n\n🔎 تحلیل: این رخداد نتیجهٔ کیفیتی نیست "
            "و هیچ leaderboardی مصرف نشده است. اقدام بعدی: حفظ artifactها، اصلاح "
            "کوچک‌ترین علت و تکرار smoke بدون تغییر recipe.",
            run=config.run_name,
            kind=run_kind,
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
    parser.add_argument("--locked-oof-summary", required=True)
    parser.add_argument(
        "--fold-checkpoint",
        dest="fold_checkpoints",
        action="append",
        required=True,
        help="Repeat exactly five times as FOLD=PATH",
    )
    parser.add_argument(
        "--heldout-fold", dest="heldout_folds", action="append", type=int
    )
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
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int)
    arguments = vars(parser.parse_args())
    arguments["fold_checkpoints"] = tuple(arguments["fold_checkpoints"])
    arguments["heldout_folds"] = tuple(
        EXPECTED_FOLDS
        if arguments["heldout_folds"] is None
        else arguments["heldout_folds"]
    )
    config = CrossfitTemporalVolumeConfig(**arguments)
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
