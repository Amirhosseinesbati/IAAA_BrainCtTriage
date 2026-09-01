"""Train a leakage-gated temporal residual head on frozen ICH encoder features."""

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
    segmentation_classification_weights,
    split_segmentation_slices,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.temporal_head import (
    ICHSequenceFeatureDataset,
    TemporalResidualHead,
    auc_summary,
    collate_ich_sequences,
    extract_frozen_encoder_features,
    temporal_classification_loss,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


GATE = {
    "minimum_selection_proxy_delta": 0.002,
    "minimum_macro_subtype_auc_delta": 0.005,
    "minimum_any_ich_auc_delta": -0.002,
    "minimum_subtype_auc_delta": -0.01,
}


@dataclass(frozen=True)
class TemporalTrainConfig:
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
    extraction_batch_size: int = 32
    workers: int = 4
    maximum_pos_weight: float = 20.0
    seed: int = 42
    max_train_steps: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluation_summary(payload: dict[str, object]) -> dict[str, object]:
    rescored = payload.get("rescored_summary")
    if isinstance(rescored, dict) and "selection_score" in rescored:
        return rescored
    if "selection_score" in payload:
        return payload
    raise ValueError("Baseline JSON does not contain a selection summary")


def _study_pos_weight(
    truth: pd.DataFrame, study_ids: list[str], maximum: float
) -> torch.Tensor:
    selected = truth.loc[truth["study_id"].astype(str).isin(study_ids)].copy()
    if selected["study_id"].astype(str).nunique() != len(set(study_ids)):
        raise ValueError("Training truth does not cover every temporal study")
    subtype = np.column_stack([
        selected[f"gt_{key}"].to_numpy(dtype=np.float64) > 0
        for key in VOLUME_KEYS
    ])
    targets = np.column_stack([subtype.any(axis=1), subtype]).astype(np.float64)
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    if np.any(positives <= 0):
        raise ValueError("Every temporal output needs positive training studies")
    return torch.as_tensor(
        np.clip(negatives / positives, 1.0, maximum), dtype=torch.float32
    )


def _cache_metadata(
    cache_path: Path,
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    split_name: str,
    frame: pd.DataFrame,
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
    }


def _ensure_feature_cache(
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
    )
    if cache_path.exists() and metadata_path.exists():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {key: observed.get(key) for key in expected}
        if comparable == expected and observed.get("cache_sha256") == file_sha256(
            cache_path
        ):
            return observed
        raise ValueError(f"Refusing to reuse stale temporal cache: {cache_path}")
    extracted = extract_frozen_encoder_features(
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
    model: TemporalResidualHead,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    study_ids: list[str] = []
    truths: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            logits = model(features, base_logits, lengths)
            steps = torch.arange(logits.shape[1], device=device)[None, :]
            padding = steps < lengths[:, None]
            baseline = torch.sigmoid(base_logits.float()).masked_fill(
                ~padding[:, :, None], -1.0
            ).amax(dim=1)
            candidate = torch.sigmoid(logits.float()).masked_fill(
                ~padding[:, :, None], -1.0
            ).amax(dim=1)
            study_ids.extend(batch["study_id"])
            truths.append(batch["study_target"].numpy())
            baseline_scores.append(baseline.cpu().numpy())
            candidate_scores.append(candidate.cpu().numpy())
    return (
        study_ids,
        np.concatenate(truths),
        np.concatenate(baseline_scores),
        np.concatenate(candidate_scores),
    )


def _deltas(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    any_delta = float(candidate["any_ich_auc"]) - float(
        baseline["any_ich_auc"]
    )
    macro_delta = float(candidate["macro_subtype_auc"]) - float(
        baseline["macro_subtype_auc"]
    )
    return {
        "any_ich_auc": any_delta,
        "macro_subtype_auc": macro_delta,
        "selection_proxy": 0.30 * any_delta + 0.15 * macro_delta,
        "subtype_auc": {
            label: float(candidate["subtype_auc"][label])
            - float(baseline["subtype_auc"][label])
            for label in OUTPUT_LABELS[1:]
        },
    }


def temporal_promotion_decision(delta: dict[str, object]) -> dict[str, object]:
    checks = {
        "selection_proxy": float(delta["selection_proxy"])
        >= GATE["minimum_selection_proxy_delta"],
        "macro_subtype_auc": float(delta["macro_subtype_auc"])
        >= GATE["minimum_macro_subtype_auc_delta"],
        "any_ich_auc_noninferiority": float(delta["any_ich_auc"])
        >= GATE["minimum_any_ich_auc_delta"],
        "subtype_safety": min(
            float(value) for value in delta["subtype_auc"].values()
        )
        >= GATE["minimum_subtype_auc_delta"],
    }
    return {
        "criteria": GATE,
        "checks": checks,
        "promotion_allowed": bool(all(checks.values())),
    }


def run(config: TemporalTrainConfig) -> dict[str, object]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("Outer and calibration folds must differ")
    if min(config.epochs, config.patience, config.study_batch_size) < 1:
        raise ValueError("Temporal epochs, patience and batch size must be positive")
    if config.max_train_steps is not None and config.max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Temporal ICH training requires CUDA BF16 support")

    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_temporal_head.pth"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {checkpoint_path}")
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    base_payload = torch.load(
        Path(config.base_checkpoint), map_location="cpu", weights_only=True
    )
    source_config = base_payload.get("config", {})
    if (
        int(source_config.get("outer_fold", -1)) != config.outer_fold
        or int(source_config.get("calibration_fold", -1)) != config.calibration_fold
    ):
        raise ValueError("Base checkpoint does not match temporal held-out folds")
    if base_payload.get("output_labels") != OUTPUT_LABELS:
        raise ValueError("Base checkpoint labels do not match temporal labels")
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
    temporal = TemporalResidualHead(
        feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    trainable_parameters = list(temporal.parameters())
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    slice_pos_weight = segmentation_classification_weights(
        training, maximum=config.maximum_pos_weight
    ).to(device)
    study_pos_weight = _study_pos_weight(
        truth,
        train_dataset.study_ids,
        config.maximum_pos_weight,
    ).to(device)

    _, calibration_truth, baseline_scores, epoch0_scores = _predict(
        temporal, calibration_loader, device=device
    )
    if not np.array_equal(baseline_scores, epoch0_scores):
        raise RuntimeError("Zero-initialized temporal head is not an exact identity")
    baseline_metrics = auc_summary(calibration_truth, baseline_scores)
    baseline_payload = json.loads(
        Path(config.baseline_summary).read_text(encoding="utf-8")
    )
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
        raise RuntimeError("Extracted temporal baseline does not match locked v4 metrics")
    baseline_selection = float(baseline_reference["selection_score"])
    best_proxy = baseline_selection
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    epoch0_delta = _deltas(baseline_metrics, baseline_metrics)
    history.append({
        "epoch": 0,
        "selection_proxy": best_proxy,
        "any_ich_auc": baseline_metrics["any_ich_auc"],
        "macro_subtype_auc": baseline_metrics["macro_subtype_auc"],
        "delta_selection_proxy": epoch0_delta["selection_proxy"],
        "delta_any_ich_auc": epoch0_delta["any_ich_auc"],
        "delta_macro_subtype_auc": epoch0_delta["macro_subtype_auc"],
    })
    torch.save({
        "state_dict": temporal.state_dict(),
        "epoch": 0,
        "config": asdict(config),
        "feature_dim": feature_dim,
        "output_labels": OUTPUT_LABELS,
        "base_checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
    }, checkpoint_path)

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    run_kind = "smoke" if config.max_train_steps else "calibration_screen"
    if run_kind != "smoke":
        notify_campaign(
            "start",
            "غربال temporal residual head مدل ICH آغاز شد. logits incumbent در "
            "epoch صفر دقیقاً حفظ شده‌اند و فقط یک BiGRU کم‌ظرفیت روی featureهای "
            "frozen encoder آموزش می‌بیند. تحلیل کوتاه: mask و حجم اصلاً وارد مسیر "
            "update نمی‌شوند و outer2 خوانده نخواهد شد؛ سود فقط وقتی معتبر است که "
            "Any و macro-AUC با هم و بدون افت subtype بهتر شوند. اقدام بعدی: اعمال "
            "گیت calibration و توقف کامل پیش از outer در صورت شکست.",
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
                "stage": "frozen_encoder_temporal_residual_head",
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
                "calibration_selection_proxy": best_proxy,
                "calibration_any_ich_auc": float(baseline_metrics["any_ich_auc"]),
                "calibration_macro_subtype_auc": float(
                    baseline_metrics["macro_subtype_auc"]
                ),
            }, step=0)

            for epoch in range(1, config.epochs + 1):
                temporal.train()
                component_history = {"loss": [], "slice": [], "study": []}
                for step, batch in enumerate(train_loader, start=1):
                    features = batch["features"].to(device, non_blocking=True)
                    base_logits = batch["base_logits"].to(device, non_blocking=True)
                    lengths = batch["lengths"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = temporal(features, base_logits, lengths)
                    components = temporal_classification_loss(
                        logits,
                        batch["slice_targets"].to(device, non_blocking=True),
                        batch["classification_known"].to(device, non_blocking=True),
                        batch["study_target"].to(device, non_blocking=True),
                        lengths,
                        slice_pos_weight=slice_pos_weight,
                        study_pos_weight=study_pos_weight,
                        study_loss_weight=config.study_loss_weight,
                        focal_gamma=config.focal_gamma,
                    )
                    components["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
                    optimizer.step()
                    for name, value in components.items():
                        component_history[name].append(float(value.detach().cpu()))
                    if config.max_train_steps and step >= config.max_train_steps:
                        break

                _, observed_truth, _, candidate_scores = _predict(
                    temporal, calibration_loader, device=device
                )
                if not np.array_equal(observed_truth, calibration_truth):
                    raise RuntimeError("Calibration study order changed during training")
                candidate_metrics = auc_summary(calibration_truth, candidate_scores)
                delta = _deltas(baseline_metrics, candidate_metrics)
                selection_proxy = baseline_selection + float(delta["selection_proxy"])
                epoch_payload = {
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
                history.append(epoch_payload)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics({
                    key: float(value)
                    for key, value in epoch_payload.items()
                    if key != "epoch"
                }, step=epoch)
                print(json.dumps(epoch_payload, sort_keys=True), flush=True)
                if selection_proxy > best_proxy + 1e-6:
                    best_proxy = selection_proxy
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({
                        "state_dict": temporal.state_dict(),
                        "epoch": epoch,
                        "config": asdict(config),
                        "feature_dim": feature_dim,
                        "output_labels": OUTPUT_LABELS,
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
            _, _, _, best_scores = _predict(
                temporal, calibration_loader, device=device
            )
            candidate_metrics = auc_summary(calibration_truth, best_scores)
            delta = _deltas(baseline_metrics, candidate_metrics)
            decision = temporal_promotion_decision(delta)
            duration = time.perf_counter() - started
            summary: dict[str, object] = {
                "analysis_kind": "ich_frozen_encoder_temporal_residual_head",
                "run_name": config.run_name,
                "run_id": mlflow_run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
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
                "baseline_selection_score_unchanged_spatial": baseline_selection,
                "candidate_selection_proxy_unchanged_spatial": baseline_selection
                + float(delta["selection_proxy"]),
                "promotion_decision": decision,
                "spatial_volume_metrics_unchanged_by_design": True,
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
                "delta_selection_proxy": float(delta["selection_proxy"]),
                "delta_any_ich_auc": float(delta["any_ich_auc"]),
                "delta_macro_subtype_auc": float(delta["macro_subtype_auc"]),
                "promotion_allowed": float(decision["promotion_allowed"]),
                "duration_s": duration,
            })
            mlflow.log_artifacts(str(output), artifact_path="ich_temporal_residual_head")

        if run_kind != "smoke":
            event = "success" if decision["promotion_allowed"] else "warning"
            notify_campaign(
                event,
                f"غربال temporal residual head مدل ICH تمام شد. delta selection="
                f"{float(delta['selection_proxy']):+.5f}، Any-AUC="
                f"{float(delta['any_ich_auc']):+.5f} و macro-AUC="
                f"{float(delta['macro_subtype_auc']):+.5f} است؛ promotion="
                f"{decision['promotion_allowed']}. تحلیل کوتاه: mask و حجم به‌علت "
                "freeze کامل بدون تغییرند و outer2 هنوز دیده نشده است؛ تصمیم فقط "
                "از ranking calibration و گیت‌های ازپیش‌ثبت‌شده آمده است. اقدام بعدی: "
                "در صورت عبور، ارزیابی یک‌باره outer2؛ در غیر این صورت بستن این "
                "معماری بدون sweep پس‌نگر.",
                run=config.run_name,
                kind=run_kind,
                best_epoch=best_epoch,
                detail=f"MLflow {summary['run_id']}; outer=false",
            )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "اجرای temporal residual head مدل ICH متوقف شد. تحلیل کوتاه: این خطا "
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
    parser.add_argument("--study-loss-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--study-batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int)
    config = TemporalTrainConfig(**vars(parser.parse_args()))
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
