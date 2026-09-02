"""Patient-disjoint five-fold OOF evaluation of the Exp88 temporal subtype head."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.train_ich_subtype_temporal_residual_local import (
    _seed_everything,
    _write_json,
    subtype_temporal_delta,
    subtype_temporal_promotion_decision,
)
from scripts.train_ich_temporal_residual_head import (
    _ensure_feature_cache,
    _predict,
    _study_pos_weight,
)
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
    SubtypeTemporalResidualHead,
    auc_summary,
    collate_ich_sequences,
    temporal_classification_loss,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import file_sha256, git_commit


FOLD_SPECS = (
    {
        "outer_fold": 0,
        "calibration_fold": 1,
        "checkpoint": "reports/ich_experiments/2p5d_segmentation/"
        "exp20_hardempty001_fprselect_p1_audited_v3_f0/best.pth",
        "sha256": "62d696d4d5f45f83c8a477893718142d643614203e025f55a929b9bc3539f1ef",
    },
    {
        "outer_fold": 1,
        "calibration_fold": 2,
        "checkpoint": "reports/ich_experiments/2p5d_segmentation/"
        "exp21b_hardempty001_fprselect_p1_audited_v3_f1_cal2/best.pth",
        "sha256": "b3125e8c8a0994875b218d6dff5085dc63d619aa04660d68717fb354e7be9d4a",
    },
    {
        "outer_fold": 2,
        "calibration_fold": 1,
        "checkpoint": "reports/ich_experiments/2p5d_segmentation/"
        "exp22_hardempty001_fprselect_p1_audited_v3_f2/best.pth",
        "sha256": "c63e609c652c2c15c2051c0c8da58a8060de31a7f67ecfa0f48d5e6d4338191d",
    },
    {
        "outer_fold": 3,
        "calibration_fold": 1,
        "checkpoint": "reports/ich_experiments/2p5d_segmentation/"
        "exp18_hardempty001_fprselect_p1_audited_v3_f3/best.pth",
        "sha256": "cef60d76040c22196511c5b6671c5bd057f4a9b3badbd2a5ba3d1149e6289084",
    },
    {
        "outer_fold": 4,
        "calibration_fold": 1,
        "checkpoint": "reports/ich_experiments/2p5d_segmentation/"
        "exp19_hardempty001_fprselect_p1_audited_v3_f4/best.pth",
        "sha256": "a5c9688563455048b47a790c09e641f50bc1267583030767d37025c806f8f02e",
    },
)

LOCKED_BASELINE = {
    "studies": 338,
    "patients": 320,
    "any_ich_auc": 0.9345288326300984,
    "macro_subtype_auc": 0.8291593268101609,
}

OOF_GATE = {
    "minimum_macro_subtype_auc_delta": 0.005,
    "minimum_selection_proxy_delta": 0.00075,
    "minimum_subtype_auc_delta": -0.01,
    "minimum_improved_subtypes": 3,
    "minimum_nonnegative_macro_folds": 3,
    "minimum_worst_fold_macro_delta": -0.025,
    "minimum_bootstrap_probability": 0.90,
    "minimum_valid_bootstrap_fraction": 0.95,
    "any_ich_absolute_tolerance": 1e-12,
}

STRONG_OOF_GATE = {
    "minimum_nonnegative_macro_folds": 4,
    "minimum_worst_fold_macro_delta": -0.01,
    "minimum_bootstrap_probability": 0.95,
    "minimum_bootstrap_ci95_lower": 0.0,
}


@dataclass(frozen=True)
class CrossfitTemporalConfig:
    run_name: str
    output_dir: str
    cache_dir: str
    manifest_path: str
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
    bootstrap_samples: int = 2000
    seed: int = 42


def validate_fold_specs(specs: tuple[dict[str, Any], ...]) -> None:
    if len(specs) != 5:
        raise ValueError("Exp89 requires exactly five fold specifications")
    outer = [int(spec["outer_fold"]) for spec in specs]
    if sorted(outer) != list(range(5)):
        raise ValueError("Exp89 outer folds must cover 0..4 exactly once")
    for spec in specs:
        if int(spec["outer_fold"]) == int(spec["calibration_fold"]):
            raise ValueError("Outer and calibration folds must differ")
        if len(str(spec["sha256"])) != 64:
            raise ValueError("Every fold checkpoint needs a locked SHA256")


def _patient_overlap(
    training: pd.DataFrame, calibration: pd.DataFrame, outer: pd.DataFrame
) -> dict[str, int]:
    groups = {
        "training": set(training["patient_id"].astype(str)),
        "calibration": set(calibration["patient_id"].astype(str)),
        "outer": set(outer["patient_id"].astype(str)),
    }
    return {
        "training_calibration": len(groups["training"] & groups["calibration"]),
        "training_outer": len(groups["training"] & groups["outer"]),
        "calibration_outer": len(groups["calibration"] & groups["outer"]),
    }


def _metric_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "accepted_samples": int(len(array)),
        "delta_ci95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "bootstrap_probability_candidate_better": float(np.mean(array > 0.0)),
    }


def fold_auc_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """AUC delta that preserves undefined rare-class fold metrics as null."""
    any_baseline = baseline["any_ich_auc"]
    any_candidate = candidate["any_ich_auc"]
    any_delta = (
        None
        if any_baseline is None or any_candidate is None
        else float(any_candidate) - float(any_baseline)
    )
    subtype: dict[str, float | None] = {}
    for label in OUTPUT_LABELS[1:]:
        reference = baseline["subtype_auc"][label]
        observed = candidate["subtype_auc"][label]
        subtype[label] = (
            None
            if reference is None or observed is None
            else float(observed) - float(reference)
        )
    macro_delta = float(candidate["macro_subtype_auc"]) - float(
        baseline["macro_subtype_auc"]
    )
    return {
        "any_ich_auc": any_delta,
        "macro_subtype_auc": macro_delta,
        "selection_proxy": (
            None
            if any_delta is None
            else 0.30 * any_delta + 0.15 * macro_delta
        ),
        "subtype_auc": subtype,
    }


def paired_patient_auc_bootstrap(
    patient_ids: np.ndarray,
    truth: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not (
        len(patient_ids)
        == len(truth)
        == len(baseline_scores)
        == len(candidate_scores)
    ):
        raise ValueError("Bootstrap arrays must have the same row count")
    unique_patients = np.unique(patient_ids.astype(str))
    grouped = {
        patient: np.flatnonzero(patient_ids.astype(str) == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    macro_deltas: list[float] = []
    subtype_deltas: dict[str, list[float]] = {
        label: [] for label in OUTPUT_LABELS[1:]
    }
    for _ in range(samples):
        selected = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([grouped[str(patient)] for patient in selected])
        sampled_truth = truth[indices]
        if any(len(np.unique(sampled_truth[:, column])) < 2 for column in range(6)):
            continue
        baseline = auc_summary(sampled_truth, baseline_scores[indices])
        candidate = auc_summary(sampled_truth, candidate_scores[indices])
        delta = subtype_temporal_delta(baseline, candidate)
        macro_deltas.append(float(delta["macro_subtype_auc"]))
        for label in OUTPUT_LABELS[1:]:
            subtype_deltas[label].append(float(delta["subtype_auc"][label]))
    if len(macro_deltas) < max(1, int(np.ceil(samples * 0.90))):
        raise RuntimeError("Too few valid patient-bootstrap samples")
    return {
        "requested_samples": int(samples),
        "accepted_samples": int(len(macro_deltas)),
        "valid_fraction": float(len(macro_deltas) / samples),
        "resampling_unit": "patient_cluster",
        "seed": int(seed),
        "macro_subtype_auc": _metric_interval(macro_deltas),
        "selection_proxy": _metric_interval(
            [0.15 * value for value in macro_deltas]
        ),
        "subtype_auc": {
            label: _metric_interval(values)
            for label, values in subtype_deltas.items()
        },
    }


def oof_promotion_decision(
    delta: dict[str, Any],
    fold_deltas: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    *,
    any_logits_exact: bool,
    baseline_locked_match: bool,
    coverage_exact: bool,
) -> dict[str, Any]:
    subtype_values = [float(value) for value in delta["subtype_auc"].values()]
    fold_macro = [float(value["macro_subtype_auc"]) for value in fold_deltas]
    interval = bootstrap["macro_subtype_auc"]
    nonnegative = sum(value >= 0.0 for value in fold_macro)
    primary_checks = {
        "any_logits_exact": bool(any_logits_exact),
        "any_auc_exact": abs(float(delta["any_ich_auc"]))
        <= OOF_GATE["any_ich_absolute_tolerance"],
        "baseline_locked_match": bool(baseline_locked_match),
        "coverage_exact": bool(coverage_exact),
        "macro_subtype_auc": float(delta["macro_subtype_auc"])
        >= OOF_GATE["minimum_macro_subtype_auc_delta"],
        "selection_proxy": float(delta["selection_proxy"])
        >= OOF_GATE["minimum_selection_proxy_delta"],
        "subtype_safety": min(subtype_values)
        >= OOF_GATE["minimum_subtype_auc_delta"],
        "improved_subtypes": sum(value > 0.0 for value in subtype_values)
        >= OOF_GATE["minimum_improved_subtypes"],
        "nonnegative_macro_folds": nonnegative
        >= OOF_GATE["minimum_nonnegative_macro_folds"],
        "worst_fold_macro": min(fold_macro)
        >= OOF_GATE["minimum_worst_fold_macro_delta"],
        "bootstrap_probability": float(
            interval["bootstrap_probability_candidate_better"]
        )
        >= OOF_GATE["minimum_bootstrap_probability"],
        "bootstrap_valid_fraction": float(bootstrap["valid_fraction"])
        >= OOF_GATE["minimum_valid_bootstrap_fraction"],
    }
    primary_allowed = all(primary_checks.values())
    strong_checks = {
        "primary_gate": primary_allowed,
        "nonnegative_macro_folds": nonnegative
        >= STRONG_OOF_GATE["minimum_nonnegative_macro_folds"],
        "worst_fold_macro": min(fold_macro)
        >= STRONG_OOF_GATE["minimum_worst_fold_macro_delta"],
        "bootstrap_probability": float(
            interval["bootstrap_probability_candidate_better"]
        )
        >= STRONG_OOF_GATE["minimum_bootstrap_probability"],
        "bootstrap_ci95_lower": float(interval["delta_ci95"][0])
        > STRONG_OOF_GATE["minimum_bootstrap_ci95_lower"],
    }
    return {
        "criteria": OOF_GATE,
        "strong_criteria": STRONG_OOF_GATE,
        "primary_checks": primary_checks,
        "strong_checks": strong_checks,
        "primary_allowed": bool(primary_allowed),
        "strong_support": bool(all(strong_checks.values())),
        "nonnegative_macro_folds": int(nonnegative),
    }


def _locked_config(config: CrossfitTemporalConfig) -> None:
    expected = {
        "projection_dim": 64,
        "hidden_dim": 32,
        "dropout": 0.2,
        "study_loss_weight": 0.5,
        "focal_gamma": 1.0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "epochs": 20,
        "patience": 4,
        "study_batch_size": 8,
        "extraction_batch_size": 16,
        "maximum_pos_weight": 20.0,
    }
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"Exp89 locked hyperparameter changed: {key}")


def _train_fold(
    config: CrossfitTemporalConfig,
    spec: dict[str, Any],
    manifest: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer_fold = int(spec["outer_fold"])
    calibration_fold = int(spec["calibration_fold"])
    checkpoint = Path(str(spec["checkpoint"]))
    if file_sha256(checkpoint) != str(spec["sha256"]):
        raise RuntimeError(f"Fold {outer_fold} checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_config = payload.get("config", {})
    if (
        int(source_config.get("outer_fold", -1)) != outer_fold
        or int(source_config.get("calibration_fold", -1)) != calibration_fold
        or tuple(payload.get("output_labels", ())) != tuple(OUTPUT_LABELS)
    ):
        raise RuntimeError(f"Fold {outer_fold} checkpoint contract mismatch")

    training, calibration, outer = split_segmentation_slices(
        manifest, outer_fold=outer_fold, calibration_fold=calibration_fold
    )
    overlap = _patient_overlap(training, calibration, outer)
    if any(overlap.values()):
        raise RuntimeError(f"Fold {outer_fold} has patient leakage: {overlap}")
    model = build_segmentation_model(
        architecture=str(source_config["architecture"]),
        encoder_name=str(source_config["encoder_name"]),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
    ).to(device)
    load_segmentation_weights(model, checkpoint)
    model.requires_grad_(False).eval()
    checkpoint_sha = str(spec["sha256"])
    manifest_sha = file_sha256(config.manifest_path)
    identity = f"f{outer_fold}c{calibration_fold}_{checkpoint_sha[:12]}"
    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    train_cache = cache_root / f"train_{identity}.npz"
    calibration_cache = cache_root / f"calibration_{identity}.npz"
    train_meta = _ensure_feature_cache(
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
    calibration_meta = _ensure_feature_cache(
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
    _, calibration_truth, baseline_scores, identity_scores = _predict(
        temporal, calibration_loader, device=device
    )
    if not np.array_equal(baseline_scores, identity_scores):
        raise RuntimeError(f"Fold {outer_fold} epoch-zero identity failed")
    baseline_metrics = auc_summary(calibration_truth, baseline_scores)
    fold_dir = output / f"fold{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = fold_dir / "best_subtype_temporal_head.pth"
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "delta_selection_proxy": 0.0,
        "delta_macro_subtype_auc": 0.0,
        "macro_subtype_auc": baseline_metrics["macro_subtype_auc"],
    }]
    best_proxy = 0.0
    best_epoch = 0
    stale = 0
    torch.save(
        {
            "state_dict": temporal.state_dict(),
            "epoch": 0,
            "config": asdict(config),
            "fold_spec": spec,
            "feature_dim": feature_dim,
            "output_labels": OUTPUT_LABELS,
            "any_ich_logit_invariant": True,
        },
        checkpoint_path,
    )
    fold_started = time.perf_counter()
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
        _, observed_truth, _, scores = _predict(
            temporal, calibration_loader, device=device
        )
        if not np.array_equal(observed_truth, calibration_truth):
            raise RuntimeError(f"Fold {outer_fold} calibration order changed")
        candidate_metrics = auc_summary(calibration_truth, scores)
        delta = subtype_temporal_delta(baseline_metrics, candidate_metrics)
        if abs(float(delta["any_ich_auc"])) > 1e-12:
            raise RuntimeError(f"Fold {outer_fold} changed Any-ICH")
        proxy = float(delta["selection_proxy"])
        row = {
            "epoch": epoch,
            "delta_selection_proxy": proxy,
            "delta_macro_subtype_auc": delta["macro_subtype_auc"],
            "macro_subtype_auc": candidate_metrics["macro_subtype_auc"],
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in component_history.items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        print(json.dumps({"outer_fold": outer_fold, **row}, sort_keys=True), flush=True)
        if proxy > best_proxy + 1e-6:
            best_proxy = proxy
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": temporal.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "fold_spec": spec,
                    "feature_dim": feature_dim,
                    "output_labels": OUTPUT_LABELS,
                    "calibration_metrics": candidate_metrics,
                    "calibration_delta": delta,
                    "any_ich_logit_invariant": True,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= config.patience:
            break

    best_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    temporal.load_state_dict(best_payload["state_dict"], strict=True)
    _, _, _, calibration_candidate = _predict(
        temporal, calibration_loader, device=device
    )
    calibration_candidate_metrics = auc_summary(
        calibration_truth, calibration_candidate
    )
    calibration_delta = subtype_temporal_delta(
        baseline_metrics, calibration_candidate_metrics
    )
    calibration_gate = subtype_temporal_promotion_decision(calibration_delta)

    # Outer features and logits are touched only after the temporal checkpoint is fixed.
    outer_cache = cache_root / f"outer_{identity}.npz"
    outer_meta = _ensure_feature_cache(
        model,
        outer,
        outer_cache,
        checkpoint_sha256=checkpoint_sha,
        manifest_sha256=manifest_sha,
        split_name="outer",
        device=device,
        batch_size=config.extraction_batch_size,
        workers=config.workers,
    )
    outer_dataset = ICHSequenceFeatureDataset(outer_cache, truth)
    outer_loader = DataLoader(
        outer_dataset,
        batch_size=config.study_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ich_sequences,
    )
    study_ids, outer_truth, outer_baseline, outer_candidate = _predict(
        temporal, outer_loader, device=device
    )
    if not np.array_equal(outer_baseline[:, 0], outer_candidate[:, 0]):
        raise RuntimeError(f"Fold {outer_fold} outer Any scores changed")
    outer_baseline_metrics = auc_summary(outer_truth, outer_baseline)
    outer_candidate_metrics = auc_summary(outer_truth, outer_candidate)
    outer_delta = fold_auc_delta(
        outer_baseline_metrics, outer_candidate_metrics
    )
    patient_rows = outer.loc[:, ["study_id", "patient_id"]].drop_duplicates()
    if patient_rows["study_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Fold {outer_fold} study maps to multiple patients")
    patient_map = dict(
        zip(
            patient_rows["study_id"].astype(str),
            patient_rows["patient_id"].astype(str),
            strict=True,
        )
    )
    patient_ids = np.asarray([patient_map[study_id] for study_id in study_ids])
    summary = {
        "outer_fold": outer_fold,
        "calibration_fold": calibration_fold,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "patient_overlap": overlap,
        "training_slices": int(len(training)),
        "calibration_slices": int(len(calibration)),
        "outer_slices": int(len(outer)),
        "training_studies": int(len(train_dataset)),
        "calibration_studies": int(len(calibration_dataset)),
        "outer_studies": int(len(outer_dataset)),
        "best_epoch": int(best_epoch),
        "calibration_baseline": baseline_metrics,
        "calibration_candidate": calibration_candidate_metrics,
        "calibration_delta": calibration_delta,
        "calibration_gate": calibration_gate,
        "outer_baseline": outer_baseline_metrics,
        "outer_candidate": outer_candidate_metrics,
        "outer_delta": outer_delta,
        "temporal_checkpoint_sha256": file_sha256(checkpoint_path),
        "feature_cache": {
            "training": train_meta,
            "calibration": calibration_meta,
            "outer": outer_meta,
        },
        "outer_inferred_after_checkpoint_selection": True,
        "row_level_predictions_persisted": False,
        "duration_s": time.perf_counter() - fold_started,
    }
    _write_json(fold_dir / "summary.json", summary)
    del model, temporal
    torch.cuda.empty_cache()
    return (
        summary,
        study_ids,
        patient_ids,
        outer_truth,
        outer_baseline,
        outer_candidate,
    )


def run(config: CrossfitTemporalConfig) -> dict[str, Any]:
    validate_fold_specs(FOLD_SPECS)
    _locked_config(config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp89 requires CUDA BF16")
    output = Path(config.output_dir)
    if output.exists():
        existing = sorted(
            path.name
            for path in output.iterdir()
            if path.name != "PREREGISTERED_PLAN.md"
        )
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite Exp89 output: {output}; existing={existing}"
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "resolved_config.json",
        {**asdict(config), "fold_specs": FOLD_SPECS},
    )
    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    manifest = load_segmentation_manifest(config.manifest_path)
    truth, truth_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    started = time.perf_counter()
    fold_summaries: list[dict[str, Any]] = []
    all_study_ids: list[str] = []
    patient_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    for spec in FOLD_SPECS:
        result = _train_fold(
            config, spec, manifest, truth, output=output, device=device
        )
        fold_summaries.append(result[0])
        all_study_ids.extend(result[1])
        patient_parts.append(result[2])
        truth_parts.append(result[3])
        baseline_parts.append(result[4])
        candidate_parts.append(result[5])
        _write_json(output / "partial_fold_summaries.json", fold_summaries)

    patient_ids = np.concatenate(patient_parts)
    oof_truth = np.concatenate(truth_parts)
    baseline_scores = np.concatenate(baseline_parts)
    candidate_scores = np.concatenate(candidate_parts)
    expected_studies = set(manifest["study_id"].astype(str))
    observed_studies = set(all_study_ids)
    coverage_exact = (
        len(all_study_ids) == len(observed_studies) == len(expected_studies)
        and observed_studies == expected_studies
    )
    if not coverage_exact:
        raise RuntimeError("Exp89 outer folds do not cover every study exactly once")
    if not np.array_equal(baseline_scores[:, 0], candidate_scores[:, 0]):
        raise RuntimeError("Exp89 pooled Any scores are not exact")
    baseline = auc_summary(oof_truth, baseline_scores)
    candidate = auc_summary(oof_truth, candidate_scores)
    delta = subtype_temporal_delta(baseline, candidate)
    baseline_locked_match = (
        len(all_study_ids) == LOCKED_BASELINE["studies"]
        and len(np.unique(patient_ids)) == LOCKED_BASELINE["patients"]
        and abs(float(baseline["any_ich_auc"]) - LOCKED_BASELINE["any_ich_auc"])
        <= 1e-12
        and abs(
            float(baseline["macro_subtype_auc"])
            - LOCKED_BASELINE["macro_subtype_auc"]
        )
        <= 1e-12
    )
    if not baseline_locked_match:
        raise RuntimeError("Exp89 baseline does not reproduce the accepted OOF model")
    fold_deltas = [summary["outer_delta"] for summary in fold_summaries]
    bootstrap = paired_patient_auc_bootstrap(
        patient_ids,
        oof_truth,
        baseline_scores,
        candidate_scores,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    decision = oof_promotion_decision(
        delta,
        fold_deltas,
        bootstrap,
        any_logits_exact=True,
        baseline_locked_match=baseline_locked_match,
        coverage_exact=coverage_exact,
    )
    if decision["strong_support"]:
        disposition = "strong_development_oof_support_build_deployment_prototype"
    elif decision["primary_allowed"]:
        disposition = "provisional_development_oof_support_replicate_before_deployment"
    else:
        disposition = "reject_temporal_subtype_head_before_deployment"
    summary = {
        "schema_version": 1,
        "analysis_kind": "patient_disjoint_five_fold_any_invariant_temporal_subtype_oof",
        "experiment": "exp89_subtype_temporal_residual_oof_v1",
        "run_name": config.run_name,
        "decision": disposition,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "fold_summaries": fold_summaries,
        "paired_patient_bootstrap": bootstrap,
        "promotion_decision": decision,
        "coverage_exact": coverage_exact,
        "baseline_locked_match": baseline_locked_match,
        "any_ich_logits_exact_by_architecture": True,
        "spatial_volume_metrics_unchanged_by_design": True,
        "development_oof_not_final_unseen_test": True,
        "row_level_predictions_persisted": False,
        "external_reporting_enabled": False,
        "studies": len(all_study_ids),
        "patients": int(len(np.unique(patient_ids))),
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "manifest_sha256": file_sha256(config.manifest_path),
        "truth_source": str(truth_source),
        "git_commit": git_commit(),
    }
    _write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
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
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    config = CrossfitTemporalConfig(**vars(parser.parse_args()))
    run(config)


if __name__ == "__main__":
    main()
