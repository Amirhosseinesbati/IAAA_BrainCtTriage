"""One-shot outer-fold evaluation of a calibration-approved ICH temporal head."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from scripts.train_ich_temporal_residual_head import (
    _deltas,
    _ensure_feature_cache,
    _predict,
)
from src.strategies.ich_2p5d.segmentation_data import (
    load_segmentation_manifest,
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
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


OUTER_GATE = {
    "minimum_selection_proxy_delta": 0.001,
    "minimum_macro_subtype_auc_delta": 0.0,
    "minimum_any_ich_auc_delta": -0.002,
    "minimum_subtype_auc_delta": -0.02,
}


@dataclass(frozen=True)
class TemporalOuterConfig:
    run_name: str
    output_dir: str
    cache_dir: str
    manifest_path: str
    base_checkpoint: str
    temporal_checkpoint: str
    calibration_run_summary: str
    outer_baseline_summary: str
    outer_fold: int = 2
    calibration_fold: int = 1
    study_batch_size: int = 8
    extraction_batch_size: int = 16
    workers: int = 4


def temporal_outer_decision(delta: dict[str, object]) -> dict[str, object]:
    checks = {
        "selection_proxy": float(delta["selection_proxy"])
        >= OUTER_GATE["minimum_selection_proxy_delta"],
        "macro_subtype_auc": float(delta["macro_subtype_auc"])
        >= OUTER_GATE["minimum_macro_subtype_auc_delta"],
        "any_ich_auc_noninferiority": float(delta["any_ich_auc"])
        >= OUTER_GATE["minimum_any_ich_auc_delta"],
        "subtype_safety": min(
            float(value) for value in delta["subtype_auc"].values()
        )
        >= OUTER_GATE["minimum_subtype_auc_delta"],
    }
    return {
        "criteria": OUTER_GATE,
        "checks": checks,
        "expansion_allowed": bool(all(checks.values())),
    }


def _direct_summary(payload: dict[str, object]) -> dict[str, object]:
    if "selection_score" in payload:
        return payload
    rescored = payload.get("rescored_summary")
    if isinstance(rescored, dict) and "selection_score" in rescored:
        return rescored
    raise ValueError("Outer baseline summary does not contain selection_score")


def run(config: TemporalOuterConfig) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Temporal outer evaluation requires CUDA BF16 support")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "outer_evaluation_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to repeat consumed outer evaluation: {summary_path}")
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    calibration_summary = json.loads(
        Path(config.calibration_run_summary).read_text(encoding="utf-8")
    )
    if not calibration_summary.get("promotion_decision", {}).get(
        "promotion_allowed", False
    ):
        raise ValueError("Temporal head did not pass its calibration promotion gate")
    if calibration_summary.get("outer_evaluation_performed") is not False:
        raise ValueError("Calibration run did not preserve the outer-fold lock")

    head_payload = torch.load(
        Path(config.temporal_checkpoint), map_location="cpu", weights_only=True
    )
    base_sha = file_sha256(config.base_checkpoint)
    head_sha = file_sha256(config.temporal_checkpoint)
    manifest_sha = file_sha256(config.manifest_path)
    if head_payload.get("base_checkpoint_sha256") != base_sha:
        raise ValueError("Temporal head was not trained on the requested base checkpoint")
    if head_payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("Temporal head manifest does not match outer evaluation")
    head_config = head_payload.get("config", {})
    if (
        int(head_config.get("outer_fold", -1)) != config.outer_fold
        or int(head_config.get("calibration_fold", -1)) != config.calibration_fold
    ):
        raise ValueError("Temporal head does not match requested held-out folds")
    if int(head_payload.get("epoch", -1)) != int(calibration_summary["best_epoch"]):
        raise ValueError("Temporal checkpoint epoch does not match calibration selection")

    base_payload = torch.load(
        Path(config.base_checkpoint), map_location="cpu", weights_only=True
    )
    source_config = base_payload.get("config", {})
    if (
        int(source_config.get("outer_fold", -1)) != config.outer_fold
        or int(source_config.get("calibration_fold", -1)) != config.calibration_fold
    ):
        raise ValueError("Base checkpoint does not match requested held-out folds")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    base_model = build_segmentation_model(
        architecture=str(source_config.get("architecture", "unetplusplus")),
        encoder_name=str(source_config.get("encoder_name", "efficientnet-b2")),
        pretrained=False,
        dropout=float(source_config.get("dropout", 0.2)),
    ).to(device)
    load_segmentation_weights(base_model, config.base_checkpoint)
    base_model.requires_grad_(False).eval()

    manifest = load_segmentation_manifest(config.manifest_path)
    _, _, outer = split_segmentation_slices(
        manifest,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
    )
    truth, truth_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = f"f{config.outer_fold}c{config.calibration_fold}_{base_sha[:12]}"
    outer_cache = cache_root / f"outer_{identity}.npz"
    outer_cache_meta = _ensure_feature_cache(
        base_model,
        outer,
        outer_cache,
        checkpoint_sha256=base_sha,
        manifest_sha256=manifest_sha,
        split_name="outer_one_shot",
        device=device,
        batch_size=config.extraction_batch_size,
        workers=config.workers,
    )
    del base_model
    torch.cuda.empty_cache()

    dataset = ICHSequenceFeatureDataset(outer_cache, truth)
    loader = DataLoader(
        dataset,
        batch_size=config.study_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ich_sequences,
    )
    temporal = TemporalResidualHead(
        int(head_payload["feature_dim"]),
        projection_dim=int(head_config["projection_dim"]),
        hidden_dim=int(head_config["hidden_dim"]),
        dropout=float(head_config["dropout"]),
    ).to(device)
    temporal.load_state_dict(head_payload["state_dict"], strict=True)
    temporal.eval()
    study_ids, observed_truth, baseline_scores, candidate_scores = _predict(
        temporal, loader, device=device
    )
    baseline = auc_summary(observed_truth, baseline_scores)
    candidate = auc_summary(observed_truth, candidate_scores)
    delta = _deltas(baseline, candidate)

    locked_payload = json.loads(
        Path(config.outer_baseline_summary).read_text(encoding="utf-8")
    )
    locked = _direct_summary(locked_payload)
    observed_any = float(baseline["any_ich_auc"])
    expected_any = float(locked["any_ich_study_auc"])
    observed_macro = float(baseline["macro_subtype_auc"])
    expected_macro = float(locked["macro_subtype_study_auc"])
    if (
        abs(observed_any - expected_any) > 1e-12
        or abs(observed_macro - expected_macro) > 1e-12
    ):
        raise RuntimeError(
            "Outer feature baseline does not match locked base evaluation: "
            f"Any {observed_any:.12f} vs {expected_any:.12f}; "
            f"macro {observed_macro:.12f} vs {expected_macro:.12f}"
        )
    decision = temporal_outer_decision(delta)
    baseline_selection = float(locked["selection_score"])
    result: dict[str, object] = {
        "analysis_kind": "ich_temporal_residual_one_shot_outer_evaluation",
        "run_name": config.run_name,
        "calibration_parent_run_id": calibration_summary["run_id"],
        "calibration_best_epoch": calibration_summary["best_epoch"],
        "studies": len(dataset),
        "study_ids_persisted": False,
        "outer_fold": config.outer_fold,
        "calibration_fold": config.calibration_fold,
        "base_checkpoint_sha256": base_sha,
        "temporal_checkpoint_sha256": head_sha,
        "manifest_sha256": manifest_sha,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "baseline_selection_score_unchanged_spatial": baseline_selection,
        "candidate_selection_proxy_unchanged_spatial": baseline_selection
        + float(delta["selection_proxy"]),
        "outer_gate": decision,
        "outer_evaluation_consumed": True,
        "training_or_selection_on_outer": False,
        "spatial_volume_metrics_unchanged_by_design": True,
        "feature_cache": {
            **outer_cache_meta,
            "persisted_in_git": False,
        },
        "truth_source": str(truth_source),
        "git_commit": git_commit(),
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    with mlflow.start_run(run_name=config.run_name) as mlflow_run:
        mlflow.set_tags({
            "task": "ich_segmentation_volume",
            "stage": "temporal_residual_one_shot_outer_evaluation",
            "outer_fold_consumed": str(config.outer_fold),
            "expansion_allowed": str(decision["expansion_allowed"]).lower(),
            "parent_run_id": calibration_summary["run_id"],
        })
        mlflow.log_params({
            **asdict(config),
            "base_checkpoint_sha256": base_sha,
            "temporal_checkpoint_sha256": head_sha,
            "manifest_sha256": manifest_sha,
            "outer_cache_sha256": outer_cache_meta["cache_sha256"],
            "git_commit": result["git_commit"],
        })
        mlflow.log_metrics({
            "baseline_any_ich_auc": float(baseline["any_ich_auc"]),
            "candidate_any_ich_auc": float(candidate["any_ich_auc"]),
            "delta_any_ich_auc": float(delta["any_ich_auc"]),
            "baseline_macro_subtype_auc": float(baseline["macro_subtype_auc"]),
            "candidate_macro_subtype_auc": float(candidate["macro_subtype_auc"]),
            "delta_macro_subtype_auc": float(delta["macro_subtype_auc"]),
            "delta_selection_proxy": float(delta["selection_proxy"]),
            "expansion_allowed": float(decision["expansion_allowed"]),
        })
        mlflow.log_artifacts(str(output), artifact_path="ich_temporal_outer_evaluation")
        result["mlflow_run_id"] = mlflow_run.info.run_id
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    event = "success" if decision["expansion_allowed"] else "warning"
    notify_campaign(
        event,
        f"ارزیابی یک‌باره outer{config.outer_fold} برای temporal head ICH کامل شد. "
        f"delta selection={float(delta['selection_proxy']):+.5f}، Any-AUC="
        f"{float(delta['any_ich_auc']):+.5f} و macro-AUC="
        f"{float(delta['macro_subtype_auc']):+.5f} است؛ expansion="
        f"{decision['expansion_allowed']}. تحلیل کوتاه: checkpoint فقط روی calibration "
        "انتخاب شده و outer هیچ training یا تنظیمی نداده است؛ mask/حجم ثابت‌اند. "
        "اقدام بعدی: در صورت عبور، اجرای همان recipe روی چهار split دیگر؛ در غیر این "
        "صورت بستن exp53 بدون اصلاح پس از outer.",
        run=config.run_name,
        kind="outer_one_shot",
        fold=f"outer={config.outer_fold}",
        detail=f"MLflow {result['mlflow_run_id']}; studies={len(study_ids)}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--calibration-run-summary", required=True)
    parser.add_argument("--outer-baseline-summary", required=True)
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--study-batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    print(json.dumps(run(TemporalOuterConfig(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
