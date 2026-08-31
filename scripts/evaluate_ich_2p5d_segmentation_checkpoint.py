"""Recover or repeat outer evaluation for a trained 2.5D ICH checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import torch

from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import (
    _flatten_summary_metrics,
    _predict_slices,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


def checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the minimum deployment/evaluation configuration."""
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("ICH checkpoint does not contain a training config")
    required = {
        "architecture",
        "encoder_name",
        "outer_fold",
        "calibration_fold",
        "batch_size",
        "workers",
        "dropout",
        "seed",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"ICH checkpoint config is missing: {sorted(missing)}")
    if int(config["outer_fold"]) == int(config["calibration_fold"]):
        raise ValueError("Checkpoint outer and calibration folds must differ")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ICH segmentation checkpoint evaluation requires CUDA BF16")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("ICH checkpoint must be a dictionary")
    config = checkpoint_config(payload)
    manifest_path = args.manifest_path or Path(
        str(config.get("manifest_path", "Data/processed/ich_2p5d/slice_manifest.csv"))
    )
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_summary_path = output_dir / "outer_summary.json"
    if outer_summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing outer evaluation: {outer_summary_path}"
        )
    run_name = args.run_name or f"{config.get('run_name', args.checkpoint.stem)}-recovery-eval"
    workers = int(args.workers if args.workers is not None else config["workers"])

    notify_campaign(
        "start",
        "ارزیابی recovery برای checkpoint مدل مستقیم ICH آغاز شد. تحلیل کوتاه: آموزش تکرار نمی‌شود و فقط بهترین checkpoint ذخیره‌شده روی outer fold از پیش تعیین‌شده سنجیده خواهد شد. اقدام بعدی: بازسازی Dice، AUC، FPR و خطای حجم و ثبت مستقل در MLflow.",
        run=run_name,
        kind="checkpoint_recovery_evaluation",
        fold=f"outer={config['outer_fold']}, calibration={config['calibration_fold']}",
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        _, _, outer_loader, _, _, outer_frame = create_segmentation_loaders(
            manifest_path,
            outer_fold=int(config["outer_fold"]),
            calibration_fold=int(config["calibration_fold"]),
            batch_size=int(config["batch_size"]),
            workers=workers,
            seed=int(config["seed"]),
        )
        truth, metadata_source = ground_truth_ich_context()
        truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
        model = build_segmentation_model(
            architecture=str(config["architecture"]),
            encoder_name=str(config["encoder_name"]),
            pretrained=False,
            dropout=float(config["dropout"]),
        ).to(device)
        load_segmentation_weights(model, args.checkpoint)
        outer_slices = _predict_slices(model, outer_loader, device=device)
        outer_studies, outer_summary = summarize_segmentation_predictions(
            outer_slices, truth
        )
        outer_slices.to_csv(output_dir / "outer_slice_predictions.csv", index=False)
        outer_studies.to_csv(output_dir / "outer_study_predictions.csv", index=False)
        outer_summary_path.write_text(
            json.dumps(outer_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        duration = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        evaluation_summary = {
            "run_name": run_name,
            "evaluation_kind": "checkpoint_recovery_outer_fold",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_git_commit": payload.get("git_commit"),
            "evaluator_git_commit": git_commit(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "metadata_source": str(metadata_source),
            "outer_fold": int(config["outer_fold"]),
            "calibration_fold": int(config["calibration_fold"]),
            "outer_studies": int(outer_frame["study_id"].nunique()),
            "duration_s": duration,
            "peak_vram_gb": peak_vram,
            "outer_summary": outer_summary,
            "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        }
        evaluation_path = output_dir / "recovery_evaluation_summary.json"
        evaluation_path.write_text(
            json.dumps(evaluation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        configure_remote_mlflow()
        mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "checkpoint_recovery_outer_evaluation",
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
                "outer_fold_policy": "checkpoint_selected_before_outer_evaluation",
            })
            mlflow.log_params({
                "checkpoint_sha256": evaluation_summary["checkpoint_sha256"],
                "checkpoint_epoch": evaluation_summary["checkpoint_epoch"],
                "checkpoint_git_commit": evaluation_summary["checkpoint_git_commit"],
                "evaluator_git_commit": evaluation_summary["evaluator_git_commit"],
                "manifest_sha256": evaluation_summary["manifest_sha256"],
                "outer_fold": evaluation_summary["outer_fold"],
                "calibration_fold": evaluation_summary["calibration_fold"],
                "outer_studies": evaluation_summary["outer_studies"],
            })
            mlflow.log_metrics({
                **_flatten_summary_metrics("outer", outer_summary),
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
            })
            mlflow.log_artifact(str(outer_summary_path), artifact_path="recovery_evaluation")
            mlflow.log_artifact(str(evaluation_path), artifact_path="recovery_evaluation")
            mlflow.log_artifact(
                str(output_dir / "outer_study_predictions.csv"),
                artifact_path="recovery_evaluation",
            )
            evaluation_summary["mlflow_run_id"] = run.info.run_id
        evaluation_path.write_text(
            json.dumps(evaluation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        notify_campaign(
            "success",
            "ارزیابی recovery مدل مستقیم ICH کامل شد. تحلیل کوتاه: outer fold فقط پس از بارگذاری checkpoint از پیش انتخاب‌شده دیده شد و آموزش تکرار نشد؛ نتیجه برای مقایسهٔ کنترل‌شده معتبر است. اقدام بعدی: مقایسه با reference و pixel-weighted و تصمیم برای تأیید روی fold مستقل.",
            run=run_name,
            kind="checkpoint_recovery_evaluation",
            dice=f"{float(outer_summary['mean_foreground_dice']):.4f}",
            any_auc=f"{float(outer_summary['any_ich_study_auc'] or 0):.4f}",
            normal_fpr=f"{float(outer_summary['normal_false_positive_rate_at_0_1ml']):.4f}",
            total_mae_ml=f"{float(outer_summary['total_volume_mae_ml']):.3f}",
        )
        print(json.dumps(evaluation_summary, indent=2, sort_keys=True))
    except Exception as exc:
        notify_campaign(
            "failure",
            "ارزیابی recovery مدل مستقیم ICH متوقف شد. تحلیل کوتاه: checkpoint آموزش‌دیده دست‌نخورده است و این خطا نتیجهٔ کیفیتی محسوب نمی‌شود. اقدام بعدی: رفع کوچک‌ترین علت فنی و تکرار فقط ارزیابی، نه آموزش.",
            run=run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise


if __name__ == "__main__":
    main()
