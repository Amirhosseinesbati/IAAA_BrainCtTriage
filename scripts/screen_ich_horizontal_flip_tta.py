"""Calibration-only screen of horizontal-flip TTA for an ICH checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import torch

from scripts.evaluate_ich_2p5d_segmentation_checkpoint import checkpoint_config
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
    checkpoint_selection_score,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


PRIMARY_METRICS = (
    "selection_score",
    "mean_foreground_dice",
    "any_ich_study_auc",
    "macro_subtype_study_auc",
    "normal_false_positive_rate_at_0_1ml",
    "presence_f1_at_0_1ml",
    "total_volume_mae_ml",
)


def tta_screen_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Apply preregistered, calibration-only non-regression gates."""
    baseline_checkpoint_score = checkpoint_selection_score(
        baseline, "fpr_penalized"
    )
    candidate_checkpoint_score = checkpoint_selection_score(
        candidate, "fpr_penalized"
    )
    gates = {
        "checkpoint_score_improves": (
            candidate_checkpoint_score > baseline_checkpoint_score + 1e-6
        ),
        "dice_noninferior_0_002": (
            float(candidate["mean_foreground_dice"])
            >= float(baseline["mean_foreground_dice"]) - 0.002
        ),
        "any_auc_noninferior_0_005": (
            float(candidate["any_ich_study_auc"] or 0.0)
            >= float(baseline["any_ich_study_auc"] or 0.0) - 0.005
        ),
        "macro_auc_noninferior_0_005": (
            float(candidate["macro_subtype_study_auc"])
            >= float(baseline["macro_subtype_study_auc"]) - 0.005
        ),
        "fpr_not_worse": (
            float(candidate["normal_false_positive_rate_at_0_1ml"])
            <= float(baseline["normal_false_positive_rate_at_0_1ml"]) + 1e-12
        ),
        "presence_f1_noninferior_0_005": (
            float(candidate["presence_f1_at_0_1ml"])
            >= float(baseline["presence_f1_at_0_1ml"]) - 0.005
        ),
        "mae_noninferior_0_1ml": (
            float(candidate["total_volume_mae_ml"])
            <= float(baseline["total_volume_mae_ml"]) + 0.1
        ),
    }
    return {
        "decision": "advance_to_oof" if all(gates.values()) else "reject_before_outer",
        "gates": gates,
        "baseline_checkpoint_score": baseline_checkpoint_score,
        "candidate_checkpoint_score": candidate_checkpoint_score,
        "candidate_minus_baseline": {
            name: float(candidate[name]) - float(baseline[name])
            for name in PRIMARY_METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Horizontal-flip TTA screening requires CUDA BF16")
    decision_path = args.output_dir / "tta_screen.json"
    if decision_path.exists():
        raise FileExistsError(f"Refusing to overwrite TTA screen: {decision_path}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("ICH checkpoint must be a dictionary")
    config = checkpoint_config(payload)
    workers = int(args.workers if args.workers is not None else config["workers"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    try:
        _, calibration_loader, _, _, calibration_frame, _ = (
            create_segmentation_loaders(
                args.manifest_path,
                outer_fold=int(config["outer_fold"]),
                calibration_fold=int(config["calibration_fold"]),
                batch_size=int(config["batch_size"]),
                workers=workers,
                seed=int(config["seed"]),
                context_radius=int(config.get("slice_context_radius", 1)),
            )
        )
        truth, metadata_source = ground_truth_ich_context()
        truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
        model = build_segmentation_model(
            architecture=str(config["architecture"]),
            encoder_name=str(config["encoder_name"]),
            pretrained=False,
            dropout=float(config["dropout"]),
            horizontal_symmetry_adapter=bool(
                config.get("horizontal_symmetry_adapter", False)
            ),
            five_slice_context_adapter=bool(
                config.get("five_slice_context_adapter", False)
            ),
        ).to(device)
        load_segmentation_weights(model, args.checkpoint)

        baseline_slices = _predict_slices(
            model, calibration_loader, device=device
        )
        baseline_studies, baseline_summary = summarize_segmentation_predictions(
            baseline_slices, truth
        )
        tta_slices = _predict_slices(
            model,
            calibration_loader,
            device=device,
            horizontal_flip_tta=True,
        )
        tta_studies, tta_summary = summarize_segmentation_predictions(
            tta_slices, truth
        )
        if not baseline_slices[["study_id", "slice_index"]].equals(
            tta_slices[["study_id", "slice_index"]]
        ):
            raise ValueError("Baseline and TTA slice alignment changed")

        decision = tta_screen_decision(baseline_summary, tta_summary)
        duration = time.perf_counter() - started
        result = {
            "analysis_kind": "calibration_only_horizontal_flip_tta_screen",
            "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
            "outer_evaluation_performed": False,
            "run_name": args.run_name,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "manifest_path": str(args.manifest_path),
            "manifest_sha256": file_sha256(args.manifest_path),
            "metadata_source": str(metadata_source),
            "outer_fold_reserved": int(config["outer_fold"]),
            "calibration_fold": int(config["calibration_fold"]),
            "calibration_studies": int(calibration_frame["study_id"].nunique()),
            "tta": "horizontal_flip_probability_average",
            "duration_s": duration,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "baseline_summary": baseline_summary,
            "tta_summary": tta_summary,
            **decision,
        }
        baseline_slices.to_csv(
            args.output_dir / "baseline_calibration_slice_predictions.csv",
            index=False,
        )
        baseline_studies.to_csv(
            args.output_dir / "baseline_calibration_study_predictions.csv",
            index=False,
        )
        tta_slices.to_csv(
            args.output_dir / "tta_calibration_slice_predictions.csv", index=False
        )
        tta_studies.to_csv(
            args.output_dir / "tta_calibration_study_predictions.csv", index=False
        )
        decision_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

        configure_remote_mlflow()
        mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "calibration_only_tta_screen",
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
                "outer_fold_policy": "not_evaluated",
                "decision": result["decision"],
            })
            mlflow.log_params({
                "checkpoint_sha256": result["checkpoint_sha256"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "manifest_sha256": result["manifest_sha256"],
                "evaluator_git_commit": git_commit(),
                "outer_fold_reserved": result["outer_fold_reserved"],
                "calibration_fold": result["calibration_fold"],
                "tta": result["tta"],
            })
            mlflow.log_metrics({
                **_flatten_summary_metrics("baseline", baseline_summary),
                **_flatten_summary_metrics("tta", tta_summary),
                "duration_s": duration,
                "peak_vram_gb": result["peak_vram_gb"],
            })
            mlflow.log_artifact(str(decision_path), artifact_path="tta_screen")
            result["mlflow_run_id"] = run.info.run_id
        decision_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

        delta = result["candidate_minus_baseline"]
        notify_campaign(
            "success" if result["decision"] == "advance_to_oof" else "info",
            (
                "غربال TTA افقی کامل شد. تحلیل کوتاه: این راهبرد وزن مدل را تغییر "
                "نمی‌دهد و فقط در صورت بهبود امتیاز checkpoint همراه با حفظ Dice، "
                "FPR، F1 و MAE اجازهٔ ادامه به OOF می‌گیرد."
            ),
            run=args.run_name,
            decision=result["decision"],
            delta_selection=f"{delta['selection_score']:+.5f}",
            delta_dice=f"{delta['mean_foreground_dice']:+.5f}",
            delta_fpr=f"{delta['normal_false_positive_rate_at_0_1ml']:+.5f}",
            delta_mae_ml=f"{delta['total_volume_mae_ml']:+.3f}",
            next_step=(
                "cross-fitted OOF TTA"
                if result["decision"] == "advance_to_oof"
                else "رد TTA و حفظ inference فعلی"
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        notify_campaign(
            "failure",
            "غربال TTA افقی به خطای فنی خورد؛ checkpoint و outer دست‌نخورده‌اند.",
            run=args.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise


if __name__ == "__main__":
    main()
