"""Measure whether a background-to-SAH adapter can cross incumbent logit margins.

The diagnostic evaluates only the locked calibration fold and persists aggregate
counts/quantiles.  It never evaluates the outer fold or logs row-level medical
predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


BACKGROUND_CLASS_ID = 0
IPH_CLASS_ID = 2
SAH_CLASS_ID = 5
CAPS = (2.0, 4.0, 6.0, 7.5, 8.0, 12.0, 16.0)
QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    return {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in QUANTILES
    }


def reachability_summary(
    margins: np.ndarray,
    *,
    caps: tuple[float, ...] = CAPS,
) -> dict[str, dict[str, float | int]]:
    margins = np.asarray(margins, dtype=np.float64)
    margins = margins[np.isfinite(margins)]
    result: dict[str, dict[str, float | int]] = {}
    for cap in caps:
        reachable = int(np.sum(margins < cap))
        result[str(cap)] = {
            "pixels": reachable,
            "fraction": float(reachable / len(margins)) if len(margins) else 0.0,
        }
    return result


def diagnostic_interpretation(
    *, eligible_fraction: float, reachable_fraction_at_8: float
) -> str:
    if eligible_fraction < 0.25:
        return "support_limited_most_sah_pixels_are_not_incumbent_background"
    if reachable_fraction_at_8 < 0.25:
        return "cap_limited_most_eligible_sah_pixels_need_more_than_8_logits"
    return "optimization_or_representation_limited_cap8_can_reach_many_pixels"


def iph_relabel_interpretation(
    *,
    sah_predicted_iph_fraction: float,
    missed_sah_reachable_at_12: float,
    correct_iph_vulnerable_at_12: float,
) -> str:
    if sah_predicted_iph_fraction < 0.15:
        return "iph_support_adds_too_few_missed_sah_pixels"
    if missed_sah_reachable_at_12 < 0.25:
        return "iph_to_sah_margins_are_mostly_beyond_cap12"
    if correct_iph_vulnerable_at_12 > 0.50:
        return "iph_support_has_high_theoretical_true_iph_relabel_risk"
    return "iph_support_is_material_but_requires_selective_negative_control"


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    config = payload["config"]
    if int(config.get("outer_fold", -1)) != args.outer_fold:
        raise ValueError("Checkpoint outer fold does not match diagnostic split")
    if int(config.get("calibration_fold", -1)) != args.calibration_fold:
        raise ValueError("Checkpoint calibration fold does not match diagnostic split")
    if config.get("architecture") != args.architecture:
        raise ValueError("Checkpoint architecture does not match")
    if config.get("encoder_name") != args.encoder_name:
        raise ValueError("Checkpoint encoder does not match")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SAH margin diagnostic requires BF16-capable CUDA")

    _, calibration_loader, _, _, calibration_frame, _ = create_segmentation_loaders(
        args.manifest,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        context_radius=1,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=False,
    ).to(device).eval()
    load_segmentation_weights(model, args.checkpoint)

    missed_margins: list[np.ndarray] = []
    missed_sah_probabilities: list[np.ndarray] = []
    sah_predicted_iph_margins: list[np.ndarray] = []
    correct_iph_margins: list[np.ndarray] = []
    true_sah_pixels = 0
    predicted_class_counts = {str(class_id): 0 for class_id in range(6)}
    true_background_incumbent_background_pixels = 0
    true_background_vulnerable = {str(cap): 0 for cap in CAPS}
    sah_studies: set[str] = set()
    reachable_sah_studies = {str(cap): set() for cap in CAPS}
    reachable_iph_confused_sah_studies = {str(cap): set() for cap in CAPS}

    with torch.inference_mode():
        for batch in calibration_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            known = batch["segmentation_known"].to(device, non_blocking=True) > 0.5
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_logits, _ = model(images)
            logits = mask_logits.float()
            predicted = logits.argmax(dim=1)
            margin = logits[:, BACKGROUND_CLASS_ID] - logits[:, SAH_CLASS_ID]
            iph_margin = logits[:, IPH_CLASS_ID] - logits[:, SAH_CLASS_ID]
            sah_probability = torch.softmax(logits, dim=1)[:, SAH_CLASS_ID]
            for index, study_id in enumerate(batch["study_id"]):
                if not bool(known[index]):
                    continue
                true_sah = masks[index] == SAH_CLASS_ID
                if torch.any(true_sah):
                    study = str(study_id)
                    sah_studies.add(study)
                    true_sah_pixels += int(true_sah.sum())
                    for class_id in range(6):
                        predicted_class_counts[str(class_id)] += int(
                            (true_sah & (predicted[index] == class_id)).sum()
                        )
                    missed = true_sah & (predicted[index] == BACKGROUND_CLASS_ID)
                    if torch.any(missed):
                        study_margins = margin[index][missed].detach().cpu().numpy()
                        missed_margins.append(study_margins)
                        missed_sah_probabilities.append(
                            sah_probability[index][missed].detach().cpu().numpy()
                        )
                        for cap in CAPS:
                            if np.any(study_margins < cap):
                                reachable_sah_studies[str(cap)].add(study)
                    predicted_iph = true_sah & (predicted[index] == IPH_CLASS_ID)
                    if torch.any(predicted_iph):
                        study_iph_margins = (
                            iph_margin[index][predicted_iph].detach().cpu().numpy()
                        )
                        sah_predicted_iph_margins.append(study_iph_margins)
                        for cap in CAPS:
                            if np.any(study_iph_margins < cap):
                                reachable_iph_confused_sah_studies[str(cap)].add(study)

                true_iph = masks[index] == IPH_CLASS_ID
                correct_iph = true_iph & (predicted[index] == IPH_CLASS_ID)
                if torch.any(correct_iph):
                    correct_iph_margins.append(
                        iph_margin[index][correct_iph].detach().cpu().numpy()
                    )

                true_background = masks[index] == BACKGROUND_CLASS_ID
                incumbent_background = true_background & (
                    predicted[index] == BACKGROUND_CLASS_ID
                )
                count = int(incumbent_background.sum())
                true_background_incumbent_background_pixels += count
                if count:
                    background_margins = margin[index][incumbent_background]
                    for cap in CAPS:
                        true_background_vulnerable[str(cap)] += int(
                            (background_margins < cap).sum()
                        )

    missed = (
        np.concatenate(missed_margins).astype(np.float64, copy=False)
        if missed_margins
        else np.empty(0, dtype=np.float64)
    )
    missed_probabilities = (
        np.concatenate(missed_sah_probabilities).astype(np.float64, copy=False)
        if missed_sah_probabilities
        else np.empty(0, dtype=np.float64)
    )
    sah_iph = (
        np.concatenate(sah_predicted_iph_margins).astype(np.float64, copy=False)
        if sah_predicted_iph_margins
        else np.empty(0, dtype=np.float64)
    )
    correct_iph = (
        np.concatenate(correct_iph_margins).astype(np.float64, copy=False)
        if correct_iph_margins
        else np.empty(0, dtype=np.float64)
    )
    if not len(correct_iph):
        raise ValueError("Calibration fold has no correctly predicted true-IPH pixels")
    reachable = reachability_summary(missed)
    for cap in CAPS:
        key = str(cap)
        reachable[key]["positive_studies_with_any_reachable_pixel"] = len(
            reachable_sah_studies[key]
        )
        vulnerable = true_background_vulnerable[key]
        reachable[key]["true_background_vulnerable_pixels"] = vulnerable
        reachable[key]["true_background_vulnerable_fraction"] = (
            float(vulnerable / true_background_incumbent_background_pixels)
            if true_background_incumbent_background_pixels
            else 0.0
        )
    missed_count = int(predicted_class_counts[str(BACKGROUND_CLASS_ID)])
    if true_sah_pixels == 0:
        raise ValueError("Calibration fold contains no spatially supervised SAH pixels")
    if missed_count == 0:
        raise ValueError("Incumbent has no background-predicted SAH pixels to diagnose")
    eligible_fraction = float(missed_count / true_sah_pixels) if true_sah_pixels else 0.0
    reachable_at_8 = float(reachable["8.0"]["fraction"])
    sah_iph_reachability = reachability_summary(sah_iph)
    correct_iph_vulnerability = reachability_summary(correct_iph)
    for cap in CAPS:
        key = str(cap)
        sah_iph_reachability[key][
            "positive_studies_with_any_reachable_pixel"
        ] = len(reachable_iph_confused_sah_studies[key])
    sah_predicted_iph_fraction = float(len(sah_iph) / true_sah_pixels)
    sah_iph_reachable_at_12 = float(sah_iph_reachability["12.0"]["fraction"])
    correct_iph_vulnerable_at_12 = float(
        correct_iph_vulnerability["12.0"]["fraction"]
    )
    result = {
        "analysis_kind": "calibration_only_sah_support_margin_diagnostic",
        "decision": diagnostic_interpretation(
            eligible_fraction=eligible_fraction,
            reachable_fraction_at_8=reachable_at_8,
        ),
        "iph_relabel_decision": iph_relabel_interpretation(
            sah_predicted_iph_fraction=sah_predicted_iph_fraction,
            missed_sah_reachable_at_12=sah_iph_reachable_at_12,
            correct_iph_vulnerable_at_12=correct_iph_vulnerable_at_12,
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "outer_fold_reserved": args.outer_fold,
        "calibration_fold": args.calibration_fold,
        "outer_evaluation_performed": False,
        "calibration_studies": int(calibration_frame["study_id"].nunique()),
        "spatially_known_calibration_slices": int(
            calibration_frame["segmentation_known"].sum()
        ),
        "sah_positive_studies": len(sah_studies),
        "true_sah_pixels": true_sah_pixels,
        "true_sah_predicted_class_counts": predicted_class_counts,
        "eligible_background_missed_sah_pixels": missed_count,
        "eligible_background_fraction_of_true_sah": eligible_fraction,
        "missed_background_minus_sah_logit_margin_quantiles": finite_quantiles(
            missed
        ),
        "missed_sah_probability_quantiles": finite_quantiles(
            missed_probabilities
        ),
        "reachability_by_maximum_logit_residual": reachable,
        "sah_predicted_iph_pixels": int(len(sah_iph)),
        "sah_predicted_iph_fraction_of_true_sah": sah_predicted_iph_fraction,
        "sah_predicted_iph_minus_sah_margin_quantiles": finite_quantiles(sah_iph),
        "sah_predicted_iph_reachability_by_maximum_logit_residual": (
            sah_iph_reachability
        ),
        "correct_true_iph_pixels": int(len(correct_iph)),
        "correct_iph_minus_sah_margin_quantiles": finite_quantiles(correct_iph),
        "correct_iph_theoretical_vulnerability_by_maximum_logit_residual": (
            correct_iph_vulnerability
        ),
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "sah_background_margin_diagnostic",
                "evaluation_scope": "calibration_only_no_outer",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "outer_fold_reserved": args.outer_fold,
                "calibration_fold": args.calibration_fold,
                "caps": json.dumps(CAPS),
            }
        )
        mlflow.log_metrics(
            {
                "eligible_background_fraction_of_true_sah": eligible_fraction,
                "reachable_fraction_at_cap8": reachable_at_8,
                "sah_predicted_iph_fraction": sah_predicted_iph_fraction,
                "sah_iph_reachable_at_cap12": sah_iph_reachable_at_12,
                "correct_iph_vulnerable_at_cap12": correct_iph_vulnerable_at_12,
                "true_sah_pixels": true_sah_pixels,
                "missed_sah_pixels": missed_count,
                "margin_median": result[
                    "missed_background_minus_sah_logit_margin_quantiles"
                ]["q50"],
                "margin_q90": result[
                    "missed_background_minus_sah_logit_margin_quantiles"
                ]["q90"],
                "duration_s": result["duration_s"],
                "peak_vram_gb": result["peak_vram_gb"],
            }
        )
        mlflow.log_artifact(str(args.output), artifact_path="ich_diagnostics")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.notify:
        notify_campaign(
            "info",
            "🔬 تحلیل margin مدل خونریزی مسابقه IAAA کامل شد. تحلیل کوتاه: این "
            "آزمایش فرصت بازیابی SAH از background/IPH را در برابر خطر relabel "
            "کردن true-IPH می‌سنجد؛ outer دست‌نخورده است و توسعهٔ support فقط با "
            "نسبت فرصت/خطر قابل‌دفاع ادامه می‌یابد.",
            experiment="sah_support_margin_diagnostic",
            decision=result["decision"],
            iph_relabel_decision=result["iph_relabel_decision"],
            sah_positive_studies=len(sah_studies),
            eligible_background_fraction=f"{eligible_fraction:.3f}",
            reachable_at_cap8=f"{reachable_at_8:.3f}",
            margin_median=f"{result['missed_background_minus_sah_logit_margin_quantiles']['q50']:.3f}",
            margin_q90=f"{result['missed_background_minus_sah_logit_margin_quantiles']['q90']:.3f}",
            sah_iph_reachable_cap12=f"{sah_iph_reachable_at_12:.3f}",
            correct_iph_vulnerable_cap12=f"{correct_iph_vulnerable_at_12:.3f}",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--architecture", default="unetplusplus")
    parser.add_argument("--encoder-name", default="tu-efficientnetv2_rw_s")
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    result = run_diagnostic(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
