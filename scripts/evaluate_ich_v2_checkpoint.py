"""Evaluate an ICH SegResNet with correct physical-volume accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from src.mlops.telegram_notifier import TelegramNotifier, format_notification
from src.strategies.ich_v2.evaluation import (
    VOLUME_KEYS,
    ground_truth_ich_context,
    write_evaluation,
)
from src.strategies.ich_v2.geometry import volumes_from_labelmap
from src.strategies.ich_v2.geometry import remove_small_components


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notify(event: str, message: str, **fields: object) -> None:
    try:
        notifier = TelegramNotifier.from_environment(Path(".env"))
        notifier.send_text(format_notification(
            event,
            message,
            title="[مسابقه IAAA Brain CT Triage 2026 | تسک ICH]",
            fields={key: str(value) for key, value in fields.items()},
        ))
    except Exception as exc:
        print(f"Telegram best-effort notification failed: {exc}")


def _build_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        dropout_prob=0.1,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _configure_remote_mlflow() -> str:
    """Map DagsHub names without relying on the dirty legacy tracking module."""
    mappings = {
        "MLFLOW_TRACKING_URI": "DAGSHUB_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME": "DAGSHUB_REPO_OWNER",
        "MLFLOW_TRACKING_PASSWORD": "DAGSHUB_USER_TOKEN",
        "MLFLOW_S3_ENDPOINT_URL": "DAGSHUB_REPO_ENDPOINT",
        "AWS_ACCESS_KEY_ID": "DAGSHUB_USER_TOKEN",
        "AWS_SECRET_ACCESS_KEY": "DAGSHUB_USER_TOKEN",
    }
    for target, source in mappings.items():
        value = os.getenv(source, "").strip()
        if value:
            os.environ.setdefault(target, value)
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri or uri.startswith(("file:", "sqlite:")):
        raise RuntimeError("A remote MLFLOW_TRACKING_URI is required for official evaluation")
    mlflow.set_tracking_uri(uri)
    return uri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Data/processed/ich_v2/BrainICHPartial"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-component-ml", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-name", default="ich-v2-corrected-baseline")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    manifest = pd.read_csv(
        args.dataset_dir / "manifest.csv", dtype={"study_id": str, "patient_id": str}
    )
    selected = manifest.loc[manifest["fold"] == args.fold].sort_values("study_id")
    if args.limit:
        selected = selected.head(args.limit)
    if selected.empty:
        raise ValueError(f"Fold {args.fold} has no studies")

    truth, metadata_source = ground_truth_ich_context()
    selected = selected.merge(truth, on="study_id", how="left", validate="one_to_one")
    if selected[[f"gt_{key}" for key in VOLUME_KEYS]].isna().any().any():
        raise ValueError("Ground-truth merge produced missing ICH volumes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This evaluator is intended for the rented CUDA worker")
    model = _build_model(args.checkpoint, device)
    transform = Compose([
        LoadImaged(keys="image", image_only=False),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys="image", a_min=-200, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys="image", source_key="image"),
    ])

    checkpoint_hash = _sha256(args.checkpoint)
    _notify(
        "start",
        "ارزیابی baseline با هندسه اصلاح‌شده آغاز شد. تحلیل کوتاه: این اجرا مشخص می‌کند چه مقدار از ضعف نتیجه قدیمی ناشی از خود مدل و چه مقدار ناشی از محاسبه نادرست حجم بوده است. اقدام بعدی: مقایسه study-level و تصمیم درباره آموزش مجدد.",
        fold=args.fold, studies=len(selected), checkpoint_sha256=checkpoint_hash[:12],
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(selected.itertuples(index=False), start=1):
            study_started = time.perf_counter()
            transformed = transform({"image": row.image})
            image = transformed["image"]
            affine = np.asarray(image.affine, dtype=np.float64)
            tensor = image.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    tensor,
                    roi_size=(args.roi_size,) * 3,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=args.overlap,
                    mode="gaussian",
                )
            labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            raw_volumes = volumes_from_labelmap(labels, affine)
            cleaned = remove_small_components(
                labels, affine, minimum_ml=args.min_component_ml
            )
            volumes = volumes_from_labelmap(cleaned, affine)
            result = {
                "study_id": str(row.study_id),
                "patient_id": str(row.patient_id),
                "fold": int(row.fold),
                "gt_triage_class": int(row.gt_triage_class),
                "gt_fracture_prob": float(row.gt_fracture_prob),
                "gt_MLS_mm": float(row.gt_MLS_mm),
                "runtime_s": float(time.perf_counter() - study_started),
            }
            result.update({f"gt_{key}": float(getattr(row, f"gt_{key}")) for key in VOLUME_KEYS})
            result.update({f"raw_pred_{key}": float(raw_volumes[key]) for key in VOLUME_KEYS})
            result.update({f"pred_{key}": float(volumes[key]) for key in VOLUME_KEYS})
            rows.append(result)
            print(f"[{index}/{len(selected)}] {row.study_id} total_ml={sum(volumes.values()):.3f}")
            if index == max(1, len(selected) // 2):
                _notify(
                    "progress",
                    "نیمی از ارزیابی fold انجام شد. تحلیل کوتاه: inference و حجم‌سنجی فیزیکی پایدار بوده و خطای اجرایی دیده نشده است؛ قضاوت کیفیت تا تجمیع کل fold انجام نمی‌شود. اقدام بعدی: تکمیل نیمه دوم و محاسبه FPR و Macro-F1.",
                    fold=args.fold, completed=index, total=len(selected),
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, summary_path, summary = write_evaluation(pd.DataFrame(rows), args.output_dir)
    duration = time.perf_counter() - started
    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "fold": args.fold,
        "metadata_source": str(metadata_source),
        "dataset_dir": str(args.dataset_dir),
        "roi_size": args.roi_size,
        "overlap": args.overlap,
        "min_component_ml": args.min_component_ml,
        "duration_s": duration,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "evaluation_context": summary["evaluation_context"],
    }
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    if not args.skip_mlflow:
        _configure_remote_mlflow()
        mlflow.set_experiment("IAAA_BrainCT-ich-v2")
        with mlflow.start_run(run_name=args.run_name):
            mlflow.set_tags({
                "task": "ich",
                "stage": "corrected_baseline_evaluation",
                "evaluation_context": summary["evaluation_context"],
            })
            mlflow.log_params({
                "fold": args.fold,
                "studies": len(rows),
                "roi_size": args.roi_size,
                "overlap": args.overlap,
                "checkpoint_sha256": checkpoint_hash,
                "min_component_ml": args.min_component_ml,
            })
            metrics = {
                "oracle_context_macro_f1": summary["oracle_context_macro_f1"],
                "total_mae_ml": summary["total"]["mae_ml"],
                "total_bias_ml": summary["total"]["bias_ml"],
                "total_presence_f1": summary["total"]["presence_f1_at_0_1ml"],
                "normal_false_positive_rate": summary["total"]["normal_false_positive_rate"],
                "runtime_total_s": duration,
            }
            for key in VOLUME_KEYS:
                metrics[f"{key}_mae_ml"] = summary["subtypes"][key]["mae_ml"]
                metrics[f"{key}_presence_f1"] = summary["subtypes"][key]["presence_f1_at_0_1ml"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(args.output_dir), artifact_path="evaluation")

    _notify(
        "success",
        "ارزیابی baseline با هندسه اصلاح‌شده تمام شد. تحلیل کوتاه: معیارهای زیر اثر خالص ICH را با MLS و شکستگی ثابت نشان می‌دهند؛ بر اساس FPR و خطای حجم، نخستین فرضیه آموزشی انتخاب می‌شود. اقدام بعدی: اجرای یا رد baseline اصلاح‌شده طبق گیت کیفیت.",
        fold=args.fold,
        studies=len(rows),
        macro_f1=f"{summary['oracle_context_macro_f1']:.4f}",
        presence_f1=f"{summary['total']['presence_f1_at_0_1ml']:.4f}",
        normal_fpr=f"{summary['total']['normal_false_positive_rate']:.4f}",
        duration_min=f"{duration / 60:.1f}",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        load_dotenv(".env", override=False)
        _notify(
            "failure",
            "اجرای ارزیابی ICH-v2 متوقف شد. تحلیل کوتاه: این یک failure فنی است و نتیجه مدل محسوب نمی‌شود؛ تا رفع علت هیچ candidateی promote نخواهد شد. اقدام بعدی: بررسی traceback، اصلاح و یک بار تکرار کنترل‌شده.",
            error=type(exc).__name__, detail=str(exc)[:500],
        )
        raise
