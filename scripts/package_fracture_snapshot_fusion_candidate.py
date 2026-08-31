"""Assemble a checksum-verified fracture MIL + detector snapshot package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_fracture_mil_candidate import (
    _copy_verified,
    _load_json,
    _sha256,
    _strip_detector,
)


def _fold_entry(payload: object, fold: int, label: str) -> dict[str, object]:
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError(f"{label} must contain five folds")
    entry = payload[fold]
    if not isinstance(entry, dict) or int(entry["fold"]) != fold:
        raise ValueError(f"{label} fold order mismatch at {fold}")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch10-cache-root", type=Path, required=True)
    parser.add_argument("--epoch15-cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--incumbent-calibration-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-calibration-manifest", type=Path, required=True)
    parser.add_argument("--decision-calibration", type=Path, required=True)
    parser.add_argument("--meta-run-id", required=True)
    parser.add_argument("--snapshot-fusion-weight", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.snapshot_fusion_weight <= 1.0:
        raise ValueError("snapshot-fusion-weight must be within [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    incumbent_calibration = _load_json(args.incumbent_calibration_manifest)
    snapshot_calibration = _load_json(args.snapshot_calibration_manifest)
    decision = _load_json(args.decision_calibration)
    if not isinstance(decision, dict):
        raise ValueError("Decision calibration must be an object")

    fold_payloads: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    for fold in range(5):
        epoch10_manifest = _load_json(
            args.epoch10_cache_root / f"fold_{fold}" / "manifest.json"
        )
        epoch15_manifest = _load_json(
            args.epoch15_cache_root / f"fold_{fold}" / "manifest.json"
        )
        if not isinstance(epoch10_manifest, dict) or not isinstance(
            epoch15_manifest, dict
        ):
            raise ValueError("Cache manifests must be JSON objects")
        for key in ("n_slices", "n_studies", "n_patients"):
            if epoch10_manifest.get(key) != epoch15_manifest.get(key):
                raise RuntimeError(f"Fold {fold} snapshot cache {key} mismatch")
        if epoch10_manifest.get("metadata_sha256") != epoch15_manifest.get(
            "metadata_sha256"
        ):
            raise RuntimeError(f"Fold {fold} snapshot metadata checksum mismatch")
        epoch10_artifacts = epoch10_manifest.get("artifacts")
        epoch15_artifacts = epoch15_manifest.get("artifacts")
        if not isinstance(epoch10_artifacts, dict) or not isinstance(
            epoch15_artifacts, dict
        ):
            raise ValueError("Cache artifact manifests must be objects")
        if epoch10_artifacts.get("slices.csv") != epoch15_artifacts.get(
            "slices.csv"
        ):
            raise RuntimeError(f"Fold {fold} snapshot slice identity mismatch")

        detector_relative = Path(f"fold_{fold}") / "detector_epoch10.pt"
        snapshot_relative = Path(f"fold_{fold}") / "detector_epoch15.pt"
        detector_source_hash, detector_hash = _strip_detector(
            Path(str(epoch10_manifest["checkpoint"])),
            args.output / detector_relative,
            str(epoch10_manifest["checkpoint_sha256"]),
        )
        snapshot_source_hash, snapshot_hash = _strip_detector(
            Path(str(epoch15_manifest["checkpoint"])),
            args.output / snapshot_relative,
            str(epoch15_manifest["checkpoint_sha256"]),
        )

        mil_relatives: list[str] = []
        for source in sorted(
            (args.model_root / f"fold_{fold}_v2").glob("model_seed*.pt")
        ):
            relative = Path(f"fold_{fold}") / source.name
            _copy_verified(source, args.output / relative)
            mil_relatives.append(relative.as_posix())
        if len(mil_relatives) != 3:
            raise RuntimeError(f"Fold {fold} does not contain three MIL heads")

        incumbent = _fold_entry(incumbent_calibration, fold, "incumbent calibration")
        snapshot = _fold_entry(snapshot_calibration, fold, "snapshot calibration")
        snapshot_scores = snapshot.get("snapshot_training_scores")
        if not isinstance(snapshot_scores, list) or not snapshot_scores:
            raise ValueError(f"Fold {fold} snapshot calibration scores are invalid")
        fold_payloads.append(
            {
                "fold": fold,
                "detector": detector_relative.as_posix(),
                "detector_source_sha256": detector_source_hash,
                "snapshot_detector": snapshot_relative.as_posix(),
                "snapshot_detector_source_sha256": snapshot_source_hash,
                "mil_heads": mil_relatives,
                "candidate_weight": float(incumbent["candidate_weight"]),
                "reference_training_scores": incumbent["reference_training_scores"],
                "candidate_training_scores": incumbent["candidate_training_scores"],
                "snapshot_training_scores": snapshot_scores,
                "snapshot_fusion_weight": args.snapshot_fusion_weight,
            }
        )
        for relative, digest in (
            (detector_relative, detector_hash),
            (snapshot_relative, snapshot_hash),
        ):
            path = args.output / relative
            artifacts[relative.as_posix()] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        for relative_text in mil_relatives:
            path = args.output / relative_text
            artifacts[relative_text] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }

    threshold = float(decision["selected_threshold_all_oof"])
    manifest = {
        "schema_version": 2,
        "candidate": "yolov8s-epoch10-15-5fold-sa-mil-snapshot-fixed040-v1",
        "status": "promising_pending_packaged_parity_runtime_and_leaderboard",
        "source_meta_mlflow_run_id": args.meta_run_id,
        "inference": {
            "image_size": 512,
            "batch_size": args.batch_size,
            "confidence": 0.001,
            "bone_window_width": 1000.0,
            "bone_window_level": 400.0,
            "jpeg_quality": 95,
            "detector_checkpoint": "optimizer_stripped_inference_only",
        },
        "decision_calibration": {
            "threshold": threshold,
            "mapping": decision["threshold_mapping"],
            "source": "five-fold OOF fixed-score leave-one-fold-out threshold evaluation",
        },
        "folds": fold_payloads,
        "artifacts": artifacts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# Fracture detector + SA-MIL + snapshot fusion candidate\n\n"
        "Five epoch-10 YOLOv8s detectors provide adjacent-pair scores and MIL "
        "embeddings. Five epoch-15 snapshots are averaged slice-wise with epoch 10, "
        "pooled by top-5 mean, calibrated with outer-train empirical CDFs, and fused "
        f"at fixed weight {args.snapshot_fusion_weight:.2f}. The official 0.5 cutoff "
        f"maps to OOF score {threshold:.6f}.\n\n"
        "Offline deployable OOF macro AUC: 0.91648 (incumbent 0.90781); worst-fold "
        "AUC: 0.86417. Bootstrap uncertainty crosses zero, so this remains an A/B "
        "candidate until packaged parity, runtime, and real leaderboard validation.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": manifest["candidate"],
                "artifact_count": len(artifacts),
                "artifact_bytes": sum(
                    int(item["bytes"]) for item in artifacts.values()
                ),
                "decision_threshold": threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
