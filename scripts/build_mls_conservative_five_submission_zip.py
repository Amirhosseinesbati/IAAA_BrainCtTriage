"""Build the conservative five-checkpoint MLS package deterministically.

The verified extracted Exp16/Exp15r package is treated as read-only. The
builder substitutes a dedicated MLS runtime plus five exact checkpoints and
creates an uncompressed archive to avoid a large CPU compression job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import BinaryIO, Any


MAX_BYTES = 1_073_741_824
EXPECTED = {
    "models/mls/fold0.pth": {
        "bytes": 124_914_021,
        "sha256": "bddcda5013cb88905a421095e71a28189181fde657aa3576be88f276d88ad15b",
    },
    "models/mls/fold0_regression.pth": {
        "bytes": 124_898_917,
        "sha256": "4b1f3847b335e4e18af989e312f6c19140948524b4e6b3390bdfe66ffc52548a",
    },
    "models/mls/fold1.pth": {
        "bytes": 124_890_565,
        "sha256": "98923f724b2d61c4a8671ef0405ab7c205913e0829b72c0e32ae317ef23cfccb",
    },
    "models/mls/fold1_regression.pth": {
        "bytes": 124_898_917,
        "sha256": "da763a0f3a5e81da8c6d2d1638a093186688789868acd450d14bf9e76f705509",
    },
    "models/mls/fold2.pth": {
        "bytes": 124_898_469,
        "sha256": "e4c5f91c4e9fb97b766477615f6e42244bed2ee53f85c98f4f1353146cb6e16e",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify(path: Path, expected: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if actual != expected:
        raise RuntimeError(f"Artifact verification failed for {label}: {actual}")


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=(2026, 9, 2, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        chunk = source.read(4 * 1024 * 1024)
        if not chunk:
            break
        destination.write(chunk)


def _write_file(archive: zipfile.ZipFile, relative: str, path: Path) -> None:
    with path.open("rb") as source, archive.open(_zip_info(relative), "w") as destination:
        _copy_stream(source, destination)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    template = args.runtime_template.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    model_paths = {
        "models/mls/fold0.pth": checkpoint_root / "mls-vast-exp16-w32-fold0-strict-ensemble-refresh" / "mls_multitask_best_selector_auc.pth",
        "models/mls/fold0_regression.pth": checkpoint_root / "mls-vast-exp19-w32-fold0-dual-selector-replication" / "mls_multitask_epoch_021.pth",
        "models/mls/fold1.pth": checkpoint_root / "mls-local-v2-exp09-w32-fold1-hybridsoft-transfer" / "mls_multitask_epoch_015.pth",
        "models/mls/fold1_regression.pth": checkpoint_root / "mls-vast-exp18-w32-fold1-dual-selector-transfer" / "mls_multitask_epoch_021.pth",
        "models/mls/fold2.pth": checkpoint_root / "mls-vast-exp15r-w32-fold2-strict-repro-control" / "mls_multitask_epoch_017.pth",
    }
    for relative, expected in EXPECTED.items():
        _verify(model_paths[relative], expected, relative)
    if not template.is_file():
        raise FileNotFoundError(template)

    original_manifest = json.loads((source / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in original_manifest["ich"]["artifacts"].items():
        _verify(source / relative, expected, relative)
    fracture_root = source / "models" / "fracture"
    fracture_manifest = json.loads((fracture_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in fracture_manifest["artifacts"].items():
        _verify(fracture_root / relative, expected, f"models/fracture/{relative}")

    files: dict[str, Path] = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if (
            "__pycache__" in path.relative_to(source).parts
            or path.suffix in {".pyc", ".zip"}
            or relative in {"MODEL_MANIFEST.json", "ARCHIVE_MANIFEST.json", "mls.py"}
            or relative.startswith("models/mls/")
        ):
            continue
        files[relative] = path
    files["mls.py"] = template
    files.update(model_paths)

    manifest = copy.deepcopy(original_manifest)
    manifest["release"] = "mls-conservative-five-2026-09-02"
    manifest["mls"]["strategy"] = (
        "three-member median ensemble; fold0 and fold1 use frozen 90/10 "
        "regression-only dual-selector complements, fold2 remains Exp15r"
    )
    manifest["mls"]["artifacts"] = EXPECTED
    manifest["mls"]["conservative_component_recipe"] = {
        "alpha": 0.1,
        "component": "mls_mm_only",
        "fold0": "Exp16 epoch16 baseline + Exp19 epoch21 regression",
        "fold1": "Exp09 epoch15 baseline + Exp18 epoch21 regression",
        "fold2": "Exp15r epoch17 unchanged",
        "pooling": {
            "family": "severity_window",
            "anchor_window_radius": 3,
            "selector_gate": 0.5,
            "min_active_slices": 3,
            "quantile": 0.75,
            "probability_weighted": True,
            "heatmap_guard_ratio": 0.0,
        },
        "oof_204": {
            "baseline_mae_mm": 1.4725910753888243,
            "candidate_mae_mm": 1.4615219590361412,
            "baseline_boundary_f1": 0.850206611570248,
            "candidate_boundary_f1": 0.8558884297520661,
            "baseline_objective": 1.7721778522483282,
            "candidate_objective": 1.749745099532009,
        },
    }
    model_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries = [
        {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for relative, path in sorted(files.items())
    ]
    entries.append(
        {
            "path": "MODEL_MANIFEST.json",
            "bytes": len(model_manifest),
            "sha256": _sha256_bytes(model_manifest),
        }
    )
    archive_manifest = (
        json.dumps({"schema_version": 1, "files": entries}, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    required = {
        "model.py", "submission.py", "triage.py", "mls.py",
        "MODEL_MANIFEST.json", "ARCHIVE_MANIFEST.json",
        "models/ich/presence_gate.pth", "models/ich/segresnet.pth",
        "models/fracture/manifest.json", *EXPECTED,
    }
    names = set(files) | {"MODEL_MANIFEST.json", "ARCHIVE_MANIFEST.json"}
    if required - names:
        raise RuntimeError(f"Missing package members: {sorted(required - names)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr(_zip_info("ARCHIVE_MANIFEST.json"), archive_manifest)
        archive.writestr(_zip_info("MODEL_MANIFEST.json"), model_manifest)
        for relative, path in sorted(files.items()):
            _write_file(archive, relative, path)
    os.replace(temporary, args.output)
    if args.output.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"Archive exceeds one GiB: {args.output.stat().st_size}")
    with zipfile.ZipFile(args.output) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or required - set(members):
            raise RuntimeError("Archive membership validation failed")
        forbidden = [
            name for name in members
            if name.startswith(("Data/", "checkpoint/", "reports/", ".git/"))
            or "__pycache__" in name or name.endswith((".pyc", ".zip"))
        ]
        if forbidden:
            raise RuntimeError(f"Archive contains forbidden paths: {forbidden[:5]}")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    result = {
        "schema_version": 1,
        "archive": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": _sha256(args.output),
        "files": len(names),
        "compression": "ZIP_STORED",
        "source_root": str(source),
        "source_tree_mutated": False,
        "runtime_sha256": _sha256(template),
        "models": EXPECTED,
    }
    _atomic_text(args.report, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
