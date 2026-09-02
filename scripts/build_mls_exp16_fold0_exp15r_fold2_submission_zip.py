"""Build the two-fold MLS challenger without mutating submission sources.

The dirty/shared ``submission`` tree is read-only.  This builder substitutes
the promoted Exp16 fold-0 and Exp15r fold-2 checkpoints only inside a
deterministic ZIP and synthesizes both manifests in memory.  ZIP_STORED avoids
an unnecessary local CPU compression job while keeping the archive below the
one-GiB competition limit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "submission"
FOLD0_CHALLENGER = (
    PROJECT_ROOT
    / "checkpoint"
    / "mls"
    / "mls-vast-exp16-w32-fold0-strict-ensemble-refresh"
    / "mls_multitask_best_selector_auc.pth"
)
FOLD2_CHALLENGER = (
    PROJECT_ROOT
    / "checkpoint"
    / "mls"
    / "mls-vast-exp15r-w32-fold2-strict-repro-control"
    / "mls_multitask_epoch_017.pth"
)
OUTPUT = SOURCE / "iaaa_brain_ct_triage_mls_exp16_fold0_exp15r_fold2_20260902.zip"
MAX_BYTES = 1_073_741_824
EXPECTED_FOLD0 = {
    "bytes": 124_914_021,
    "sha256": "bddcda5013cb88905a421095e71a28189181fde657aa3576be88f276d88ad15b",
}
EXPECTED_FOLD2 = {
    "bytes": 124_898_469,
    "sha256": "e4c5f91c4e9fb97b766477615f6e42244bed2ee53f85c98f4f1353146cb6e16e",
}
SUBSTITUTIONS = {
    "models/mls/fold0.pth": (FOLD0_CHALLENGER, EXPECTED_FOLD0),
    "models/mls/fold2.pth": (FOLD2_CHALLENGER, EXPECTED_FOLD2),
}
REQUIRED = {
    "model.py",
    "submission.py",
    "triage.py",
    "MODEL_MANIFEST.json",
    "ARCHIVE_MANIFEST.json",
    "models/ich/presence_gate.pth",
    "models/ich/segresnet.pth",
    "models/mls/fold0.pth",
    "models/mls/fold1.pth",
    "models/mls/fold2.pth",
    "models/fracture/manifest.json",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify(path: Path, expected: dict[str, object], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    actual_hash = sha256_path(path)
    if actual_size != int(expected["bytes"]) or actual_hash != str(expected["sha256"]):
        raise RuntimeError(
            f"Artifact verification failed for {label}: "
            f"bytes={actual_size}, sha256={actual_hash}"
        )


def _source_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(SOURCE).as_posix()
        if (
            "__pycache__" in path.relative_to(SOURCE).parts
            or path.suffix in {".pyc", ".zip"}
            or relative in {"MODEL_MANIFEST.json", "ARCHIVE_MANIFEST.json"}
        ):
            continue
        files[relative] = path
    for relative, (checkpoint, _) in SUBSTITUTIONS.items():
        files[relative] = checkpoint
    return files


def _locked_profile() -> dict[str, object]:
    return {
        "family": "severity_window",
        "anchor_window_radius": 3,
        "selector_gate": 0.5,
        "min_active_slices": 3,
        "quantile": 0.75,
        "probability_weighted": True,
        "heatmap_guard_ratio": 0.0,
    }


def _model_manifest() -> bytes:
    original = json.loads((SOURCE / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    manifest = copy.deepcopy(original)
    manifest["release"] = "mls-exp16-fold0-exp15r-fold2-2026-09-02"
    manifest["mls"]["strategy"] = (
        "three-fold HRNet-W32 median ensemble with Exp16 strict fold0, "
        "historical fold1, Exp15r strict fold2, and locked severity-window pooling"
    )
    manifest["mls"]["artifacts"]["models/mls/fold0.pth"] = EXPECTED_FOLD0
    manifest["mls"]["artifacts"]["models/mls/fold2.pth"] = EXPECTED_FOLD2
    manifest["mls"]["fold0_promotion"] = {
        "run_name": "mls-vast-exp16-w32-fold0-strict-ensemble-refresh",
        "mlflow_run_id": "a2478b8410d74de2b2806ef08d79051d",
        "checkpoint": "best_selector_auc_epoch016",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "training_determinism": "strict",
        "locked_profile": _locked_profile(),
        "fold0_oof": {
            "n_studies": 70,
            "mae_mm": 1.604477701016835,
            "rmse_mm": 3.351474576216932,
            "bias_mm": 0.037587506260190716,
            "boundary_f1": 0.8273325590398761,
            "selection_objective": 1.9498125829370827,
        },
        "historical_fold0_delta": {
            "reference": "mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/epoch15",
            "mae_mm": -0.06007561564445485,
            "boundary_f1": 0.005069618132867326,
            "selection_objective": -0.07021485191019017,
        },
    }
    manifest["mls"]["fold2_promotion"] = {
        "run_name": "mls-vast-exp15r-w32-fold2-strict-repro-control",
        "mlflow_run_id": "a35975ea4dd242f4b9c12dbdbc1e7491",
        "checkpoint": "epoch017",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "training_determinism": "strict",
        "locked_profile": _locked_profile(),
        "fold2_oof": {
            "n_studies": 67,
            "mae_mm": 1.5483543317709396,
            "rmse_mm": 2.5451124028714527,
            "bias_mm": -0.3347211731013967,
            "boundary_f1": 0.8925925925925926,
            "selection_objective": 1.7631691465857544,
        },
        "historical_fold2_delta": {
            "reference": "mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/epoch15",
            "mae_mm": -0.1660044584701314,
            "boundary_f1": -0.0021043771043770745,
            "selection_objective": -0.16179570426137757,
        },
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def main() -> None:
    base_manifest = json.loads((SOURCE / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    declared = {
        **base_manifest["ich"]["artifacts"],
        **base_manifest["mls"]["artifacts"],
    }
    for relative, expected in declared.items():
        if relative in SUBSTITUTIONS:
            continue
        _verify(SOURCE / relative, expected, relative)
    for relative, (checkpoint, expected) in SUBSTITUTIONS.items():
        _verify(checkpoint, expected, f"promoted {relative}")

    fracture_root = SOURCE / "models" / "fracture"
    fracture_manifest = json.loads((fracture_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in fracture_manifest["artifacts"].items():
        _verify(fracture_root / relative, expected, f"models/fracture/{relative}")

    files = _source_files()
    model_manifest = _model_manifest()
    archive_entries: list[dict[str, object]] = []
    for relative, path in sorted(files.items()):
        archive_entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    archive_entries.append(
        {
            "path": "MODEL_MANIFEST.json",
            "bytes": len(model_manifest),
            "sha256": sha256_bytes(model_manifest),
        }
    )
    archive_manifest = (
        json.dumps(
            {"schema_version": 1, "files": sorted(archive_entries, key=lambda item: str(item["path"]))},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    names = set(files) | {"MODEL_MANIFEST.json", "ARCHIVE_MANIFEST.json"}
    missing = REQUIRED - names
    if missing:
        raise RuntimeError(f"Submission source is missing: {sorted(missing)}")
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr(_zip_info("ARCHIVE_MANIFEST.json"), archive_manifest)
        archive.writestr(_zip_info("MODEL_MANIFEST.json"), model_manifest)
        for relative, path in sorted(files.items()):
            _write_file(archive, relative, path)
    os.replace(temporary, OUTPUT)

    if OUTPUT.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"Archive exceeds one GiB: {OUTPUT.stat().st_size}")
    with zipfile.ZipFile(OUTPUT) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or REQUIRED - set(members):
            raise RuntimeError("Archive membership validation failed")
        forbidden = [
            name
            for name in members
            if name.startswith(("Data/", "checkpoint/", "reports/", ".git/"))
            or "__pycache__" in name
            or name.endswith((".pyc", ".zip"))
        ]
        if forbidden:
            raise RuntimeError(f"Archive contains forbidden paths: {forbidden[:5]}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")

    print(
        json.dumps(
            {
                "archive": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256_path(OUTPUT),
                "files": len(names),
                "compression": "ZIP_STORED",
                "source_tree_mutated": False,
                "fold0": EXPECTED_FOLD0,
                "fold2": EXPECTED_FOLD2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
