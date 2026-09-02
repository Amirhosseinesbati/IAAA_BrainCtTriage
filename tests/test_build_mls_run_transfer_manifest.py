from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_mls_run_transfer_manifest import build_manifest
from scripts.verify_mls_run_transfer import verify_transfer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, status: str = "completed", include_epoch15: bool = True):
    root = tmp_path / "project"
    run = "mls-test-fold0-seed2026"
    training_manifest = root / "config.yaml"
    training_manifest.parent.mkdir(parents=True)
    training_manifest.write_text(
        f"run_name: {run}\n"
        "task: mls\n"
        "strategy: mls_heatmap\n"
        "tags:\n"
        "  compute_policy: cuda_only_no_cpu_fallback\n"
        "  fixed_audit_epoch: 15\n"
        "training_config:\n"
        "  fold: 0\n"
        "  seed: 3407\n"
        "  epochs: 23\n",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "launcher_status.json").write_text(json.dumps({
        "status": status,
        "exit_code": 0 if status == "completed" else 1,
        "manifest_sha256": _sha(training_manifest),
        "compute_policy": "cuda_only_no_cpu_fallback",
    }), encoding="utf-8")
    (artifact_root / "run.log").write_text("done\n", encoding="utf-8")
    checkpoint = (
        root / "models/checkpoints/mls_multitask" / run
        / "mls_multitask_epoch_015.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    report_dir = root / "reports/mls_experiments" / run
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text(
        "- Status: `completed`\n", encoding="utf-8",
    )
    epochs = list(range(1, 24)) if include_epoch15 else list(range(1, 15))
    (report_dir / "epoch_metrics.jsonl").write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in epochs),
        encoding="utf-8",
    )
    return root, run, training_manifest, artifact_root


class TransferManifestTests(unittest.TestCase):
    def test_builds_checksum_manifest_only_for_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, run, training_manifest, artifact_root = _fixture(tmp_path)
            output = artifact_root / "transfer_manifest.json"
            result = build_manifest(
                project_root=root,
                run_name=run,
                training_manifest=training_manifest,
                artifact_root=artifact_root,
                output=output,
            )
            self.assertEqual(result["status"], "ready_for_checksum_transfer")
            self.assertFalse(result["raw_medical_predictions_included"])
            self.assertEqual(
                result["artifacts"]["fixed_epoch_checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint").hexdigest(),
            )
            self.assertEqual(result["manifest_sha256"], _sha(output))
            local_dir = tmp_path / "downloaded"
            local_dir.mkdir()
            manifest_payload = json.loads(output.read_text(encoding="utf-8"))
            for metadata in manifest_payload["artifacts"].values():
                source = Path(metadata["source_path"])
                (local_dir / metadata["transfer_filename"]).write_bytes(source.read_bytes())
            verification = verify_transfer(
                manifest=output,
                expected_manifest_sha256=_sha(output),
                artifact_dir=local_dir,
                output=local_dir / "verification.json",
            )
            self.assertEqual(verification["artifacts_expected"], 6)
            self.assertEqual(verification["artifacts_verified"], 6)

    def test_refuses_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, run, training_manifest, artifact_root = _fixture(
                Path(directory), status="failed",
            )
            with self.assertRaises(RuntimeError):
                build_manifest(
                    project_root=root,
                    run_name=run,
                    training_manifest=training_manifest,
                    artifact_root=artifact_root,
                    output=artifact_root / "transfer_manifest.json",
                )

    def test_refuses_missing_fixed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, run, training_manifest, artifact_root = _fixture(
                Path(directory), include_epoch15=False,
            )
            with self.assertRaises(ValueError):
                build_manifest(
                    project_root=root,
                    run_name=run,
                    training_manifest=training_manifest,
                    artifact_root=artifact_root,
                    output=artifact_root / "transfer_manifest.json",
                )

    def test_local_verifier_refuses_wrong_remote_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, run, training_manifest, artifact_root = _fixture(tmp_path)
            manifest = artifact_root / "transfer_manifest.json"
            build_manifest(
                project_root=root,
                run_name=run,
                training_manifest=training_manifest,
                artifact_root=artifact_root,
                output=manifest,
            )
            with self.assertRaises(ValueError):
                verify_transfer(
                    manifest=manifest,
                    expected_manifest_sha256="0" * 64,
                    artifact_dir=tmp_path / "downloaded",
                    output=tmp_path / "verification.json",
                )

    def test_local_verifier_refuses_checksum_valid_but_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, run, training_manifest, artifact_root = _fixture(tmp_path)
            manifest = artifact_root / "transfer_manifest.json"
            build_manifest(
                project_root=root,
                run_name=run,
                training_manifest=training_manifest,
                artifact_root=artifact_root,
                output=manifest,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifacts"].pop("fixed_epoch_checkpoint")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            local_dir = tmp_path / "downloaded"
            local_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "artifact contract mismatch"):
                verify_transfer(
                    manifest=manifest,
                    expected_manifest_sha256=_sha(manifest),
                    artifact_dir=local_dir,
                    output=local_dir / "verification.json",
                )


if __name__ == "__main__":
    unittest.main()
