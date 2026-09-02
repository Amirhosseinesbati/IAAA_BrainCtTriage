from __future__ import annotations

import unittest
from pathlib import Path


class SftpDownloadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).resolve().parents[1] / "scripts/download_mls_run_sftp.ps1"
        ).read_text(encoding="utf-8")

    def test_uses_resumable_sftp_and_never_scp(self) -> None:
        self.assertIn("reget", self.source)
        self.assertIn("& sftp @arguments", self.source)
        self.assertNotIn("& scp", self.source)

    def test_requires_exact_six_artifact_contract(self) -> None:
        for name in (
            "training_manifest", "launcher_status", "fixed_epoch_checkpoint",
            "report", "epoch_metrics", "run_log",
        ):
            self.assertIn(f"'{name}'", self.source)
        self.assertIn("artifacts_verified -ne 6", self.source)
        self.assertIn("artifacts_expected -ne 6", self.source)
        self.assertIn("checks.PSObject.Properties", self.source)

    def test_rejects_paths_outside_workspace_and_pins_manifest_hash(self) -> None:
        self.assertIn("/workspace/", self.source)
        self.assertIn("ExpectedManifestSha256", self.source)
        self.assertIn("Downloaded manifest checksum mismatch", self.source)

    def test_runs_independent_local_verifier(self) -> None:
        self.assertIn("scripts/verify_mls_run_transfer.py", self.source)
        self.assertIn("transfer_verification.json", self.source)

    def test_normalizes_windows_paths_for_sftp_batch_parser(self) -> None:
        self.assertIn("ConvertTo-SftpLocalPath", self.source)
        self.assertIn(".Replace('\\', '/')", self.source)


if __name__ == "__main__":
    unittest.main()
