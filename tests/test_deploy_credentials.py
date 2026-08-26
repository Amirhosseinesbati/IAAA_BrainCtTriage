from __future__ import annotations

import unittest

from src.deploy.deploy import _validate_dagshub_url


class TestDagsHubCredentialValidation(unittest.TestCase):
    def test_dagshub_url_requires_exact_repository_owner(self) -> None:
        _validate_dagshub_url(
            "https://dagshub.com/amiresbati62/BrainCtTriage.s3",
            "amiresbati62",
            "BrainCtTriage",
            ".s3",
            "DAGSHUB_REPO_ENDPOINT",
        )

    def test_dagshub_url_rejects_stale_repository_owner(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "amiresbati62/BrainCtTriage"):
            _validate_dagshub_url(
                "https://dagshub.com/amiresbati52/BrainCtTriage.s3",
                "amiresbati62",
                "BrainCtTriage",
                ".s3",
                "DAGSHUB_REPO_ENDPOINT",
            )
