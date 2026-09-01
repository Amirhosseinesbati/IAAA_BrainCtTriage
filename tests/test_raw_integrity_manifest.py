from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.raw_integrity_manifest import build, compare


class RawIntegrityManifestTests(unittest.TestCase):
    def test_build_is_deterministic_and_compare_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            left_root = base / "left"
            right_root = base / "right"
            left_root.mkdir()
            right_root.mkdir()
            (left_root / "a.bin").write_bytes(b"abc")
            (right_root / "a.bin").write_bytes(b"abc")
            (left_root / "nested").mkdir()
            (right_root / "nested").mkdir()
            (left_root / "nested" / "b.bin").write_bytes(b"xyz")
            (right_root / "nested" / "b.bin").write_bytes(b"xyz")

            left_manifest = base / "left.jsonl"
            right_manifest = base / "right.jsonl"
            left_summary = build(left_root, left_manifest, base / "left-summary.json")
            right_summary = build(right_root, right_manifest, base / "right-summary.json")

            self.assertEqual(left_summary["tree_sha256"], right_summary["tree_sha256"])
            self.assertTrue(compare(left_manifest, right_manifest, 5)["identical"])

            (right_root / "a.bin").write_bytes(b"changed")
            build(right_root, right_manifest, base / "right-summary.json")
            comparison = compare(left_manifest, right_manifest, 5)
            self.assertFalse(comparison["identical"])
            self.assertEqual(comparison["changed_count"], 1)


if __name__ == "__main__":
    unittest.main()
