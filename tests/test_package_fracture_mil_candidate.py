from __future__ import annotations

import hashlib

from scripts.package_fracture_mil_candidate import _sha256


def test_sha256_matches_known_payload(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"fracture-mil")
    assert _sha256(path) == hashlib.sha256(b"fracture-mil").hexdigest()
