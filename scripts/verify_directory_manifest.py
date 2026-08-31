"""Compute a deterministic content fingerprint for a directory tree.

The aggregate includes every relative path, file size, and SHA-256 digest.  It
is therefore suitable for verifying that a DVC checkout on a remote worker is
byte-for-byte identical to the local dataset without copying the dataset back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def directory_manifest(root: Path) -> dict[str, int | str]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    aggregate = hashlib.sha256()
    total_bytes = 0

    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                file_hash.update(chunk)
        aggregate.update(relative)
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.digest())
        total_bytes += size

    return {
        "root": str(root),
        "files": len(files),
        "bytes": total_bytes,
        "manifest_sha256": aggregate.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = directory_manifest(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
