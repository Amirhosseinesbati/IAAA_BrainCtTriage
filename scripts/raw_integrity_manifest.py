"""Build and compare independent SHA-256 manifests for ``Data/raw``.

This is deliberately independent of DVC's directory hash.  It verifies every
file byte-for-byte after a remote ``dvc pull`` without decoding medical images
or running any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterator[Path]:
    yield from sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def build(root: Path, manifest: Path, summary: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw-data directory does not exist: {root}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    tree = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for path in _files(root):
            record = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            canonical = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            stream.write(canonical + "\n")
            tree.update(canonical.encode("utf-8") + b"\n")
            file_count += 1
            total_bytes += int(record["bytes"])
    os.replace(temporary, manifest)
    result = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
        "manifest_filename": manifest.name,
    }
    temporary_summary = summary.with_suffix(summary.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary_summary, summary)
    return result


def _load_manifest(path: Path) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[str(record["path"])] = (int(record["bytes"]), str(record["sha256"]))
    return records


def compare(left: Path, right: Path, max_differences: int) -> dict[str, Any]:
    a = _load_manifest(left)
    b = _load_manifest(right)
    missing_right = sorted(set(a) - set(b))
    missing_left = sorted(set(b) - set(a))
    changed = sorted(path for path in set(a) & set(b) if a[path] != b[path])
    result = {
        "identical": not missing_right and not missing_left and not changed,
        "left_files": len(a),
        "right_files": len(b),
        "missing_from_right_count": len(missing_right),
        "missing_from_left_count": len(missing_left),
        "changed_count": len(changed),
        "examples": {
            "missing_from_right": missing_right[:max_differences],
            "missing_from_left": missing_left[:max_differences],
            "changed": changed[:max_differences],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--summary", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--max-differences", type=int, default=20)
    args = parser.parse_args()

    if args.command == "build":
        result = build(args.root, args.manifest, args.summary)
    else:
        result = compare(args.left, args.right, args.max_differences)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "compare" and not result["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
