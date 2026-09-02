"""Issue a checksum-bound MLS packaging authorization only after full OOF gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_STUDIES = 338


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def authorize(summary_path: Path, output_path: Path) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    output_path = output_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "promotion_eligible": summary.get("promotion_eligible") is True,
        "evaluation_scope_full_oof": summary.get("evaluation_scope") == "full_oof",
        "full_fold_coverage": summary.get("full_fold_coverage") is True,
        "selected_folds_exact": summary.get("selected_folds") == EXPECTED_FOLDS,
        "available_folds_exact": summary.get("available_folds") == EXPECTED_FOLDS,
        "studies_exact": int(summary.get("studies", -1)) == EXPECTED_STUDIES,
        "coverage_gate_true": summary.get("promotion_gates", {}).get(
            "full_immutable_fold_coverage"
        ) is True,
        "no_failed_hard_gates": summary.get("failed_hard_gates") == [],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"MLS final packaging authorization refused: {failed}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "authorized_for_clean_submission_packaging",
        "authorized_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_summary_path": str(summary_path),
        "aggregate_summary_sha256": _sha256(summary_path),
        "folds": EXPECTED_FOLDS,
        "studies": EXPECTED_STUDIES,
        "checks": checks,
        "zip_created": False,
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(authorize(args.aggregate_summary, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
