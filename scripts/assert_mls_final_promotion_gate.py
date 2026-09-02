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
EXPECTED_PROTOCOL = "deploy_aligned_fixed_three_seed_median_canonical_triage"
EXPECTED_HARD_GATES = {
    "macro_f1_improved",
    "accuracy_noninferior",
    "urgent_f1_improved",
    "normal_f1_not_below_minus_0p01",
    "critical_f1_not_below_minus_0p01",
    "no_fold_macro_drop_below_minus_0p01",
    "bootstrap_probability_at_least_0p95",
    "f1_3mm_noninferior",
    "f1_5mm_noninferior",
    "normal_to_critical_not_worse",
    "critical_to_normal_not_worse",
    "oracle_and_frozen_macro_direction_consistent",
    "oracle_and_frozen_urgent_direction_consistent",
    "full_immutable_fold_coverage",
}


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
    gates_value = summary.get("promotion_gates", {})
    gates = gates_value if isinstance(gates_value, dict) else {}
    sources_value = summary.get("sources", {})
    sources = sources_value if isinstance(sources_value, dict) else {}
    frozen_value = sources.get("frozen_champion_predictions", {})
    frozen_source = frozen_value if isinstance(frozen_value, dict) else {}
    baseline_value = sources.get("baseline_folds", [])
    candidate_value = sources.get("candidate_folds", [])
    baseline_sources = baseline_value if isinstance(baseline_value, list) else []
    candidate_sources = candidate_value if isinstance(candidate_value, list) else []
    baseline_folds = sorted(
        row.get("fold") for row in baseline_sources
        if isinstance(row, dict) and isinstance(row.get("fold"), int)
    )
    candidate_folds = sorted(
        row.get("fold") for row in candidate_sources
        if isinstance(row, dict) and isinstance(row.get("fold"), int)
    )
    expected_fold_sizes = {0: 70, 1: 67, 2: 67, 3: 66, 4: 68}

    def _valid_fold_sources(rows: Any) -> bool:
        if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
            return False
        for row in rows:
            if not isinstance(row, dict):
                return False
            fold = row.get("fold")
            checkpoint_hashes = row.get("checkpoint_sha256")
            if fold not in expected_fold_sizes or row.get("studies") != expected_fold_sizes[fold]:
                return False
            if not isinstance(checkpoint_hashes, dict) or len(checkpoint_hashes) != 3:
                return False
            hashes = [
                row.get("sha256"), row.get("audit_summary_sha256"),
                *checkpoint_hashes.values(),
            ]
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in hashes
            ):
                return False
        return True

    checks = {
        "schema_version": summary.get("schema_version") == 1,
        "canonical_protocol": summary.get("protocol") == EXPECTED_PROTOCOL,
        "promotion_eligible": summary.get("promotion_eligible") is True,
        "evaluation_scope_full_oof": summary.get("evaluation_scope") == "full_oof",
        "full_fold_coverage": summary.get("full_fold_coverage") is True,
        "selected_folds_exact": summary.get("selected_folds") == EXPECTED_FOLDS,
        "available_folds_exact": summary.get("available_folds") == EXPECTED_FOLDS,
        "studies_exact": int(summary.get("studies", -1)) == EXPECTED_STUDIES,
        "all_hard_gates_present": EXPECTED_HARD_GATES.issubset(set(gates)),
        "all_hard_gates_true": all(gates.get(name) is True for name in EXPECTED_HARD_GATES),
        "no_failed_hard_gates": summary.get("failed_hard_gates") == [],
        "baseline_fold_sources_exact": baseline_folds == EXPECTED_FOLDS,
        "candidate_fold_sources_exact": candidate_folds == EXPECTED_FOLDS,
        "baseline_fold_sources_checksum_bound": _valid_fold_sources(baseline_sources),
        "candidate_fold_sources_checksum_bound": _valid_fold_sources(candidate_sources),
        "frozen_champion_full_oof": frozen_source.get("studies") == EXPECTED_STUDIES,
        "frozen_champion_expected_hash_matches": (
            isinstance(frozen_source.get("sha256"), str)
            and frozen_source.get("sha256") == frozen_source.get("expected_sha256")
        ),
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
