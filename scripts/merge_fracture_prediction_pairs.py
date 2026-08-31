"""Merge private OOF prediction columns with strict identity validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _candidate(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected PATH=COLUMN")
    path, column = value.rsplit("=", 1)
    return Path(path), column


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identity = ["study_id", "truth", "outer_fold"]
    merged = pd.read_csv(args.reference, dtype={"study_id": str})
    missing = set(identity).difference(merged.columns)
    if missing:
        raise ValueError(f"Reference is missing columns: {sorted(missing)}")
    if merged[identity].duplicated().any():
        raise ValueError("Reference contains duplicate study identities")
    for path, column in args.candidate:
        candidate = pd.read_csv(path, dtype={"study_id": str})
        if "outer_fold" not in candidate and "fold" in candidate:
            candidate = candidate.rename(columns={"fold": "outer_fold"})
        required = set(identity + [column])
        missing = required.difference(candidate.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        merged = merged.merge(
            candidate[identity + [column]],
            on=identity,
            how="inner",
            validate="one_to_one",
        )
    if merged.empty:
        raise ValueError("Merged prediction table is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        {
            "rows": len(merged),
            "studies": int(merged["study_id"].nunique()),
            "folds": sorted(int(value) for value in merged["outer_fold"].unique()),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
