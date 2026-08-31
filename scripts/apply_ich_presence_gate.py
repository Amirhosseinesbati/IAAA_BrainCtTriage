"""Apply a fixed 2.5D presence rule to study-level ICH volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.strategies.ich_2p5d.evaluation import PresenceRule
from src.strategies.ich_2p5d.gating import gate_volume_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-predictions", required=True)
    parser.add_argument("--presence-predictions", required=True)
    parser.add_argument("--presence-rule", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rule = PresenceRule(**json.loads(Path(args.presence_rule).read_text(encoding="utf-8")))
    predictions, summary = gate_volume_predictions(
        pd.read_csv(args.volume_predictions, dtype={"study_id": str}),
        pd.read_csv(args.presence_predictions, dtype={"study_id": str}),
        rule,
    )
    predictions.to_csv(output / "gated_study_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
