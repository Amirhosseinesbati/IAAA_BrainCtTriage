"""CLI for leakage-safe adjacent-slice ICH gate training."""

from __future__ import annotations

import argparse
import json

from src.strategies.ich_2p5d.train import ICH25DTrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="Data/processed/ich_2p5d/slice_manifest.csv")
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--any-loss-weight", type=float, default=2.0)
    parser.add_argument("--maximum-pos-weight", type=float, default=20.0)
    parser.add_argument("--minimum-calibration-sensitivity", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    args = vars(parser.parse_args())
    args["pretrained"] = not args.pop("no_pretrained")
    print(json.dumps(run_training(ICH25DTrainConfig(**args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
