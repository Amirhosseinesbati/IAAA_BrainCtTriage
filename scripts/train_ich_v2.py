"""CLI for staged ICH-v2 SegResNet training."""

from __future__ import annotations

import argparse
import json

from src.strategies.ich_v2.train import ICHV2TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="Data/processed/ich_v2/BrainICHPartial")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--samples-per-volume", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--min-component-ml", type=float, default=0.1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train-studies", type=int)
    parser.add_argument("--max-val-studies", type=int)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--focal-weight", type=float, default=0.4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--background-weight", type=float, default=0.2)
    args = parser.parse_args()
    config = ICHV2TrainConfig(**vars(args))
    print(json.dumps(run_training(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
