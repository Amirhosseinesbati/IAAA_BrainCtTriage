"""CLI for direct 2.5D ICH segmentation and physical-volume training."""

from __future__ import annotations

import argparse
import json

from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    run_segmentation_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--manifest-path", default="Data/processed/ich_2p5d/slice_manifest.csv"
    )
    parser.add_argument("--architecture", default="unetplusplus")
    parser.add_argument("--encoder-name", default="efficientnet-b2")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classification-loss-weight", type=float, default=0.25)
    parser.add_argument("--classification-focal-gamma", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.15)
    parser.add_argument("--maximum-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    args = vars(parser.parse_args())
    args["pretrained"] = not args.pop("no_pretrained")
    result = run_segmentation_training(ICH25DSegmentationTrainConfig(**args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
