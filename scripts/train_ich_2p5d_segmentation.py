"""CLI for direct 2.5D ICH segmentation and physical-volume training."""

from __future__ import annotations

import argparse
import json

from src.strategies.ich_2p5d.segmentation_loss import (
    CONDITIONAL_SUBTYPE_MODES,
    SEGMENTATION_OBJECTIVES,
)
from src.strategies.ich_2p5d.segmentation_train import (
    CHECKPOINT_SELECTION_STRATEGIES,
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
    parser.add_argument("--empty-foreground-weight", type=float, default=0.0)
    parser.add_argument("--empty-foreground-top-fraction", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-selection-strategy",
        choices=CHECKPOINT_SELECTION_STRATEGIES,
        default="legacy",
    )
    parser.add_argument("--maximum-pos-weight", type=float, default=20.0)
    parser.add_argument("--segmentation-class-weight-power", type=float, default=0.0)
    parser.add_argument(
        "--maximum-segmentation-class-weight", type=float, default=8.0
    )
    parser.add_argument(
        "--segmentation-class-weight-basis",
        choices=("slice", "pixel"),
        default="slice",
    )
    parser.add_argument("--sampler-study-balance-power", type=float, default=0.0)
    parser.add_argument("--hard-negative-manifest")
    parser.add_argument("--hard-negative-multiplier", type=float, default=1.0)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--horizontal-symmetry-adapter", action="store_true")
    parser.add_argument("--five-slice-context-adapter", action="store_true")
    parser.add_argument("--sah-residual-adapter", action="store_true")
    parser.add_argument("--sah-residual-hidden-channels", type=int, default=16)
    parser.add_argument("--sah-maximum-logit-residual", type=float, default=8.0)
    parser.add_argument("--sah-include-incumbent-iph", action="store_true")
    parser.add_argument("--slice-context-radius", type=int, choices=(1, 2), default=1)
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument("--classification-head-only", action="store_true")
    parser.add_argument("--ivh-center-loss-weight", type=float, default=0.0)
    parser.add_argument("--ivh-center-square-size", type=int, default=11)
    parser.add_argument("--physical-volume-loss-weight", type=float, default=0.0)
    parser.add_argument("--diffuse-tversky-loss-weight", type=float, default=0.0)
    parser.add_argument("--sah-tversky-loss-weight", type=float, default=0.0)
    parser.add_argument("--sah-positive-pixel-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--segmentation-objective",
        choices=SEGMENTATION_OBJECTIVES,
        default="multiclass",
    )
    parser.add_argument("--foreground-dice-weight", type=float, default=0.40)
    parser.add_argument("--foreground-focal-weight", type=float, default=0.20)
    parser.add_argument("--conditional-subtype-weight", type=float, default=0.30)
    parser.add_argument("--subtype-ovr-weight", type=float, default=0.10)
    parser.add_argument(
        "--conditional-subtype-mode",
        choices=CONDITIONAL_SUBTYPE_MODES,
        default="cross_entropy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--skip-outer-evaluation",
        action="store_true",
        help="Train fully and select on calibration without reading the outer fold.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = vars(parser.parse_args())
    args["pretrained"] = not args.pop("no_pretrained")
    args["evaluate_outer"] = not args.pop("skip_outer_evaluation")
    result = run_segmentation_training(ICH25DSegmentationTrainConfig(**args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
