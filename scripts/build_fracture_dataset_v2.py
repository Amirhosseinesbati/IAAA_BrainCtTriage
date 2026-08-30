"""Build the study-aware fracture detector dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fracture.dataset import FractureDatasetConfig, FractureDatasetV2Builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dicom", type=Path, default=Path("Data/raw/training"))
    parser.add_argument("--raw-json", type=Path, default=Path("Data/raw/annotations"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--negative-slices", type=int, default=12)
    parser.add_argument("--positive-extra-negatives", type=int, default=8)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--positive-repeat", type=int, default=1)
    parser.add_argument(
        "--neighbor-channels",
        action="store_true",
        help="Encode previous/current/next bone-window slices as model RGB channels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = FractureDatasetConfig(
        fold=args.fold,
        negative_slices_per_study=args.negative_slices,
        positive_extra_negative_slices=args.positive_extra_negatives,
        positive_context_radius=args.context_radius,
        positive_slice_repeat=args.positive_repeat,
        neighbor_channels=args.neighbor_channels,
        overwrite=args.overwrite,
    )
    summary = FractureDatasetV2Builder(
        args.raw_dicom, args.raw_json, args.output, config
    ).build()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
