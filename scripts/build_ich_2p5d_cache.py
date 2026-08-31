"""CLI for the compact 2.5D multi-window ICH cache."""

from __future__ import annotations

import argparse
import json

from src.strategies.ich_2p5d.cache import build_slice_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="Data/processed/ich_v2/BrainICHPartial")
    parser.add_argument("--output-dir", default="Data/processed/ich_2p5d")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    frame = build_slice_cache(
        args.dataset_dir,
        args.output_dir,
        image_size=args.image_size,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "slices": int(len(frame)),
        "known_slices": int(frame["known"].sum()),
        "positive_slices": int(
            frame.loc[frame["classification_known"] == 1, "any_ich"].sum()
        ),
        "studies": int(frame["study_id"].nunique()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
