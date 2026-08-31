"""Build the corrected all-study ICH-v2 NIfTI cache."""

from __future__ import annotations

import argparse
import logging

from src.strategies.ich_v2.dataset_builder import ICHV2DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", action="append", dest="study_ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    builder = ICHV2DatasetBuilder(output_dir=args.output_dir)
    manifest = builder.build(study_ids=args.study_ids, overwrite=args.overwrite)
    print(manifest.groupby("supervision_type").size().to_string())
    print(f"studies={len(manifest)} unknown_slices={int(manifest['unknown_slices'].sum())}")


if __name__ == "__main__":
    main()
