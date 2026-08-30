"""Render private diagnostic montages from fracture error-analysis rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _tile(path: str, size: int, label: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size + 30, size, 3), dtype=np.uint8)
    canvas[:size] = image
    cv2.putText(
        canvas,
        label[:42],
        (5, size + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--outcome", choices=["FP", "FN"], required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.errors, dtype={"study_id": str})
    frame = frame.loc[frame["outcome"].eq(args.outcome)].head(args.limit)
    if frame.empty:
        raise ValueError(f"No {args.outcome} rows found")
    rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        prefix = (
            f"{row.study_id} {row.driver} margin={float(row.decision_margin):+.3f}"
        )
        rows.append(
            np.concatenate(
                [
                    _tile(row.max_slice_image_path, args.tile_size, prefix + " max"),
                    _tile(
                        row.adjacent_start_image_path,
                        args.tile_size,
                        f"adjacent z={row.adjacent_start_slice_index}",
                    ),
                    _tile(
                        row.adjacent_next_image_path,
                        args.tile_size,
                        f"adjacent z={int(row.adjacent_start_slice_index) + 1}",
                    ),
                ],
                axis=1,
            )
        )
    montage = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise IOError(args.output)
    print(
        {
            "output": str(args.output),
            "outcome": args.outcome,
            "rows": len(rows),
            "shape": list(montage.shape),
        }
    )


if __name__ == "__main__":
    main()
