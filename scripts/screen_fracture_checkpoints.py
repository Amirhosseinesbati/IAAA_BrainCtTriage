"""Resume-safe study-level screening of all saved fracture checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPOCH_PATTERN = re.compile(r"epoch(\d+)\.pt$")


def discover_checkpoints(weights_dir: Path) -> list[Path]:
    """Return periodic checkpoints first, then best/last, without assuming epochs."""
    candidates = [path for path in weights_dir.glob("*.pt") if path.is_file()]

    def order(path: Path) -> tuple[int, int, str]:
        match = EPOCH_PATTERN.fullmatch(path.name)
        if match:
            return (0, int(match.group(1)), path.name)
        rank = {"best.pt": 0, "last.pt": 1}.get(path.name, 2)
        return (1, rank, path.name)

    return sorted(candidates, key=order)


def _pooling_auc(payload: dict[str, object], method: str) -> float | None:
    pooling = payload.get("pooling")
    if not isinstance(pooling, dict):
        return None
    values = pooling.get(method)
    if not isinstance(values, dict):
        return None
    value = values.get("auc")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args.weights_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints in {args.weights_dir}")
    if not (args.dataset / "manifest.csv").is_file():
        raise FileNotFoundError(args.dataset / "manifest.csv")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        checkpoint_output = args.output / checkpoint.stem
        metrics_path = checkpoint_output / "metrics.json"
        if args.overwrite or not metrics_path.is_file():
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_fracture_detector_v2.py"),
                "--checkpoint", str(checkpoint),
                "--dataset", str(args.dataset),
                "--output", str(checkpoint_output),
                "--device", args.device,
                "--image-size", str(args.image_size),
                "--batch-size", str(args.batch_size),
                "--confidence", str(args.confidence),
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {"checkpoint": checkpoint.name}
        for method in (
            "max",
            "adjacent_pair",
            "noisy_or",
            "top2_mean",
            "top3_mean",
            "top5_mean",
            "window3_mean",
        ):
            row[f"{method}_auc"] = _pooling_auc(payload, method)
        rows.append(row)

    fieldnames = list(rows[0])
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps({"checkpoints": rows}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"screened": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
