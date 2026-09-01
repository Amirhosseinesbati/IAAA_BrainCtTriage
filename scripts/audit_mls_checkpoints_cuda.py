"""Resume-safe orchestration for strict CUDA-only MLS checkpoint audits.

Each candidate is evaluated by ``evaluate_mls_multitask_checkpoint.py``.  The
child evaluator owns model loading/inference and rejects CPU fallback.  This
wrapper only records durable progress and skips candidates whose aggregate
metrics already prove a complete 67-study evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=CHECKPOINT")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not LABEL_PATTERN.fullmatch(label):
        raise argparse.ArgumentTypeError(f"unsafe candidate label: {label!r}")
    checkpoint = Path(raw_path).expanduser()
    return label, checkpoint


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _complete_metrics(path: Path, expected_studies: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("n_studies", -1)) == expected_studies
        and int(payload.get("failures", -1)) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="Repeat LABEL=CHECKPOINT for every frozen checkpoint.",
    )
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--expected-studies", type=int, default=67)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.candidate]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS audit found no available GPU")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "audit_status.json"
    status = {
        "schema_version": 1,
        "state": "running",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "cuda_device": torch.cuda.get_device_name(0),
        "fold": args.fold,
        "expected_studies": args.expected_studies,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "candidates": {},
    }
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_utc"] = previous.get("started_utc", status["started_utc"])
        status["candidates"] = previous.get("candidates", {})
    _atomic_json(status_path, status)

    evaluator = PROJECT_ROOT / "scripts" / "evaluate_mls_multitask_checkpoint.py"
    for label, raw_checkpoint in args.candidate:
        checkpoint = raw_checkpoint
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output_dir = output_root / label
        metrics_path = output_dir / "metrics.json"
        if _complete_metrics(metrics_path, args.expected_studies):
            status["candidates"][label] = {
                "state": "completed",
                "checkpoint": str(checkpoint),
                "metrics": str(metrics_path),
                "resumed": True,
            }
            _atomic_json(status_path, status)
            continue

        candidate_state = {
            "state": "running",
            "checkpoint": str(checkpoint),
            "metrics": str(metrics_path),
            "started_utc": _utc_now(),
            "finished_utc": None,
            "exit_code": None,
            "resumed": (output_dir / "study_slice_predictions.csv").is_file(),
        }
        status["candidates"][label] = candidate_state
        _atomic_json(status_path, status)
        command = [
            sys.executable,
            str(evaluator),
            "--checkpoint",
            str(checkpoint),
            "--fold",
            str(args.fold),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        candidate_state["exit_code"] = int(result.returncode)
        candidate_state["finished_utc"] = _utc_now()
        candidate_state["state"] = (
            "completed"
            if result.returncode == 0 and _complete_metrics(metrics_path, args.expected_studies)
            else "failed"
        )
        _atomic_json(status_path, status)
        if candidate_state["state"] != "completed":
            status["state"] = "failed"
            status["finished_utc"] = _utc_now()
            _atomic_json(status_path, status)
            return int(result.returncode or 1)

    status["state"] = "completed"
    status["finished_utc"] = _utc_now()
    _atomic_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
