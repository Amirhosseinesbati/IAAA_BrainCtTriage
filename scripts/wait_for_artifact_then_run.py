"""Wait for a completed artifact, then execute one follow-up command."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def wait_for_artifact(
    path: Path,
    *,
    min_bytes: int,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    """Wait until ``path`` exists at the requested size or raise on timeout."""
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    while True:
        try:
            if path.is_file() and path.stat().st_size >= min_bytes:
                return
        except OSError:
            pass
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for artifact: {path}")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--min-bytes", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a follow-up command is required after --")
    if args.min_bytes < 1:
        parser.error("--min-bytes must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    wait_for_artifact(
        args.artifact,
        min_bytes=args.min_bytes,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Artifact ready: {args.artifact}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
