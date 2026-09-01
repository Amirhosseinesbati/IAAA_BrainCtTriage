"""Launch exactly one durable CUDA-only MLS run inside tmux on Vast.

The launcher is intentionally separate from training. It provides an atomic
run lock and durable status JSON so reconnecting clients never infer liveness
from a broad ``pgrep`` pattern or accidentally start a duplicate run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _resolve_manifest(project_dir: Path, value: str | Path) -> tuple[Path, str]:
    project = project_dir.resolve()
    manifest = Path(value)
    if not manifest.is_absolute():
        manifest = project / manifest
    manifest = manifest.resolve()
    allowed_root = (project / "config" / "experiments").resolve()
    if not manifest.is_relative_to(allowed_root) or manifest.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Manifest must be a YAML file under config/experiments")
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if payload.get("task") != "mls" or payload.get("strategy") != "mls_heatmap":
        raise ValueError("Persistent launcher accepts only MLS heatmap manifests")
    run_name = str(payload.get("run_name", "")).strip()
    if not SESSION_PATTERN.fullmatch(run_name):
        raise ValueError("Manifest run_name must contain only safe filename characters")
    return manifest, run_name


def _git_commit(project_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _worker_command(
    python: str,
    script: Path,
    project_dir: Path,
    manifest: Path,
    artifact_root: Path,
    session: str,
) -> str:
    return shlex.join([
        python,
        str(script),
        "--worker",
        "--project-dir",
        str(project_dir),
        "--manifest",
        str(manifest),
        "--artifact-root",
        str(artifact_root),
        "--session",
        session,
    ])


def _run_worker(args: argparse.Namespace) -> int:
    project_dir = args.project_dir.resolve()
    manifest, run_name = _resolve_manifest(project_dir, args.manifest)
    run_dir = args.artifact_root.resolve() / run_name
    lock_dir = run_dir / "run.lock"
    status_path = run_dir / "status.json"
    log_path = run_dir / "train.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Atomic run lock already exists: {lock_dir}") from exc

    status = {
        "schema_version": 1,
        "run_name": run_name,
        "session": args.session,
        "manifest": str(manifest),
        "git_commit": _git_commit(project_dir),
        "state": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "exit_code": None,
        "log_path": str(log_path),
        "compute_policy": "cuda_only",
        "auto_destroy": False,
    }
    _atomic_json(status_path, status)
    command = [
        "bash",
        str(project_dir / "scripts" / "run_vast_mls_experiment.sh"),
        "--manifest",
        str(manifest),
        "--allow-training",
    ]
    return_code = 1
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(f"\n[{_utc_now()}] durable launcher started\n")
            stream.flush()
            result = subprocess.run(
                command,
                cwd=project_dir,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
            return_code = int(result.returncode)
    except BaseException:
        status["state"] = "launcher_error"
        raise
    finally:
        status["state"] = "completed" if return_code == 0 else status.get("state", "failed")
        if return_code != 0 and status["state"] == "running":
            status["state"] = "failed"
        status["exit_code"] = return_code
        status["finished_utc"] = _utc_now()
        _atomic_json(status_path, status)
        try:
            lock_dir.rmdir()
        except OSError:
            # A non-empty or externally modified lock is preserved for audit.
            pass
    return return_code


def _launch(args: argparse.Namespace) -> int:
    if not SESSION_PATTERN.fullmatch(args.session):
        raise ValueError("Unsafe tmux session name")
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for persistent Vast training")
    project_dir = args.project_dir.resolve()
    manifest, run_name = _resolve_manifest(project_dir, args.manifest)
    run_dir = args.artifact_root.resolve() / run_name
    status_path = run_dir / "status.json"
    lock_dir = run_dir / "run.lock"
    if status_path.exists() or lock_dir.exists():
        raise RuntimeError(
            f"Run already has durable state; inspect instead of restarting: {run_dir}"
        )
    if _tmux_session_exists(args.session):
        raise RuntimeError(f"tmux session already exists: {args.session}")
    command = _worker_command(
        sys.executable,
        Path(__file__).resolve(),
        project_dir,
        manifest,
        args.artifact_root.resolve(),
        args.session,
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.session, command],
        check=True,
        cwd=project_dir,
    )
    if not _tmux_session_exists(args.session):
        raise RuntimeError("tmux did not retain the MLS worker session")
    print(json.dumps({
        "started": True,
        "run_name": run_name,
        "session": args.session,
        "run_dir": str(run_dir),
        "git_commit": _git_commit(project_dir),
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(os.environ.get("IAAA_PROJECT_DIR", "/workspace/IAAA_BrainCtTriage")),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/iaaa_artifacts/logs"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return _run_worker(args) if args.worker else _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
