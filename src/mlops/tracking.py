"""Consistent MLflow lifecycle and artifacts for every training task."""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from src.config import PROJECT_CONFIG, PROJECT_ROOT, config_section, get_experiment_name


def _safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flatten_mapping(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, name))
        elif isinstance(value, (list, tuple, set)):
            flattened[name] = json.dumps(_safe(value), ensure_ascii=False)
        else:
            flattened[name] = _safe(value)
    return flattened


def log_flat_params(values: Mapping[str, Any], prefix: str = "") -> None:
    import mlflow
    params = flatten_mapping(values, prefix)
    mlflow.log_params({key[:250]: str(value)[:500] for key, value in params.items()})


def _git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return "unknown"
    status = run("status", "--porcelain")
    return {
        "git.commit": run("rev-parse", "HEAD"),
        "git.branch": run("branch", "--show-current"),
        "git.dirty": str(status not in ("", "unknown")).lower(),
    }


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
    }
    try:
        import torch
        result.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        })
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        result["torch_probe_error"] = str(exc)
    return result


def build_source_snapshot(output_path: Path) -> dict[str, Any]:
    cfg = config_section("mlflow", "source_snapshot")
    patterns = cfg["exclude_names"]
    max_bytes = int(cfg["max_file_bytes"])
    count = 0
    skipped: list[str] = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in cfg["include"]:
            source = PROJECT_ROOT / item
            if not source.exists():
                continue
            candidates = [source] if source.is_file() else source.rglob("*")
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(PROJECT_ROOT)
                if any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in patterns):
                    continue
                if candidate.stat().st_size > max_bytes:
                    skipped.append(relative.as_posix())
                    continue
                archive.write(candidate, relative.as_posix())
                count += 1
        manifest = {"included_files": count, "skipped_large_files": skipped, **_git_metadata()}
        archive.writestr("snapshot_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def log_source_snapshot() -> None:
    import mlflow
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "project_source.zip"
        build_source_snapshot(target)
        mlflow.log_artifact(str(target), artifact_path=config_section("mlflow", "artifact_paths", "code"))


def _log_payload(payload: Mapping[str, Any], filename: str, artifact_path: str, *, yaml_format: bool = False) -> None:
    import mlflow
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / filename
        if yaml_format:
            content = yaml.safe_dump(_safe(payload), sort_keys=False, allow_unicode=True)
        else:
            content = json.dumps(_safe(payload), indent=2, ensure_ascii=False)
        target.write_text(content, encoding="utf-8")
        mlflow.log_artifact(str(target), artifact_path=artifact_path)


@dataclass(frozen=True)
class ExperimentContext:
    task_key: str
    run_name: str
    run_config: Mapping[str, Any]
    strategy: str | None = None
    tags: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def experiment_name(self) -> str:
        return get_experiment_name(self.task_key)


@contextmanager
def experiment_run(context: ExperimentContext) -> Iterator[Any]:
    import mlflow
    mlflow.set_experiment(context.experiment_name)
    with mlflow.start_run(
        run_name=context.run_name,
        log_system_metrics=bool(config_section("mlflow", "log_system_metrics")),
    ) as run:
        mlflow.set_tags({
            "task": context.task_key,
            "strategy": context.strategy or "",
            "source": os.getenv("IAAA_RUN_SOURCE", "local"),
            **_git_metadata(),
            **{str(key): str(value) for key, value in context.tags.items()},
        })
        if context.notes:
            mlflow.set_tag("mlflow.note.content", context.notes[:5000])
        log_flat_params(context.run_config, "config")
        path = config_section("mlflow", "artifact_paths", "config")
        _log_payload(context.run_config, "resolved_run_config.yaml", path, yaml_format=True)
        _log_payload(PROJECT_CONFIG, "project_config.yaml", path, yaml_format=True)
        if config_section("mlflow", "log_environment"):
            _log_payload(
                collect_environment(), "runtime.json",
                config_section("mlflow", "artifact_paths", "environment"),
            )
        try:
            yield run
        finally:
            log_source_snapshot()


def log_run_summary(summary: Mapping[str, Any], filename: str = "run_summary.json") -> None:
    _log_payload(summary, filename, config_section("mlflow", "artifact_paths", "reports"))
