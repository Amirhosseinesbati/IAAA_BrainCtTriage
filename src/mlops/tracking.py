"""Consistent MLflow lifecycle and artifacts for every training task."""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

import yaml
from dotenv import load_dotenv

from src.config import PROJECT_CONFIG, PROJECT_ROOT, config_section, get_experiment_name


_T = TypeVar("_T")
_MLFLOW_CIRCUIT_OPEN_UNTIL = 0.0


def _pending_queue_path() -> Path:
    configured = os.getenv("IAAA_MLFLOW_PENDING_QUEUE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "reports" / "mlflow_pending_events.jsonl"


def _active_run_id() -> str | None:
    try:
        import mlflow

        active = mlflow.active_run()
        return active.info.run_id if active is not None else None
    except Exception:
        return None


def _queue_mlflow_event(
    operation: str,
    payload: Mapping[str, Any],
    error: BaseException | str,
) -> None:
    """Persist a replayable MLflow event without recording credentials."""
    target = _pending_queue_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "run_id": _active_run_id(),
        "payload": _safe(payload),
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "Deferred",
        "error": str(error)[:2000],
    }
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def resilient_mlflow_call(
    operation: str,
    call: Callable[[], _T],
    *,
    payload: Mapping[str, Any] | None = None,
    attempts: int | None = None,
    base_delay_seconds: float | None = None,
) -> _T | None:
    """Retry an MLflow call, then defer it instead of killing training.

    A short circuit breaker avoids repeatedly stalling every epoch during a
    sustained DNS or artifact-store outage.  The queue contains only metrics,
    tags, parameters, and local artifact paths; secret environment values are
    never serialized.
    """
    global _MLFLOW_CIRCUIT_OPEN_UNTIL
    payload = dict(payload or {})
    now = time.monotonic()
    if now < _MLFLOW_CIRCUIT_OPEN_UNTIL:
        _queue_mlflow_event(operation, payload, "MLflow circuit breaker is open")
        return None

    retry_count = attempts if attempts is not None else int(
        os.getenv("IAAA_MLFLOW_RETRY_ATTEMPTS", "3")
    )
    retry_count = max(1, retry_count)
    delay = base_delay_seconds if base_delay_seconds is not None else float(
        os.getenv("IAAA_MLFLOW_RETRY_BASE_SECONDS", "1")
    )
    last_error: BaseException | None = None
    for attempt in range(1, retry_count + 1):
        try:
            result = call()
            _MLFLOW_CIRCUIT_OPEN_UNTIL = 0.0
            return result
        except Exception as exc:  # Network/client failures must not kill CUDA work.
            last_error = exc
            if attempt < retry_count:
                time.sleep(max(0.0, delay) * (2 ** (attempt - 1)))

    cooldown = float(os.getenv("IAAA_MLFLOW_CIRCUIT_SECONDS", "300"))
    _MLFLOW_CIRCUIT_OPEN_UNTIL = time.monotonic() + max(0.0, cooldown)
    assert last_error is not None
    _queue_mlflow_event(operation, payload, last_error)
    print(
        f"MLflow operation deferred after {retry_count} attempt(s): "
        f"{operation}: {type(last_error).__name__}: {last_error}",
        file=sys.stderr,
        flush=True,
    )
    return None


def log_metrics_resilient(metrics: Mapping[str, float], *, step: int) -> bool:
    import mlflow

    clean = {str(key): float(value) for key, value in metrics.items()}
    result = resilient_mlflow_call(
        "log_metrics",
        lambda: (mlflow.log_metrics(clean, step=step), True)[1],
        payload={"metrics": clean, "step": int(step)},
    )
    return result is not None


def log_artifact_resilient(local_path: str | Path, *, artifact_path: str) -> bool:
    import mlflow

    source = str(Path(local_path).resolve())
    result = resilient_mlflow_call(
        "log_artifact",
        lambda: (mlflow.log_artifact(source, artifact_path=artifact_path), True)[1],
        payload={"local_path": source, "artifact_path": artifact_path},
    )
    return result is not None


def configure_tracking_environment() -> None:
    """Map local ``.env`` DagsHub settings to standard MLflow variables.

    Vast already exports the standard variables. Local CLI runs previously
    loaded ``.env`` only through Streamlit and could silently write to a local
    ``mlruns`` directory. Credential values are never printed or logged.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    mappings = {
        "MLFLOW_TRACKING_URI": "DAGSHUB_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME": "DAGSHUB_REPO_OWNER",
        "MLFLOW_TRACKING_PASSWORD": "DAGSHUB_USER_TOKEN",
        "MLFLOW_S3_ENDPOINT_URL": "DAGSHUB_REPO_ENDPOINT",
        "AWS_ACCESS_KEY_ID": "DAGSHUB_USER_TOKEN",
        "AWS_SECRET_ACCESS_KEY": "DAGSHUB_USER_TOKEN",
    }
    for target, source in mappings.items():
        value = os.getenv(source, "").strip()
        if value:
            os.environ.setdefault(target, value)
    if os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    # MLflow 3 prints a flag emoji when it closes a run.  Windows terminals
    # commonly expose a cp1252 stream, which can turn a successfully completed
    # training run into a UnicodeEncodeError during context-manager teardown.
    # Reconfigure only the console encoding; this does not affect model compute.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


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


def context_from_environment(
    task_key: str,
    default_run_name: str,
    run_config: Mapping[str, Any],
    *,
    strategy: str | None = None,
) -> ExperimentContext:
    """Build context from UI/Vast metadata without overriding task ownership."""
    raw_tags = os.getenv("IAAA_RUN_TAGS_JSON", "{}")
    try:
        tags = json.loads(raw_tags)
        if not isinstance(tags, dict):
            raise TypeError
    except (json.JSONDecodeError, TypeError):
        tags = {"invalid_tags_payload": raw_tags[:250]}
    return ExperimentContext(
        task_key=task_key,
        run_name=os.getenv("IAAA_RUN_NAME", default_run_name).strip() or default_run_name,
        run_config=run_config,
        strategy=strategy,
        tags=tags,
        notes=os.getenv("IAAA_RUN_NOTES", ""),
    )


@contextmanager
def experiment_run(context: ExperimentContext) -> Iterator[Any]:
    import mlflow
    configure_tracking_environment()
    mlflow.set_experiment(context.experiment_name)
    run = mlflow.start_run(
        run_name=context.run_name,
        log_system_metrics=bool(config_section("mlflow", "log_system_metrics")),
    )
    try:
        tags = {
            "task": context.task_key,
            "strategy": context.strategy or "",
            "source": os.getenv("IAAA_RUN_SOURCE", "local"),
            **_git_metadata(),
            **{str(key): str(value) for key, value in context.tags.items()},
        }
        resilient_mlflow_call(
            "set_tags",
            lambda: (mlflow.set_tags(tags), True)[1],
            payload={"tags": tags},
        )
        if context.notes:
            note = context.notes[:5000]
            resilient_mlflow_call(
                "set_tags",
                lambda: (mlflow.set_tag("mlflow.note.content", note), True)[1],
                payload={"tags": {"mlflow.note.content": note}},
            )
        flat_params = flatten_mapping(context.run_config, "config")
        clean_params = {
            key[:250]: str(value)[:500] for key, value in flat_params.items()
        }
        resilient_mlflow_call(
            "log_params",
            lambda: (mlflow.log_params(clean_params), True)[1],
            payload={"params": clean_params},
        )
        path = config_section("mlflow", "artifact_paths", "config")
        resilient_mlflow_call(
            "log_payload",
            lambda: (_log_payload(
                context.run_config, "resolved_run_config.yaml", path, yaml_format=True,
            ), True)[1],
            payload={
                "content": _safe(context.run_config),
                "filename": "resolved_run_config.yaml",
                "artifact_path": path,
                "yaml_format": True,
            },
        )
        resilient_mlflow_call(
            "log_payload",
            lambda: (_log_payload(
                PROJECT_CONFIG, "project_config.yaml", path, yaml_format=True,
            ), True)[1],
            payload={
                "content": _safe(PROJECT_CONFIG),
                "filename": "project_config.yaml",
                "artifact_path": path,
                "yaml_format": True,
            },
        )
        if config_section("mlflow", "log_environment"):
            environment = collect_environment()
            environment_path = config_section(
                "mlflow", "artifact_paths", "environment"
            )
            resilient_mlflow_call(
                "log_payload",
                lambda: (_log_payload(
                    environment, "runtime.json", environment_path,
                ), True)[1],
                payload={
                    "content": environment,
                    "filename": "runtime.json",
                    "artifact_path": environment_path,
                    "yaml_format": False,
                },
            )
        yield run
    finally:
        resilient_mlflow_call(
            "log_source_snapshot",
            lambda: (log_source_snapshot(), True)[1],
            payload={"rebuild_from_project": True},
        )
        resilient_mlflow_call(
            "end_run",
            lambda: (mlflow.end_run(), True)[1],
            payload={},
        )


def log_run_summary(summary: Mapping[str, Any], filename: str = "run_summary.json") -> None:
    artifact_path = config_section("mlflow", "artifact_paths", "reports")
    resilient_mlflow_call(
        "log_payload",
        lambda: (_log_payload(summary, filename, artifact_path), True)[1],
        payload={
            "content": _safe(summary),
            "filename": filename,
            "artifact_path": artifact_path,
            "yaml_format": False,
        },
    )
