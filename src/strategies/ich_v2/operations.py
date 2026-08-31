"""Operational helpers shared by ICH-v2 experiments."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.mlops.telegram_notifier import (
    TelegramNotifier,
    format_persian_campaign_notification,
)


CAMPAIGN_TITLE = "IAAA Brain CT Triage 2026 — تشخیص خونریزی (ICH)"
DEFAULT_TELEGRAM_EVENTS = "start,success,failure,warning,info"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def configure_remote_mlflow() -> str:
    """Load minimal DagsHub credentials and reject local tracking stores."""
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
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri or uri.startswith(("file:", "sqlite:")):
        raise RuntimeError("ICH-v2 official runs require a remote MLflow tracking URI")
    mlflow.set_tracking_uri(uri)
    return uri


def campaign_event_enabled(event: str, configured: str | None = None) -> bool:
    """Keep routine per-epoch events quiet unless explicitly requested."""
    raw = (
        configured
        if configured is not None
        else os.getenv("IAAA_TELEGRAM_EVENTS", DEFAULT_TELEGRAM_EVENTS)
    )
    enabled = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return "*" in enabled or "all" in enabled or event.strip().lower() in enabled


def notify_campaign(event: str, message: str, **fields: object) -> None:
    """Best-effort Persian Telegram event with a stable competition prefix."""
    if not campaign_event_enabled(event):
        return
    try:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        notifier = TelegramNotifier.from_environment(PROJECT_ROOT / ".env")
        notifier.send_text(format_persian_campaign_notification(
            event,
            message,
            title=CAMPAIGN_TITLE,
            fields={key: str(value) for key, value in fields.items()},
        ))
    except Exception as exc:
        print(f"Telegram best-effort notification failed: {exc}")
