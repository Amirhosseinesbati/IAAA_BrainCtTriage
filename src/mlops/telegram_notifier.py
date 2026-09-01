"""Small, dependency-light Telegram notifier for long-running experiments."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


TELEGRAM_TEXT_LIMIT = 4096
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_RETRIES = 3

EVENT_ICONS = {
    "start": "START",
    "progress": "PROGRESS",
    "checkpoint": "CHECKPOINT",
    "success": "SUCCESS",
    "failure": "FAILURE",
    "warning": "WARNING",
    "info": "INFO",
}

PERSIAN_EVENT_LABELS = {
    "start": ("🚀", "آغاز آزمایش"),
    "progress": ("⏳", "گزارش پیشرفت"),
    "checkpoint": ("💾", "checkpoint بهتر"),
    "success": ("✅", "پایان موفق"),
    "failure": ("❌", "توقف به‌علت خطا"),
    "warning": ("⚠️", "هشدار پژوهشی"),
    "info": ("ℹ️", "گزارش پژوهش"),
}

PERSIAN_FIELD_LABELS = {
    "run": "نام اجرا",
    "kind": "نوع اجرا",
    "fold": "فولد",
    "train_studies": "مطالعات آموزش",
    "val_studies": "مطالعات اعتبارسنجی",
    "distill_weight": "وزن distillation",
    "epoch": "ایپاک",
    "macro_f1": "Macro-F1",
    "best_macro_f1": "بهترین Macro-F1",
    "normal_fpr": "نرخ مثبت کاذب نرمال",
    "total_mae_ml": "خطای حجم کل (mL)",
    "best_epoch": "بهترین ایپاک",
    "peak_vram_gb": "اوج VRAM (GB)",
    "duration_min": "مدت اجرا (دقیقه)",
    "sampler_study_balance_power": "توان توازن مطالعه‌ای sampler",
    "sampler_weight_max": "بیشینهٔ وزن sampling",
    "small_ivh_studies": "تعداد مطالعات IVH کوچک (≤۲mL)",
    "small_ivh_dice": "Dice مربوط به IVH کوچک",
    "small_ivh_sensitivity": "حساسیت IVH کوچک در آستانهٔ ۰٫۱mL",
    "error": "نوع خطا",
    "detail": "جزئیات",
}

PERSIAN_FIELD_VALUES = {
    "smoke": "گیت فنی کوچک",
    "full_fold": "آزمایش کامل فولد",
}


class TelegramNotificationError(RuntimeError):
    """Raised when Telegram cannot accept an experiment notification."""


def split_message(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split text on useful boundaries while respecting Telegram's hard limit."""
    if limit < 1:
        raise ValueError("limit must be positive")
    remaining = text.strip()
    if not remaining:
        raise ValueError("message text must not be empty")

    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def format_notification(
    event: str,
    message: str,
    *,
    title: str = "IAAA ICH research",
    fields: Mapping[str, Any] | None = None,
    hostname: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Build a plain-text notification without Telegram markup edge cases."""
    event_name = event.strip().lower()
    label = EVENT_ICONS.get(event_name, event_name.upper() or "INFO")
    created_at = timestamp or datetime.now(timezone.utc)
    lines = [
        f"[{label}] {title.strip() or 'IAAA ICH research'}",
        message.strip(),
        "",
        f"time_utc: {created_at.astimezone(timezone.utc).isoformat(timespec='seconds')}",
        f"host: {hostname or socket.gethostname()}",
    ]
    for key, value in (fields or {}).items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_persian_campaign_notification(
    event: str,
    message: str,
    *,
    title: str = "IAAA Brain CT Triage 2026 — تشخیص خونریزی (ICH)",
    fields: Mapping[str, Any] | None = None,
    hostname: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Render an easy-to-scan Persian research update without markup."""
    event_name = event.strip().lower()
    icon, status = PERSIAN_EVENT_LABELS.get(event_name, ("ℹ️", "گزارش پژوهش"))
    body = message.strip()
    report, analysis, action = body, "", ""
    if "تحلیل کوتاه:" in body:
        report, analysis = body.split("تحلیل کوتاه:", 1)
    if "اقدام بعدی:" in analysis:
        analysis, action = analysis.split("اقدام بعدی:", 1)
    elif "اقدام بعدی:" in report:
        report, action = report.split("اقدام بعدی:", 1)

    lines = [
        f"🧠 {title.strip()}",
        "━━━━━━━━━━━━━━━━━━",
        f"{icon} وضعیت: {status}",
    ]
    if report.strip():
        lines.extend(["", "📝 گزارش", report.strip()])
    if analysis.strip():
        lines.extend(["", "🔎 تحلیل کوتاه", analysis.strip()])
    if action.strip():
        lines.extend(["", "🧭 اقدام بعدی", action.strip()])

    rendered_fields = []
    for key, value in (fields or {}).items():
        label = PERSIAN_FIELD_LABELS.get(str(key), str(key).replace("_", " "))
        rendered_value = PERSIAN_FIELD_VALUES.get(str(value), str(value))
        rendered_fields.append(f"• {label}: {rendered_value}")
    if rendered_fields:
        lines.extend(["", "📊 اطلاعات کلیدی", *rendered_fields])

    created_at = timestamp or datetime.now(timezone.utc)
    iran_time = created_at.astimezone(timezone(timedelta(hours=3, minutes=30)))
    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━",
        f"🕒 زمان ایران: {iran_time:%Y-%m-%d %H:%M:%S}",
        f"🖥 سرور: {hostname or socket.gethostname()}",
    ])
    return "\n".join(lines)


def _redact(value: str, token: str) -> str:
    return value.replace(token, "[REDACTED]") if token else value


@dataclass
class TelegramNotifier:
    token: str
    chat_id: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    opener: Callable[..., Any] = urllib.request.urlopen

    def __post_init__(self) -> None:
        self.token = self.token.strip()
        self.chat_id = self.chat_id.strip()
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is not configured")
        if self.retries < 1:
            raise ValueError("retries must be at least 1")

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "TelegramNotifier":
        if env_file is not None:
            load_dotenv(env_file, override=False)
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    @property
    def endpoint(self) -> str:
        return f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_text(self, text: str) -> list[int | None]:
        """Send all chunks and return their Telegram message identifiers."""
        message_ids: list[int | None] = []
        for chunk in split_message(text):
            message_ids.append(self._send_chunk(chunk))
        return message_ids

    def _send_chunk(self, text: str) -> int | None:
        payload = json.dumps(
            {"chat_id": self.chat_id, "text": text},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not result.get("ok"):
                    raise TelegramNotificationError(
                        str(result.get("description", "Telegram returned ok=false"))
                    )
                message_id = result.get("result", {}).get("message_id")
                return int(message_id) if message_id is not None else None
            except (OSError, ValueError, urllib.error.HTTPError, TelegramNotificationError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))

        detail = _redact(str(last_error or "unknown Telegram error"), self.token)
        raise TelegramNotificationError(
            f"Telegram notification failed after {self.retries} attempts: {detail}"
        ) from last_error
