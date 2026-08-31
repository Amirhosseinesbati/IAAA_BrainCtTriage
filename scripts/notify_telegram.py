"""Send a structured experiment event to the configured Telegram chat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.telegram_notifier import (
    TelegramNotifier,
    format_notification,
)


def _field(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("fields must use KEY=VALUE")
    return key.strip(), item.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        default="info",
        choices=("start", "progress", "checkpoint", "success", "failure", "warning", "info"),
    )
    parser.add_argument("--message", required=True)
    parser.add_argument(
        "--title",
        default="[مسابقه IAAA Brain CT Triage 2026 | تسک ICH]",
    )
    parser.add_argument("--field", action="append", default=[], type=_field)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Report notification failures without failing the parent workflow.",
    )
    args = parser.parse_args()

    text = format_notification(
        args.event,
        args.message,
        title=args.title,
        fields=dict(args.field),
    )
    if args.dry_run:
        print(text)
        return 0

    try:
        notifier = TelegramNotifier.from_environment(args.env_file)
        message_ids = notifier.send_text(text)
    except Exception as exc:
        print(f"Telegram notification error: {exc}", file=sys.stderr)
        return 0 if args.best_effort else 1

    print(f"Telegram notification sent: chunks={len(message_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
