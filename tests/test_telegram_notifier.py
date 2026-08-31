from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from src.mlops.telegram_notifier import (
    TelegramNotificationError,
    TelegramNotifier,
    format_notification,
    split_message,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TelegramNotifierTests(unittest.TestCase):
    def test_split_message_respects_limit(self) -> None:
        chunks = split_message("alpha " * 1500, limit=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 200 for chunk in chunks))

    def test_format_notification_is_deterministic_with_explicit_context(self) -> None:
        rendered = format_notification(
            "success",
            "training completed",
            fields={"dice": "0.91"},
            hostname="worker-1",
            timestamp=datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc),
        )
        self.assertIn("[SUCCESS] IAAA ICH research", rendered)
        self.assertIn("time_utc: 2026-08-31T10:30:00+00:00", rendered)
        self.assertIn("host: worker-1", rendered)
        self.assertIn("dice: 0.91", rendered)

    def test_send_text_posts_json_without_markup(self) -> None:
        captured = []

        def opener(request, timeout):
            captured.append((request, timeout))
            return _Response({"ok": True, "result": {"message_id": 42}})

        notifier = TelegramNotifier("secret-token", "123", opener=opener)
        self.assertEqual(notifier.send_text("hello"), [42])
        request, timeout = captured[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body, {"chat_id": "123", "text": "hello"})
        self.assertEqual(timeout, notifier.timeout_seconds)
        self.assertNotIn("parse_mode", body)

    def test_error_redacts_token(self) -> None:
        def opener(_request, timeout):
            self.assertEqual(timeout, notifier.timeout_seconds)
            raise OSError("request to secret-token failed")

        notifier = TelegramNotifier("secret-token", "123", retries=1, opener=opener)
        with self.assertRaises(TelegramNotificationError) as context:
            notifier.send_text("hello")
        self.assertNotIn("secret-token", str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))


if __name__ == "__main__":
    unittest.main()
