from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from src.mlops.telegram_notifier import (
    TelegramNotificationError,
    TelegramNotifier,
    format_notification,
    format_persian_campaign_notification,
    split_message,
)
from src.strategies.ich_v2.operations import campaign_event_enabled


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
    def test_campaign_events_suppress_routine_checkpoints_by_default(self):
        self.assertTrue(campaign_event_enabled("success", "start,success,failure"))
        self.assertFalse(campaign_event_enabled("checkpoint", "start,success,failure"))
        self.assertTrue(campaign_event_enabled("checkpoint", "all"))

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

    def test_persian_campaign_format_has_scannable_sections(self) -> None:
        rendered = format_persian_campaign_notification(
            "success",
            "آموزش تمام شد. تحلیل کوتاه: مدل بهتر شد. اقدام بعدی: ارزیابی کامل.",
            fields={"run": "exp03", "kind": "full_fold", "macro_f1": "0.72"},
            hostname="gpu-1",
            timestamp=datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc),
        )
        self.assertTrue(rendered.startswith("🧠 IAAA Brain CT Triage 2026"))
        self.assertIn("🔎 تحلیل کوتاه\nمدل بهتر شد.", rendered)
        self.assertIn("🧭 اقدام بعدی\nارزیابی کامل.", rendered)
        self.assertIn("• نوع اجرا: آزمایش کامل فولد", rendered)
        self.assertIn("🕒 زمان ایران: 2026-08-31 14:00:00", rendered)

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
