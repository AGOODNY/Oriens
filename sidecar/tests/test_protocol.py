from __future__ import annotations

import unittest

from oriens.protocol import EventParseError, parse_event_line


VALID_JSON = (
    '{"schema_version":1,"seq":1,"run_id":"seed:0",'
    '"type":"run_started","game_frame":0,"context":{},"payload":{}}'
)


class ProtocolTests(unittest.TestCase):
    def test_parses_prefixed_game_log_line(self) -> None:
        event = parse_event_line(f"[INFO] - Lua Debug: [ORIENS_EVENT]{VALID_JSON}\n")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.type, "run_started")
        self.assertEqual(event.seq, 1)

    def test_parses_plain_recording_line(self) -> None:
        event = parse_event_line(VALID_JSON)
        self.assertIsNotNone(event)

    def test_ignores_unrelated_line(self) -> None:
        self.assertIsNone(parse_event_line("[INFO] - unrelated"))

    def test_rejects_invalid_prefixed_json(self) -> None:
        with self.assertRaises(EventParseError):
            parse_event_line("[ORIENS_EVENT]{not-json}")

    def test_rejects_boolean_sequence(self) -> None:
        with self.assertRaises(EventParseError):
            parse_event_line(VALID_JSON.replace('"seq":1', '"seq":true'))


if __name__ == "__main__":
    unittest.main()

