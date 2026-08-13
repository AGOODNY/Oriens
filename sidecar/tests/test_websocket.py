from __future__ import annotations

import ssl
from threading import Event
import unittest

from oriens.websocket import StandardWebSocket


class _PartialTlsSocket:
    def __init__(self, values=None) -> None:
        self._values = iter(values or (
            b"\x81",
            ssl.SSLWantReadError(ssl.SSL_ERROR_WANT_READ, "want read"),
            b"\x02ok",
        ))

    def recv(self, _length: int) -> bytes:
        value = next(self._values)
        if isinstance(value, BaseException):
            raise value
        return value


class WebSocketTests(unittest.TestCase):
    def test_ssl_want_read_keeps_partial_frame_for_next_receive(self) -> None:
        transport = StandardWebSocket("wss://example.invalid", {}, 1.0)
        transport._socket = _PartialTlsSocket()  # type: ignore[assignment]

        self.assertIsNone(transport.receive(Event()))
        self.assertEqual(transport.receive(Event()), "ok")

    def test_ssl_want_read_keeps_fragmented_message_for_next_receive(self) -> None:
        transport = StandardWebSocket("wss://example.invalid", {}, 1.0)
        transport._socket = _PartialTlsSocket((
            b"\x01\x01a",
            ssl.SSLWantReadError(ssl.SSL_ERROR_WANT_READ, "want read"),
            b"\x80\x01b",
        ))  # type: ignore[assignment]

        self.assertIsNone(transport.receive(Event()))
        self.assertEqual(transport.receive(Event()), "ab")


if __name__ == "__main__":
    unittest.main()
