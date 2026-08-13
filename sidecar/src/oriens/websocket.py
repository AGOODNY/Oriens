"""小型、可取消的 RFC 6455 客户端，仅供阶段 3 百炼 WSS 适配器使用。"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
from threading import Event, Lock
from typing import Any
from urllib.parse import urlsplit


class WebSocketError(RuntimeError):
    """不包含 URL、请求头或凭据的安全网络错误。"""


class WebSocketClosed(WebSocketError):
    pass


class StandardWebSocket:
    """同步 WSS 传输；业务适配器在自己的 Worker 线程中使用。"""

    def __init__(self, url: str, headers: dict[str, str], timeout: float) -> None:
        self._url = url
        self._headers = dict(headers)
        self._timeout = timeout
        self._socket: socket.socket | ssl.SSLSocket | None = None
        self._receive_buffer = bytearray()
        self._send_lock = Lock()

    def connect(self) -> None:
        parsed = urlsplit(self._url)
        if parsed.scheme != "wss" or not parsed.hostname:
            raise WebSocketError("语音服务端点必须使用有效的 wss 地址")
        port = parsed.port or 443
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            raw = socket.create_connection((parsed.hostname, port), self._timeout)
            self._socket = raw
            wrapped = ssl.create_default_context().wrap_socket(
                raw, server_hostname=parsed.hostname
            )
            # 握手期间也暴露给 close()，使程序退出/取消能中断正在建立的连接。
            self._socket = wrapped
            wrapped.settimeout(self._timeout)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            lines = [
                f"GET {path} HTTP/1.1",
                f"Host: {parsed.hostname}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {key}",
                "Sec-WebSocket-Version: 13",
            ]
            lines.extend(f"{name}: {value}" for name, value in self._headers.items())
            wrapped.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
            response, remainder = self._read_http_headers(wrapped)
            status = response[0].split(" ", 2)
            if len(status) < 2 or status[1] != "101":
                code = status[1] if len(status) >= 2 and status[1].isdigit() else "未知"
                self.close()
                raise WebSocketError(f"语音服务拒绝建立连接（HTTP {code}）")
            headers = {}
            for line in response[1:]:
                if ":" in line:
                    name, value = line.split(":", 1)
                    headers[name.lower().strip()] = value.strip()
            expected = base64.b64encode(
                hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            ).decode("ascii")
            if headers.get("sec-websocket-accept") != expected:
                self.close()
                raise WebSocketError("语音服务握手校验失败")
            self._socket = wrapped
            self._receive_buffer.extend(remainder)
            wrapped.settimeout(min(self._timeout, 0.25))
        except WebSocketError:
            self.close()
            raise
        except (OSError, ssl.SSLError, TimeoutError):
            self.close()
            raise WebSocketError("无法连接语音服务") from None

    def send_json(self, value: dict[str, Any]) -> None:
        self.send_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def send_text(self, value: str) -> None:
        self._send_frame(0x1, value.encode("utf-8"))

    def receive(self, cancel: Event) -> str | bytes | None:
        fragments = bytearray()
        fragment_opcode: int | None = None
        while not cancel.is_set():
            try:
                opcode, final, payload = self._receive_frame()
            except socket.timeout:
                return None
            if opcode == 0x8:
                raise WebSocketClosed("语音服务连接已关闭")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x2}:
                fragment_opcode = opcode
                fragments.extend(payload)
            elif opcode == 0x0 and fragment_opcode is not None:
                fragments.extend(payload)
            else:
                continue
            if not final:
                continue
            data = bytes(fragments)
            opcode = fragment_opcode or opcode
            if opcode == 0x1:
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    raise WebSocketError("语音服务返回了无效文本帧") from None
            return data
        raise WebSocketClosed("语音请求已取消")

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        sock = self._socket
        if sock is None:
            raise WebSocketClosed("语音服务尚未连接")
        mask = os.urandom(4)
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(0x80 | length)
        elif length <= 65535:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        with self._send_lock:
            try:
                sock.sendall(bytes(header) + mask + masked)
            except OSError:
                raise WebSocketClosed("语音服务连接已中断") from None

    def _receive_frame(self) -> tuple[int, bool, bytes]:
        first, second = self._recv_exact(2)
        final = bool(first & 0x80)
        opcode = first & 0x0F
        length = second & 0x7F
        masked = bool(second & 0x80)
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        if length > 32 * 1024 * 1024:
            raise WebSocketError("语音服务返回数据过大")
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, final, payload

    def _recv_exact(self, length: int) -> bytes:
        sock = self._socket
        if sock is None:
            raise WebSocketClosed("语音服务连接已关闭")
        data = bytearray()
        if self._receive_buffer:
            take = min(length, len(self._receive_buffer))
            data.extend(self._receive_buffer[:take])
            del self._receive_buffer[:take]
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise WebSocketClosed("语音服务连接已关闭")
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    def _read_http_headers(sock: ssl.SSLSocket) -> tuple[list[str], bytes]:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise WebSocketError("语音服务握手未完成")
            data.extend(chunk)
            if len(data) > 65536:
                raise WebSocketError("语音服务握手响应过大")
        try:
            headers, remainder = data.split(b"\r\n\r\n", 1)
            return headers.decode("iso-8859-1").split("\r\n"), bytes(remainder)
        except UnicodeDecodeError:
            raise WebSocketError("语音服务握手响应无效") from None
