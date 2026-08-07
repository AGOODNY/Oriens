"""支持截断和文件替换恢复的轮询式日志监听器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class TailPoll:
    lines: tuple[str, ...]
    reopened: bool = False


class LogTailer:
    def __init__(self, path: Path, *, from_start: bool = False) -> None:
        self.path = path
        self.from_start = from_start
        self._stream: TextIO | None = None
        self._identity: tuple[int, int] | None = None
        self._first_open = True

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def poll(self) -> TailPoll:
        if not self.path.exists():
            self.close()
            self._identity = None
            return TailPoll(())

        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        reopened = False

        if self._stream is None:
            self._open(at_end=self._first_open and not self.from_start)
            self._first_open = False
        elif identity != self._identity or stat.st_size < self._stream.tell():
            self.close()
            self._open(at_end=False)
            reopened = True

        assert self._stream is not None
        lines: list[str] = []
        while True:
            position = self._stream.tell()
            line = self._stream.readline()
            if not line:
                break
            if not line.endswith("\n"):
                self._stream.seek(position)
                break
            lines.append(line)
        return TailPoll(tuple(lines), reopened)

    def _open(self, *, at_end: bool) -> None:
        self._stream = self.path.open("r", encoding="utf-8", errors="replace")
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino)
        if at_end:
            self._stream.seek(0, 2)

