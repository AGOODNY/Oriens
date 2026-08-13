"""阶段 4 长期记忆边界；本模块不实现任何持久化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class MemoryContext:
    items: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    content: str
    source: str


class MemoryStore(Protocol):
    enabled: bool

    def begin_session(self, session_id: str) -> None: ...
    def end_session(self, session_id: str) -> None: ...
    def recall(self, query: str) -> MemoryContext: ...
    def submit_candidates(self, candidates: Sequence[MemoryCandidate]) -> None: ...
    def list_items(self) -> tuple[object, ...]: ...
    def delete(self, memory_id: str) -> bool: ...
    def set_enabled(self, enabled: bool) -> None: ...
    def close(self) -> None: ...


class NullMemoryStore:
    """明确的空实现：不创建文件，不保留会话、问题或候选记忆。"""

    enabled = False

    def begin_session(self, session_id: str) -> None:
        return None

    def end_session(self, session_id: str) -> None:
        return None

    def recall(self, query: str) -> MemoryContext:
        return MemoryContext()

    def submit_candidates(self, candidates: Sequence[MemoryCandidate]) -> None:
        return None

    def list_items(self) -> tuple[object, ...]:
        return ()

    def delete(self, memory_id: str) -> bool:
        return False

    def set_enabled(self, enabled: bool) -> None:
        # 阶段 3.5 不允许把空实现切换成实际记忆。
        self.enabled = False

    def close(self) -> None:
        return None

