"""本地长期记忆边界、隐私策略与 SQLite 实现。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Protocol, Sequence
from uuid import uuid4


MEMORY_KINDS = frozenset(
    {"profile", "stable_preference", "guidance_preference", "milestone"}
)
MEMORY_STATUSES = frozenset({"active", "disabled", "pending", "conflicted", "deleted"})
CONFIRMATION_LEVELS = frozenset({"manual", "explicit", "confirmed", "inferred"})
MAX_MEMORY_CHARS = 240


class MemoryStoreError(RuntimeError):
    """可安全展示的记忆服务错误。"""


class MemoryUnavailableError(MemoryStoreError):
    """数据库或迁移不可用。"""


class MemoryValidationError(MemoryStoreError):
    """记忆内容不符合允许的长期信息规则。"""


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    source: str
    excerpt: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: str
    kind: str
    content: str
    status: str
    confidence: float
    confirmation_level: str
    source_summary: str
    source_session_id: str | None
    source_run_id: str | None
    created_at: str
    updated_at: str
    last_used_at: str | None
    evidence: tuple[MemoryEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """结构化召回；内容始终作为不可信偏好数据而非系统指令。"""

    items: tuple[MemoryItem, ...] = ()
    total_chars: int = 0


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    content: str
    source: str
    kind: str = "stable_preference"
    confidence: float = 1.0
    confirmation_level: str = "explicit"
    evidence: str | None = None
    source_session_id: str | None = None
    source_run_id: str | None = None
    topic_key: str | None = None


class MemoryStore(Protocol):
    enabled: bool

    def begin_session(self, session_id: str) -> None: ...
    def end_session(self, session_id: str) -> None: ...
    def recall(
        self, query: str, *, max_items: int = 3, max_chars: int = 360
    ) -> MemoryContext: ...
    def submit_candidates(
        self, candidates: Sequence[MemoryCandidate]
    ) -> tuple[MemoryItem, ...]: ...
    def list_items(self, *, include_deleted: bool = False) -> tuple[MemoryItem, ...]: ...
    def add(self, candidate: MemoryCandidate) -> MemoryItem: ...
    def update(
        self, memory_id: str, *, content: str, kind: str | None = None
    ) -> MemoryItem: ...
    def set_item_enabled(self, memory_id: str, enabled: bool) -> bool: ...
    def delete(self, memory_id: str) -> bool: ...
    def clear_all(self) -> int: ...
    def set_enabled(self, enabled: bool) -> None: ...
    def close(self) -> None: ...


class NullMemoryStore:
    """明确的空实现：不创建文件，不保留会话、问题或候选记忆。"""

    enabled = False

    def begin_session(self, session_id: str) -> None:
        return None

    def end_session(self, session_id: str) -> None:
        return None

    def recall(
        self, query: str, *, max_items: int = 3, max_chars: int = 360
    ) -> MemoryContext:
        return MemoryContext()

    def submit_candidates(
        self, candidates: Sequence[MemoryCandidate]
    ) -> tuple[MemoryItem, ...]:
        return ()

    def list_items(self, *, include_deleted: bool = False) -> tuple[MemoryItem, ...]:
        return ()

    def add(self, candidate: MemoryCandidate) -> MemoryItem:
        raise MemoryUnavailableError("长期记忆尚未启用。")

    def update(
        self, memory_id: str, *, content: str, kind: str | None = None
    ) -> MemoryItem:
        raise MemoryUnavailableError("长期记忆尚未启用。")

    def set_item_enabled(self, memory_id: str, enabled: bool) -> bool:
        return False

    def delete(self, memory_id: str) -> bool:
        return False

    def clear_all(self) -> int:
        return 0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = False

    def close(self) -> None:
        return None


_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind IN ('profile','stable_preference','guidance_preference','milestone')),
            topic_key TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('active','disabled','pending','conflicted','deleted')),
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            confirmation_level TEXT NOT NULL CHECK(confirmation_level IN ('manual','explicit','confirmed','inferred')),
            source_summary TEXT NOT NULL,
            source_session_id TEXT,
            source_run_id TEXT,
            supersedes_id TEXT REFERENCES memories(id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_used_at TEXT
        )
        """,
        """
        CREATE TABLE memory_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            excerpt TEXT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX memories_status_updated_idx ON memories(status, updated_at DESC)",
        "CREATE INDEX memories_topic_idx ON memories(kind, topic_key, updated_at DESC)",
        "CREATE INDEX memory_evidence_memory_idx ON memory_evidence(memory_id, id)",
    ),
)


class SQLiteMemoryStore:
    """单连接本地记忆库；所有迁移和写入都使用显式事务。"""

    enabled = True

    def __init__(
        self,
        memory_dir: Path,
        *,
        busy_timeout_ms: int = 3000,
        migrations: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        self.memory_dir = memory_dir.resolve()
        self.database_path = self.memory_dir / "memory.sqlite3"
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self._active_sessions: set[str] = set()
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.database_path,
                timeout=max(0.1, busy_timeout_ms / 1000),
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {max(100, int(busy_timeout_ms))}")
            self._connection = connection
            self._migrate(migrations or _MIGRATIONS)
            self._validate_schema()
        except Exception as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self.enabled = False
            raise MemoryUnavailableError("本地长期记忆暂时不可用。") from exc

    @property
    def closed(self) -> bool:
        return self._connection is None

    def begin_session(self, session_id: str) -> None:
        if session_id:
            with self._lock:
                self._require_connection()
                self._active_sessions.add(session_id)

    def end_session(self, session_id: str) -> None:
        with self._lock:
            if self._connection is not None:
                self._active_sessions.discard(session_id)

    def recall(
        self, query: str, *, max_items: int = 3, max_chars: int = 360
    ) -> MemoryContext:
        if max_items <= 0 or max_chars <= 0:
            return MemoryContext()
        with self._lock:
            connection = self._require_connection()
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE status = 'active'
                ORDER BY updated_at DESC, id DESC
                LIMIT 100
                """
            ).fetchall()
            ranked = sorted(
                rows,
                key=lambda row: (
                    _recall_score(query, row["content"], row["kind"]),
                    row["updated_at"],
                ),
                reverse=True,
            )
            selected: list[sqlite3.Row] = []
            used = 0
            for row in ranked:
                score = _recall_score(query, row["content"], row["kind"])
                if score <= 0:
                    continue
                length = len(row["content"])
                if length > max_chars - used:
                    continue
                selected.append(row)
                used += length
                if len(selected) >= max_items:
                    break
            if not selected:
                return MemoryContext()
            now = _utc_now()
            self._transaction(
                lambda: connection.executemany(
                    "UPDATE memories SET last_used_at = ? WHERE id = ?",
                    ((now, row["id"]) for row in selected),
                )
            )
            return MemoryContext(
                tuple(self._item_from_row(row) for row in selected), used
            )

    def submit_candidates(
        self, candidates: Sequence[MemoryCandidate]
    ) -> tuple[MemoryItem, ...]:
        stored: list[MemoryItem] = []
        for candidate in candidates:
            try:
                stored.append(self.add(candidate))
            except MemoryValidationError:
                continue
        return tuple(stored)

    def list_items(self, *, include_deleted: bool = False) -> tuple[MemoryItem, ...]:
        with self._lock:
            connection = self._require_connection()
            where = "" if include_deleted else "WHERE status != 'deleted'"
            rows = connection.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC, id DESC"
            ).fetchall()
            return tuple(self._item_from_row(row) for row in rows)

    def add(self, candidate: MemoryCandidate) -> MemoryItem:
        validated = _validate_candidate(candidate)
        with self._lock:
            connection = self._require_connection()
            existing = connection.execute(
                """
                SELECT * FROM memories
                WHERE kind = ? AND topic_key = ? AND status != 'deleted'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (validated.kind, _topic_key(validated)),
            ).fetchone()
            normalized = _normalize(validated.content)
            now = _utc_now()
            if existing is not None and existing["normalized_content"] == normalized:
                self._transaction(
                    lambda: self._refresh_existing(connection, existing["id"], validated, now)
                )
                return self._get(existing["id"])

            memory_id = uuid4().hex
            status = _candidate_status(validated)
            supersedes_id = existing["id"] if existing is not None else None

            def write() -> None:
                if existing is not None and existing["status"] in {"active", "disabled", "pending"}:
                    connection.execute(
                        "UPDATE memories SET status = 'conflicted', updated_at = ? WHERE id = ?",
                        (now, existing["id"]),
                    )
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, kind, topic_key, content, normalized_content, status,
                        confidence, confirmation_level, source_summary,
                        source_session_id, source_run_id, supersedes_id,
                        created_at, updated_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        memory_id, validated.kind, _topic_key(validated), validated.content,
                        normalized, status, validated.confidence,
                        validated.confirmation_level, validated.source,
                        validated.source_session_id, validated.source_run_id,
                        supersedes_id, now, now,
                    ),
                )
                self._insert_evidence(connection, memory_id, validated, now)

            self._transaction(write)
            return self._get(memory_id)

    def update(
        self, memory_id: str, *, content: str, kind: str | None = None
    ) -> MemoryItem:
        with self._lock:
            connection = self._require_connection()
            existing = connection.execute(
                "SELECT * FROM memories WHERE id = ? AND status != 'deleted'", (memory_id,)
            ).fetchone()
            if existing is None:
                raise MemoryValidationError("要纠正的记忆不存在。")
            candidate = _validate_candidate(
                MemoryCandidate(
                    content=content,
                    source="用户在记忆管理中纠正",
                    kind=kind or existing["kind"],
                    confidence=1.0,
                    confirmation_level="manual",
                    evidence="用户手动纠正现有记忆",
                    source_session_id=existing["source_session_id"],
                    source_run_id=existing["source_run_id"],
                    topic_key=existing["topic_key"],
                )
            )
            now = _utc_now()

            def write() -> None:
                connection.execute(
                    """
                    UPDATE memories
                    SET kind = ?, content = ?, normalized_content = ?, status = 'active',
                        confidence = 1.0, confirmation_level = 'manual',
                        source_summary = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (candidate.kind, candidate.content, _normalize(candidate.content),
                     candidate.source, now, memory_id),
                )
                self._insert_evidence(connection, memory_id, candidate, now)

            self._transaction(write)
            return self._get(memory_id)

    def set_item_enabled(self, memory_id: str, enabled: bool) -> bool:
        with self._lock:
            connection = self._require_connection()
            status = "active" if enabled else "disabled"
            now = _utc_now()
            changed = 0

            def write() -> None:
                nonlocal changed
                cursor = connection.execute(
                    """
                    UPDATE memories SET status = ?,
                        confirmation_level = CASE WHEN ? THEN 'confirmed' ELSE confirmation_level END,
                        confidence = CASE WHEN ? THEN MAX(confidence, 0.8) ELSE confidence END,
                        updated_at = ?
                    WHERE id = ? AND status IN ('active','disabled','pending')
                    """,
                    (status, enabled, enabled, now, memory_id),
                )
                changed = cursor.rowcount

            self._transaction(write)
            return bool(changed)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            connection = self._require_connection()
            changed = 0

            def write() -> None:
                nonlocal changed
                cursor = connection.execute(
                    "DELETE FROM memories WHERE id = ?",
                    (memory_id,),
                )
                changed = cursor.rowcount

            self._transaction(write)
            return bool(changed)

    def clear_all(self) -> int:
        with self._lock:
            connection = self._require_connection()
            changed = 0

            def write() -> None:
                nonlocal changed
                cursor = connection.execute(
                    "DELETE FROM memories",
                )
                changed = cursor.rowcount

            self._transaction(write)
            return changed

    def set_enabled(self, enabled: bool) -> None:
        # 全局开关由配置白名单保存并在下次启动时重新装配，避免热替换共享连接。
        self.enabled = bool(enabled) and self._connection is not None

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            if connection is None:
                return
            self.enabled = False
            self._active_sessions.clear()
            connection.close()
            self._connection = None

    def _migrate(self, migrations: tuple[tuple[str, ...], ...]) -> None:
        connection = self._require_connection()
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        current = 0
        if exists:
            row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
            current = int(row[0])
        if current > len(migrations):
            raise MemoryUnavailableError("记忆数据库版本高于当前程序支持范围。")
        if current == len(migrations):
            return

        def apply() -> None:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            for version, statements in enumerate(migrations[current:], start=current + 1):
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (version, _utc_now()),
                )

        self._transaction(apply)

    def _validate_schema(self) -> None:
        connection = self._require_connection()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise MemoryUnavailableError("记忆数据库完整性检查失败。")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not {"schema_version", "memories", "memory_evidence"} <= tables:
            raise MemoryUnavailableError("记忆数据库结构不完整。")

    def _transaction(self, operation) -> None:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            operation()
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def _get(self, memory_id: str) -> MemoryItem:
        connection = self._require_connection()
        row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise MemoryValidationError("记忆不存在。")
        return self._item_from_row(row)

    def _item_from_row(self, row: sqlite3.Row) -> MemoryItem:
        connection = self._require_connection()
        evidence_rows = connection.execute(
            "SELECT source, excerpt, created_at FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        return MemoryItem(
            id=row["id"], kind=row["kind"], content=row["content"], status=row["status"],
            confidence=float(row["confidence"]), confirmation_level=row["confirmation_level"],
            source_summary=row["source_summary"], source_session_id=row["source_session_id"],
            source_run_id=row["source_run_id"], created_at=row["created_at"],
            updated_at=row["updated_at"], last_used_at=row["last_used_at"],
            evidence=tuple(
                MemoryEvidence(item["source"], item["excerpt"], item["created_at"])
                for item in evidence_rows
            ),
        )

    def _refresh_existing(
        self, connection: sqlite3.Connection, memory_id: str,
        candidate: MemoryCandidate, now: str,
    ) -> None:
        status = _candidate_status(candidate)
        connection.execute(
            """
            UPDATE memories SET status = ?, confidence = MAX(confidence, ?),
                confirmation_level = ?, source_summary = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, candidate.confidence, candidate.confirmation_level,
             candidate.source, now, memory_id),
        )
        self._insert_evidence(connection, memory_id, candidate, now)

    @staticmethod
    def _insert_evidence(
        connection: sqlite3.Connection, memory_id: str,
        candidate: MemoryCandidate, now: str,
    ) -> None:
        connection.execute(
            "INSERT INTO memory_evidence(memory_id, source, excerpt, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, candidate.source, candidate.evidence, now),
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise MemoryUnavailableError("本地长期记忆已经关闭。")
        return self._connection


def extract_explicit_candidates(
    text: str, *, session_id: str | None = None, run_id: str | None = None
) -> tuple[MemoryCandidate, ...]:
    """只从少量明确句式提取确定性候选，不调用模型、不保存整段对话。"""

    value = " ".join(text.strip().split())
    if not value or len(value) > 300 or _looks_unsafe(value):
        return ()
    candidates: list[MemoryCandidate] = []
    patterns = (
        (r"(?:以后)?请叫我[“\"']?([^，。！？!?；;]{1,24})", "profile", "profile:preferred_name", "称呼偏好：{}"),
        (r"(?:以后)?(?:请)?(?:少|不要|别)(?:再)?提醒我?([^，。！？!?；;]{1,60})", "guidance_preference", "guidance:reminder", "提示偏好：少提醒{}"),
        (r"(?:以后)?(?:请)?(?:解释|讲)(?:得)?(详细|简短|简单|深入)(?:一点|些)?", "guidance_preference", "guidance:explanation_depth", "解释深度偏好：{}"),
        (r"我(?:很)?喜欢[“\"']?([^，。！？!?；;]{1,60})", "stable_preference", None, "喜欢{}"),
    )
    for pattern, kind, topic, template in patterns:
        match = re.search(pattern, value)
        if match:
            detail = match.group(1).strip(" “\"'”")
            if detail:
                candidates.append(
                    MemoryCandidate(
                        content=template.format(detail), source="玩家明确表达",
                        kind=kind, confidence=1.0, confirmation_level="explicit",
                        evidence=match.group(0)[:120], source_session_id=session_id,
                        source_run_id=run_id, topic_key=topic,
                    )
                )
    return tuple(candidates)


def _validate_candidate(candidate: MemoryCandidate) -> MemoryCandidate:
    content = " ".join(candidate.content.strip().split())
    source = " ".join(candidate.source.strip().split())
    if candidate.kind not in MEMORY_KINDS:
        raise MemoryValidationError("不支持这种长期记忆类型。")
    if candidate.confirmation_level not in CONFIRMATION_LEVELS:
        raise MemoryValidationError("记忆确认级别无效。")
    if not 0 <= candidate.confidence <= 1:
        raise MemoryValidationError("记忆置信度无效。")
    if not content or len(content) > MAX_MEMORY_CHARS:
        raise MemoryValidationError("记忆内容应为 1–240 个字符。")
    if not source or len(source) > 120:
        raise MemoryValidationError("记忆来源摘要无效。")
    if _looks_unsafe(content) or (candidate.evidence and _looks_unsafe(candidate.evidence)):
        raise MemoryValidationError("该内容不适合保存为长期记忆。")
    return MemoryCandidate(
        content=content, source=source, kind=candidate.kind,
        confidence=float(candidate.confidence),
        confirmation_level=candidate.confirmation_level,
        evidence=candidate.evidence[:160] if candidate.evidence else None,
        source_session_id=candidate.source_session_id,
        source_run_id=candidate.source_run_id,
        topic_key=candidate.topic_key,
    )


def _looks_unsafe(value: str) -> bool:
    lowered = value.casefold()
    sensitive = (
        "api key", "apikey", "authorization:", "bearer ", "workspace_id",
        "workspace id", "访问令牌", "授权头", "密钥", "密码", "token=",
        "prompt injection", "忽略之前的指令", "system prompt",
        "```", "<script", "select * from", "-----begin private key-----",
    )
    return any(marker in lowered for marker in sensitive) or "\x00" in value


def _candidate_status(candidate: MemoryCandidate) -> str:
    if candidate.confirmation_level == "inferred" or candidate.confidence < 0.8:
        return "pending"
    return "active"


def _topic_key(candidate: MemoryCandidate) -> str:
    if candidate.topic_key:
        normalized = _normalize(candidate.topic_key)[:120]
        if normalized:
            return normalized
    digest = hashlib.sha256(_normalize(candidate.content).encode("utf-8")).hexdigest()[:24]
    return f"content:{digest}"


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _recall_score(query: str, content: str, kind: str) -> int:
    # 称呼和表达方式可跨主题生效；其他信息必须与当前问题有字符二元组重合。
    base = 4 if kind in {"profile", "guidance_preference"} else 0
    query_key = _normalize(query)
    content_key = _normalize(content)
    if not query_key or not content_key:
        return base
    query_pairs = {query_key[index:index + 2] for index in range(max(1, len(query_key) - 1))}
    content_pairs = {content_key[index:index + 2] for index in range(max(1, len(content_key) - 1))}
    return base + len(query_pairs & content_pairs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
