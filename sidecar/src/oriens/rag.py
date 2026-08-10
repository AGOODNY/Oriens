"""阶段 2 本地混合检索服务。

业务层只依赖 :class:`RagService`；SQLite FTS5 与向量 Worker 都封装在本模块边界内。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Protocol, Sequence


class RagError(RuntimeError):
    """本地 RAG 数据或索引不可用。"""


@dataclass(frozen=True, slots=True)
class RagSource:
    id: str
    title: str
    url: str
    source_type: str
    acquired_on: str
    license_note: str


@dataclass(frozen=True, slots=True)
class RagChunk:
    chunk_id: str
    document_id: str
    entity_type: str
    entity_id: str
    name_zh: str
    name_en: str
    aliases: tuple[str, ...]
    title: str
    text: str
    source: RagSource
    game_version: str
    content_version: str
    checksum: str
    stale: bool


@dataclass(frozen=True, slots=True)
class RagFilters:
    entity_types: tuple[str, ...] = ()
    game_version: str | None = None
    source_types: tuple[str, ...] = ()
    include_stale: bool = False


@dataclass(frozen=True, slots=True)
class RagHit:
    chunk: RagChunk
    methods: tuple[str, ...]
    scores: dict[str, float]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "entity_type": self.chunk.entity_type,
            "entity_id": self.chunk.entity_id,
            "name_zh": self.chunk.name_zh,
            "name_en": self.chunk.name_en,
            "source": asdict(self.chunk.source),
            "methods": list(self.methods),
            "scores": dict(self.scores),
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class RagResult:
    query: str
    hits: tuple[RagHit, ...]
    latency_ms: float
    degraded: bool
    degradation_reason: str | None = None

    @property
    def no_answer(self) -> bool:
        return not self.hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.as_dict() for hit in self.hits],
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
            "no_answer": self.no_answer,
        }


class VectorSearch(Protocol):
    """可替换的向量检索客户端；具体索引不会泄漏给业务层。"""

    def search(self, query: str, limit: int) -> Sequence[tuple[str, float]]: ...

    @property
    def available(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...


@dataclass(slots=True)
class _Candidate:
    chunk: RagChunk
    scores: dict[str, float] = field(default_factory=dict)


class RagService:
    """统一混合检索入口：实体/别名、FTS5/BM25、向量与确定性重排。"""

    def __init__(
        self,
        index_path: Path,
        vector_search: VectorSearch | None = None,
        *,
        vector_min_similarity: float = 0.52,
    ) -> None:
        self._index_path = index_path.resolve()
        self._vector = vector_search
        self._vector_min_similarity = vector_min_similarity
        if not self._index_path.is_file():
            raise RagError(f"本地 RAG 索引不存在：{self._index_path}")

    def close(self) -> None:
        close = getattr(self._vector, "close", None)
        if callable(close):
            close()

    def describe_entity(self, entity_type: str, entity_id: str) -> RagChunk | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT payload FROM chunks WHERE entity_type=? AND entity_id=? "
                "AND stale=0 ORDER BY chunk_id LIMIT 1",
                (entity_type, entity_id),
            ).fetchone()
        return _chunk_from_payload(row[0]) if row else None

    def retrieve(
        self,
        query: str,
        *,
        filters: RagFilters | None = None,
        top_k: int = 5,
    ) -> RagResult:
        started = time.perf_counter()
        text = query.strip()
        if not text or top_k < 1:
            return RagResult(query, (), 0.0, self._vector is None, "空查询")
        active_filters = filters or RagFilters()
        candidates: dict[str, _Candidate] = {}
        with closing(self._connect()) as db:
            exact_rows = self._exact_candidates(db, text, active_filters)
            for chunk_id, quality in exact_rows:
                candidate = self._candidate(db, candidates, chunk_id)
                if candidate is not None:
                    candidate.scores["exact"] = max(
                        candidate.scores.get("exact", 0.0), quality
                    )
            fts_rows = (
                []
                if not exact_rows and _looks_like_exact_id(text)
                else self._fts_candidates(db, text, active_filters, max(top_k * 4, 12))
            )
            for rank, chunk_id in enumerate(fts_rows):
                candidate = self._candidate(db, candidates, chunk_id)
                if candidate is not None:
                    candidate.scores["bm25"] = max(
                        candidate.scores.get("bm25", 0.0), 1.0 / (rank + 1)
                    )

        degraded = self._vector is None or not self._vector.available
        reason = "向量 Worker 未配置，已使用关键词检索"
        if self._vector is not None:
            reason = self._vector.unavailable_reason
            if self._vector.available and not (
                _looks_like_exact_id(text) and not exact_rows
            ):
                try:
                    vector_rows = self._vector.search(text, max(top_k * 4, 12))
                except Exception:
                    degraded = True
                    reason = "向量 Worker 查询失败，已使用关键词检索"
                else:
                    with closing(self._connect()) as db:
                        for chunk_id, similarity in vector_rows:
                            candidate = self._candidate(db, candidates, chunk_id)
                            if candidate is None or not _passes(candidate.chunk, active_filters):
                                continue
                            candidate.scores["vector"] = max(
                                candidate.scores.get("vector", 0.0),
                                max(0.0, min(1.0, similarity)),
                            )

        hits: list[RagHit] = []
        for candidate in candidates.values():
            if not _passes(candidate.chunk, active_filters):
                continue
            scores = candidate.scores
            exact = scores.get("exact", 0.0)
            lexical = scores.get("bm25", 0.0)
            vector = scores.get("vector", 0.0)
            if set(scores) == {"vector"} and vector < self._vector_min_similarity:
                continue
            score = max(exact, 0.65 * lexical, 0.60 * vector)
            if len(scores) > 1:
                score = min(1.0, score + 0.05 * (len(scores) - 1))
            if score < 0.12:
                continue
            methods = tuple(
                method for method in ("exact", "bm25", "vector") if method in scores
            )
            hits.append(RagHit(candidate.chunk, methods, dict(scores), round(score, 6)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.entity_type, hit.chunk.entity_id, hit.chunk.chunk_id))
        elapsed = (time.perf_counter() - started) * 1000
        return RagResult(text, tuple(hits[:top_k]), round(elapsed, 3), degraded, reason)

    def _connect(self) -> sqlite3.Connection:
        try:
            db = sqlite3.connect(self._index_path)
            db.row_factory = sqlite3.Row
            return db
        except sqlite3.Error as exc:
            raise RagError("无法打开本地 RAG 索引") from exc

    @staticmethod
    def _candidate(
        db: sqlite3.Connection,
        candidates: dict[str, _Candidate],
        chunk_id: str,
    ) -> _Candidate | None:
        existing = candidates.get(chunk_id)
        if existing is not None:
            return existing
        row = db.execute("SELECT payload FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None:
            return None
        candidate = _Candidate(_chunk_from_payload(row[0]))
        candidates[chunk_id] = candidate
        return candidate

    @staticmethod
    def _exact_candidates(
        db: sqlite3.Connection, query: str, filters: RagFilters
    ) -> list[tuple[str, float]]:
        normalized = normalize_alias(query)
        if not normalized:
            return []
        rows = db.execute(
            "SELECT a.chunk_id, a.normalized FROM aliases a "
            "JOIN chunks c ON c.chunk_id=a.chunk_id WHERE c.stale=0"
        ).fetchall()
        matches: list[tuple[str, float]] = []
        for row in rows:
            alias = row["normalized"]
            if alias == normalized:
                matches.append((row["chunk_id"], 1.0))
            elif (
                len(alias) >= 2
                and not alias.isdigit()
                and not normalized.isdigit()
                and alias in normalized
            ):
                matches.append((row["chunk_id"], 0.85))
        return matches

    @staticmethod
    def _fts_candidates(
        db: sqlite3.Connection, query: str, filters: RagFilters, limit: int
    ) -> list[str]:
        terms = _fts_terms(query)
        if not terms:
            return []
        clauses = ["1=1"]
        params: list[Any] = [" OR ".join(f'"{term}"' for term in terms)]
        if not filters.include_stale:
            clauses.append("c.stale=0")
        if filters.entity_types:
            clauses.append("c.entity_type IN (%s)" % ",".join("?" * len(filters.entity_types)))
            params.extend(filters.entity_types)
        if filters.game_version:
            clauses.append("c.game_version=?")
            params.append(filters.game_version)
        if filters.source_types:
            clauses.append("c.source_type IN (%s)" % ",".join("?" * len(filters.source_types)))
            params.extend(filters.source_types)
        params.append(limit)
        sql = (
            "SELECT f.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id=f.chunk_id "
            "WHERE chunks_fts MATCH ? AND " + " AND ".join(clauses) +
            " ORDER BY bm25(chunks_fts), f.chunk_id LIMIT ?"
        )
        try:
            return [row[0] for row in db.execute(sql, params)]
        except sqlite3.OperationalError:
            return []


def normalize_alias(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())


def _looks_like_exact_id(value: str) -> bool:
    text = value.strip().casefold()
    return text.isdigit() or re.fullmatch(r"[a-z_]+:\d+", text) is not None


def _fts_terms(query: str) -> list[str]:
    cleaned = re.sub(r"[\"'():*^]", " ", query.casefold()).strip()
    values = [cleaned]
    values.extend(re.findall(r"[0-9a-z]+|[\u3400-\u9fff]{2,}", cleaned))
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in result:
            result.append(value)
    return result[:8]


def _passes(chunk: RagChunk, filters: RagFilters) -> bool:
    if chunk.stale and not filters.include_stale:
        return False
    if filters.entity_types and chunk.entity_type not in filters.entity_types:
        return False
    if filters.game_version and chunk.game_version != filters.game_version:
        return False
    if filters.source_types and chunk.source.source_type not in filters.source_types:
        return False
    return True


def _chunk_from_payload(payload: str) -> RagChunk:
    value = json.loads(payload)
    source = value["source"]
    return RagChunk(
        chunk_id=value["chunk_id"],
        document_id=value["document_id"],
        entity_type=value["entity_type"],
        entity_id=value["entity_id"],
        name_zh=value["name_zh"],
        name_en=value["name_en"],
        aliases=tuple(value["aliases"]),
        title=value["title"],
        text=value["text"],
        source=RagSource(
            id=source["id"],
            title=source["title"],
            url=source["url"],
            source_type=source["type"],
            acquired_on=value["acquired_on"],
            license_note=value["license_note"],
        ),
        game_version=value["game_version"],
        content_version=value["content_version"],
        checksum=value["checksum"],
        stale=bool(value["stale"]),
    )
