"""可重复的阶段 2 语料清洗、分块、去重和索引构建。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable, Iterator

from .rag import normalize_alias


class CorpusValidationError(ValueError):
    pass


def build_corpus(source_path: Path, chunks_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("documents"), list):
        raise CorpusValidationError("RAG 语料 schema_version/documents 无效")
    defaults = raw.get("defaults")
    if not isinstance(defaults, dict):
        raise CorpusValidationError("RAG 语料缺少 defaults")
    chunks: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    checksums: dict[str, str] = {}
    for document in raw["documents"]:
        chunk = _normalize_document(document, defaults)
        document_id = chunk["document_id"]
        if document_id in document_ids:
            raise CorpusValidationError(f"文档 ID 重复：{document_id}")
        document_ids.add(document_id)
        checksum = chunk["checksum"]
        duplicate = checksums.get(checksum)
        if duplicate is not None:
            raise CorpusValidationError(f"内容重复：{document_id} 与 {duplicate}")
        checksums[checksum] = document_id
        chunks.append(chunk)
    chunks.sort(key=lambda value: value["chunk_id"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks),
        encoding="utf-8",
        newline="\n",
    )
    corpus_checksum = sha256(chunks_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "corpus_id": raw["corpus_id"],
        "content_version": raw["content_version"],
        "game_version": raw["game_version"],
        "curated_on": raw["curated_on"],
        "document_count": len(document_ids),
        "chunk_count": len(chunks),
        "corpus_checksum": f"sha256:{corpus_checksum}",
        # 清单应可跨机器复现，不记录维护者工作区的绝对路径。
        "source_file": source_path.name,
        "scope": raw["scope"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return chunks


def build_keyword_index(
    chunks: Iterable[dict[str, Any]],
    index_path: Path,
    *,
    corpus_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """流式构建关键词索引；调用方不需要把完整分块集载入内存。"""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=index_path.name + ".", suffix=".tmp", dir=index_path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    db = sqlite3.connect(temporary)
    chunk_count = 0
    content_version = "unknown"
    started = __import__("time").perf_counter()
    try:
        db.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks(
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                game_version TEXT NOT NULL,
                source_type TEXT NOT NULL,
                stale INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX chunks_entity ON chunks(entity_type, entity_id, stale);
            CREATE INDEX chunks_filters ON chunks(game_version, source_type, stale);
            CREATE TABLE aliases(
                normalized TEXT NOT NULL,
                alias TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
                PRIMARY KEY(normalized, chunk_id)
            );
            CREATE INDEX aliases_entity ON aliases(entity_type, entity_id);
            CREATE INDEX aliases_normalized ON aliases(normalized);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED, name_zh, name_en, aliases, title, text,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        for chunk in chunks:
            chunk_count += 1
            content_version = str(chunk.get("content_version") or content_version)
            payload = json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            db.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    chunk["chunk_id"], chunk["document_id"], chunk["entity_type"],
                    chunk["entity_id"], chunk["game_version"], chunk["source"]["type"],
                    int(chunk["stale"]), chunk["checksum"], payload,
                ),
            )
            aliases = _all_aliases(chunk)
            for alias in aliases:
                normalized = normalize_alias(alias)
                if normalized:
                    db.execute(
                        "INSERT OR IGNORE INTO aliases VALUES(?,?,?,?,?)",
                        (normalized, alias, chunk["entity_type"], chunk["entity_id"], chunk["chunk_id"]),
                    )
            db.execute(
                "INSERT INTO chunks_fts VALUES(?,?,?,?,?,?)",
                (chunk["chunk_id"], chunk["name_zh"], chunk["name_en"], " ".join(aliases), chunk["title"], chunk["text"]),
            )
        db.execute("INSERT INTO metadata VALUES('schema_version','1')")
        db.execute("INSERT INTO metadata VALUES('chunk_count',?)", (str(chunk_count),))
        db.execute("INSERT INTO metadata VALUES('content_version',?)", (content_version,))
        for key, value in sorted((corpus_metadata or {}).items()):
            db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, str(value)))
        db.execute("INSERT INTO metadata VALUES('vector_backend','none')")
        db.commit()
    finally:
        db.close()
    temporary.replace(index_path)
    elapsed = (__import__("time").perf_counter() - started) * 1000
    return {
        "chunk_count": chunk_count,
        "build_latency_ms": round(elapsed, 3),
        "index_bytes": index_path.stat().st_size,
    }


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return list(iter_chunks(path))


def iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(
                    f"{path.name}:{line_number} 不是有效分块 JSON"
                ) from exc
            if not isinstance(value, dict):
                raise CorpusValidationError(f"{path.name}:{line_number} 分块必须是对象")
            yield value


def _normalize_document(document: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise CorpusValidationError("文档必须是对象")
    required = (
        "document_id", "entity_type", "entity_id", "name_zh", "name_en", "aliases",
        "title", "text", "source",
    )
    for name in required:
        if name not in document:
            raise CorpusValidationError(f"文档缺少字段：{name}")
    document_id = _stable_id(document["document_id"], "document_id")
    entity_type = _stable_id(document["entity_type"], "entity_type")
    entity_id = _stable_id(document["entity_id"], "entity_id")
    name_zh = _text(document["name_zh"], "name_zh")
    name_en = _text(document["name_en"], "name_en")
    title = _text(document["title"], "title")
    text = _text(document["text"], "text")
    aliases_raw = document["aliases"]
    if not isinstance(aliases_raw, list):
        raise CorpusValidationError(f"{document_id} aliases 必须是数组")
    aliases = sorted({_text(alias, "alias") for alias in aliases_raw}, key=str.casefold)
    source = document["source"]
    if not isinstance(source, dict):
        raise CorpusValidationError(f"{document_id} source 必须是对象")
    source_value = {
        "id": _stable_id(source.get("id"), "source.id"),
        "title": _text(source.get("title"), "source.title"),
        "url": _text(source.get("url"), "source.url"),
        "type": _stable_id(source.get("type"), "source.type"),
    }
    if not source_value["url"].startswith("https://"):
        raise CorpusValidationError(f"{document_id} 来源必须使用 HTTPS")
    chunk: dict[str, Any] = {
        "chunk_id": document_id + "#0001",
        "document_id": document_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "name_zh": name_zh,
        "name_en": name_en,
        "aliases": aliases,
        "title": title,
        "text": re.sub(r"\s+", " ", text).strip(),
        "source": source_value,
        "acquired_on": _text(document.get("acquired_on", defaults.get("acquired_on")), "acquired_on"),
        "game_version": _text(document.get("game_version", defaults.get("game_version")), "game_version"),
        "license_note": _text(document.get("license_note", defaults.get("license_note")), "license_note"),
        "content_version": _text(document.get("content_version", defaults.get("content_version")), "content_version"),
        "stale": document.get("stale", defaults.get("stale", False)),
    }
    if type(chunk["stale"]) is not bool:
        raise CorpusValidationError(f"{document_id} stale 必须是布尔值")
    canonical = json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chunk["checksum"] = "sha256:" + sha256(canonical).hexdigest()
    return chunk


def _all_aliases(chunk: dict[str, Any]) -> list[str]:
    values = [chunk["name_zh"], chunk["name_en"], chunk["entity_id"], *chunk["aliases"]]
    suffix = chunk["entity_id"].rsplit(":", 1)[-1]
    if suffix.isdigit():
        values.append(suffix)
    return list(dict.fromkeys(values))


def _stable_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if not re.fullmatch(r"[A-Za-z0-9_./:+-]+", text):
        raise CorpusValidationError(f"{name} 不是稳定 ID：{text}")
    return text


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusValidationError(f"{name} 必须是非空字符串")
    return value.strip()
