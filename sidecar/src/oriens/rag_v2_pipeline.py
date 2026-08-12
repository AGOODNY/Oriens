"""灰机 Wiki 完整快照到 ``rag-v2`` 的确定性流式导入流水线。

原始 JSONL 始终逐行读取。页面正文只在处理当前页面时进入内存；页面目录、
重定向图、去重指纹和实体合并状态存放在临时 SQLite 目录库中。Lua 仅进行
白名单静态字面量提取，绝不导入或执行下载的模块。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any, Iterable, Iterator, TextIO

from .rag import normalize_alias
from .rag_pipeline import CorpusValidationError


SCHEMA_VERSION = 2
DEFAULT_CONTENT_VERSION = "rag-v2-huiji-2026-08-10"
DEFAULT_GAME_VERSION = "Repentance+ 1.9.7.17.J460"
DATA_NAMESPACE = 3500

_REQUIRED_RAW_FIELDS = {
    "authorization_ref",
    "content_checksum",
    "content_model",
    "document_id",
    "license_note",
    "namespace",
    "page_id",
    "redirect",
    "retrieved_at",
    "revision_id",
    "revision_sha1",
    "revision_timestamp",
    "revision_url",
    "source_title",
    "source_type",
    "source_url",
    "stale",
    "title",
    "wikitext",
}

_NOISE_SECTIONS = {
    "画廊",
    "音乐",
    "轶事",
    "参考资料",
    "参考文献",
    "外部链接",
    "导航",
    "注释",
    "漏洞",
    "更新历史",
    "版本历史",
    "gallery",
    "trivia",
    "references",
    "external links",
}

_DROP_TEMPLATES = {
    "nav",
    "masternav",
    "itemnav",
    "characternav",
    "roomnav",
    "stagenav",
    "entitynav",
    "消歧义",
    "消歧义页",
    "clear",
    "toc",
    "notoc",
    "quote",
    "quotation",
    "主条目",
    "main",
    "see also",
    "参见",
    "dont confuse",
    "about",
    "需要翻译",
    "施工中",
    "stub",
}

_DISPLAY_FIRST_ARG = {
    "en",
    "math",
    "kbd",
    "key",
    "item",
    "道具",
    "chara",
    "character",
    "stage",
    "章节",
    "room",
    "房间",
    "pool",
    "道具池",
    "mode",
    "challenge",
    "挑战",
    "achievement",
    "成就",
    "card",
    "trinket",
    "pill",
    "effect",
    "效果",
    "ttl",
}

_DROP_TAGS = ("gallery", "imagemap", "timeline", "poem", "syntaxhighlight", "source")
_STATIC_LUA_MODULES = {
    "模块:Rooms",
    "模块:EntityQuery",
    "模块:EntityQuery/Data",
    "模块:AchievementQuery",
    "模块:CostumeQuery",
    "模块:CostumeQuery/Data",
}
_REQUIRED_TEMPLATES = {"模板:Infobox item", "模板:ItemSummary"}
_REQUIRED_MODULES = {
    "模块:Rooms",
    "模块:EntityQuery",
    "模块:AchievementQuery",
    "模块:CostumeQuery",
}


@dataclass(frozen=True, slots=True)
class RagV2Paths:
    raw_paths: tuple[Path, ...]
    chunks_path: Path
    manifest_path: Path
    entities_path: Path
    redirects_path: Path
    dependency_audit_path: Path
    lua_facts_path: Path
    overrides_path: Path


@dataclass(frozen=True, slots=True)
class RagV2BuildReport:
    document_count: int
    chunk_count: int
    entity_count: int
    raw_record_count: int
    redirect_count: int
    redirect_resolved: int
    redirect_cycles: int
    redirect_broken: int
    exact_duplicates: int
    near_duplicates: int
    skipped_noise_pages: int
    lua_static_fact_count: int
    elapsed_seconds: float
    peak_working_set_bytes: int | None
    corpus_checksum: str
    data_document_count: int = 0
    data_chunk_count: int = 0
    room_layout_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "raw_record_count": self.raw_record_count,
            "redirect_count": self.redirect_count,
            "redirect_resolved": self.redirect_resolved,
            "redirect_cycles": self.redirect_cycles,
            "redirect_broken": self.redirect_broken,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "skipped_noise_pages": self.skipped_noise_pages,
            "lua_static_fact_count": self.lua_static_fact_count,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "corpus_checksum": self.corpus_checksum,
            "data_document_count": self.data_document_count,
            "data_chunk_count": self.data_chunk_count,
            "room_layout_count": self.room_layout_count,
        }


def build_full_corpus(
    paths: RagV2Paths,
    *,
    content_version: str = DEFAULT_CONTENT_VERSION,
    game_version: str = DEFAULT_GAME_VERSION,
    progress: callable | None = None,
) -> RagV2BuildReport:
    """流式导入完整快照并原子发布派生语料文件。"""

    started = time.perf_counter()
    notify = progress or (lambda _message: None)
    for raw_path in paths.raw_paths:
        if not raw_path.is_file():
            raise CorpusValidationError(f"灰机快照不存在：{raw_path}")
    overrides = _load_overrides(paths.overrides_path)
    output_dir = paths.chunks_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_handle = tempfile.NamedTemporaryFile(
        prefix="rag-v2-catalog-", suffix=".sqlite", dir=output_dir, delete=False
    )
    catalog_path = Path(catalog_handle.name)
    catalog_handle.close()
    db = sqlite3.connect(catalog_path)
    db.row_factory = sqlite3.Row
    temporary_outputs: list[tuple[Path, Path]] = []
    try:
        _create_catalog(db)
        counters, template_counts, module_counts, lua_fact_count = _catalog_raw_pages(
            db, paths.raw_paths, notify
        )
        notify(f"已流式校验并登记 {counters['raw_records']} 条原始记录")
        redirect_stats = _resolve_redirects(db)
        notify(
            "重定向解析完成："
            f"{redirect_stats['resolved']} 条有效，"
            f"{redirect_stats['cycle']} 条循环，"
            f"{redirect_stats['broken']} 条失效"
        )

        chunk_temp = _temporary_output(paths.chunks_path)
        entity_temp = _temporary_output(paths.entities_path)
        redirect_temp = _temporary_output(paths.redirects_path)
        audit_temp = _temporary_output(paths.dependency_audit_path)
        lua_temp = _temporary_output(paths.lua_facts_path)
        temporary_outputs.extend(
            [
                (chunk_temp, paths.chunks_path),
                (entity_temp, paths.entities_path),
                (redirect_temp, paths.redirects_path),
                (audit_temp, paths.dependency_audit_path),
                (lua_temp, paths.lua_facts_path),
            ]
        )

        _write_redirects(db, redirect_temp)
        _write_lua_facts(db, lua_temp)
        build_stats = _write_chunks_and_entities(
            db,
            chunk_temp,
            entity_temp,
            overrides,
            content_version,
            game_version,
            notify,
        )
        audit = _dependency_audit(
            db, template_counts, module_counts, counters, redirect_stats, lua_fact_count
        )
        _write_json(audit_temp, audit)
        corpus_checksum = "sha256:" + _file_sha256(chunk_temp)
        peak_rss = _working_set_bytes()
        elapsed = time.perf_counter() - started
        corpus_id = (
            "oriens-rag-v2.1-huiji-data-complete"
            if counters[f"namespace_{DATA_NAMESPACE}"]
            else "oriens-rag-v2-huiji-complete"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "content_version": content_version,
            "game_version": game_version,
            "source_snapshots": [path.parent.name for path in paths.raw_paths],
            "raw_record_count": counters["raw_records"],
            "raw_page_count": counters["namespace_0"],
            "raw_template_count": counters["namespace_10"],
            "raw_module_count": counters["namespace_828"],
            "raw_data_count": counters[f"namespace_{DATA_NAMESPACE}"],
            "document_count": build_stats["documents"],
            "chunk_count": build_stats["chunks"],
            "entity_count": build_stats["entities"],
            "redirect_count": counters["redirects"],
            "redirect_resolved": redirect_stats["resolved"],
            "redirect_cycles": redirect_stats["cycle"],
            "redirect_broken": redirect_stats["broken"],
            "exact_duplicates": build_stats["exact_duplicates"],
            "near_duplicates": build_stats["near_duplicates"],
            "skipped_noise_pages": build_stats["skipped_noise_pages"],
            "lua_static_fact_count": lua_fact_count,
            "data_document_count": build_stats["data_documents"],
            "data_chunk_count": build_stats["data_chunks"],
            "room_layout_count": build_stats["room_layouts"],
            "skipped_data_pages": build_stats["skipped_data_pages"],
            "corpus_checksum": corpus_checksum,
            "license_note": (
                "以撒中文 Wiki 原创内容按 CC BY-NC-SA 3.0 及管理员书面授权处理；"
                "第三方内容与游戏素材仍需逐页核对；仅限本地个人非商业研究"
            ),
            "authorization_ref": "private-written-authorization",
            "normalization": "oriens-rag-v2-schema-2",
            "elapsed_seconds": round(elapsed, 3),
            "peak_working_set_bytes": peak_rss,
        }
        manifest_temp = _temporary_output(paths.manifest_path)
        temporary_outputs.append((manifest_temp, paths.manifest_path))
        _write_json(manifest_temp, manifest)
        for temporary, destination in temporary_outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        return RagV2BuildReport(
            build_stats["documents"],
            build_stats["chunks"],
            build_stats["entities"],
            counters["raw_records"],
            counters["redirects"],
            redirect_stats["resolved"],
            redirect_stats["cycle"],
            redirect_stats["broken"],
            build_stats["exact_duplicates"],
            build_stats["near_duplicates"],
            build_stats["skipped_noise_pages"],
            lua_fact_count,
            round(elapsed, 3),
            peak_rss,
            corpus_checksum,
            build_stats["data_documents"],
            build_stats["data_chunks"],
            build_stats["room_layouts"],
        )
    finally:
        db.close()
        try:
            catalog_path.unlink()
        except OSError:
            # 临时目录库清理由操作系统或下一次人工维护处理；不影响已发布结果。
            pass


def _create_catalog(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE pages(
            title TEXT NOT NULL,
            title_key TEXT PRIMARY KEY,
            namespace INTEGER NOT NULL,
            page_id INTEGER NOT NULL,
            revision_id INTEGER NOT NULL,
            raw_document_id TEXT NOT NULL,
            redirect INTEGER NOT NULL,
            redirect_target TEXT,
            wikitext TEXT NOT NULL,
            source_url TEXT NOT NULL,
            revision_url TEXT NOT NULL,
            source_title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            revision_timestamp TEXT NOT NULL,
            content_checksum TEXT NOT NULL,
            license_note TEXT NOT NULL,
            authorization_ref TEXT NOT NULL,
            stale INTEGER NOT NULL
            ,content_model TEXT NOT NULL
        );
        CREATE TABLE redirect_resolution(
            source_key TEXT PRIMARY KEY,
            final_key TEXT,
            status TEXT NOT NULL,
            chain_json TEXT NOT NULL
        );
        CREATE TABLE dependencies(
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(kind, name)
        );
        CREATE TABLE lua_facts(
            module_title TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            source_document_id TEXT NOT NULL,
            page_id INTEGER NOT NULL,
            revision_id INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            PRIMARY KEY(module_title, fact_key, fact_value)
        );
        CREATE TABLE seen_content(
            content_checksum TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            simhash TEXT NOT NULL,
            text_length INTEGER NOT NULL
        );
        CREATE TABLE simhash_bands(
            band INTEGER NOT NULL,
            band_value INTEGER NOT NULL,
            content_checksum TEXT NOT NULL,
            PRIMARY KEY(band, band_value, content_checksum)
        );
        CREATE INDEX simhash_band_lookup ON simhash_bands(band, band_value);
        CREATE TABLE entities(
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name_zh TEXT NOT NULL,
            name_en TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            document_ids_json TEXT NOT NULL
        );
        """
    )


def _catalog_raw_pages(
    db: sqlite3.Connection,
    raw_paths: Iterable[Path],
    notify: callable,
) -> tuple[Counter[str], Counter[str], Counter[str], int]:
    counters: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    modules: Counter[str] = Counter()
    seen_page_revisions: set[tuple[int, int]] = set()
    lua_fact_count = 0
    insert = "INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    for raw_path in raw_paths:
        with raw_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusValidationError(
                        f"{raw_path.name}:{line_number} 不是有效 JSON"
                    ) from exc
                _validate_raw_record(record, raw_path, line_number)
                key = (record["page_id"], record["revision_id"])
                if key in seen_page_revisions:
                    raise CorpusValidationError(f"页面/修订重复：{key[0]}/{key[1]}")
                seen_page_revisions.add(key)
                title_key = _title_key(record["title"])
                redirect_target = (
                    _parse_redirect_target(record["wikitext"])
                    if record["redirect"]
                    else None
                )
                db.execute(
                    insert,
                    (
                        record["title"],
                        title_key,
                        record["namespace"],
                        record["page_id"],
                        record["revision_id"],
                        record["document_id"],
                        int(record["redirect"]),
                        redirect_target,
                        record["wikitext"],
                        record["source_url"],
                        record["revision_url"],
                        record["source_title"],
                        record["source_type"],
                        record["retrieved_at"],
                        record["revision_timestamp"],
                        record["content_checksum"],
                        record["license_note"],
                        record["authorization_ref"],
                        int(record["stale"]),
                        record["content_model"],
                    ),
                )
                counters["raw_records"] += 1
                counters[f"namespace_{record['namespace']}"] += 1
                if record["redirect"]:
                    counters["redirects"] += 1
                if record["namespace"] in {0, 10, 828}:
                    for template in _template_dependencies(record["wikitext"]):
                        templates[template] += 1
                    for module in _module_dependencies(record["wikitext"]):
                        modules[module] += 1
                if record["namespace"] == 828 and record["title"] in _STATIC_LUA_MODULES:
                    facts = _extract_lua_literals(record["wikitext"])
                    for fact_key, fact_value in facts:
                        db.execute(
                            "INSERT OR IGNORE INTO lua_facts VALUES(?,?,?,?,?,?,?)",
                            (
                                record["title"],
                                fact_key,
                                fact_value,
                                record["document_id"],
                                record["page_id"],
                                record["revision_id"],
                                record["source_url"],
                            ),
                        )
                    lua_fact_count += len(facts)
                if counters["raw_records"] % 1000 == 0:
                    notify(f"已读取 {counters['raw_records']} 条原始记录")
    db.commit()
    for name, count in sorted(templates.items(), key=lambda item: item[0].casefold()):
        db.execute("INSERT INTO dependencies VALUES('template',?,?)", (name, count))
    for name, count in sorted(modules.items(), key=lambda item: item[0].casefold()):
        db.execute("INSERT INTO dependencies VALUES('module',?,?)", (name, count))
    db.commit()
    return counters, templates, modules, lua_fact_count


def _validate_raw_record(record: Any, path: Path, line_number: int) -> None:
    if not isinstance(record, dict) or not _REQUIRED_RAW_FIELDS <= set(record):
        raise CorpusValidationError(f"{path.name}:{line_number} 缺少原始记录字段")
    if record.get("schema_version") != 1:
        raise CorpusValidationError(f"{path.name}:{line_number} schema_version 无效")
    if type(record["page_id"]) is not int or type(record["revision_id"]) is not int:
        raise CorpusValidationError(f"{path.name}:{line_number} 页面/修订 ID 无效")
    if type(record["namespace"]) is not int or record["namespace"] not in {
        0, 10, 828, DATA_NAMESPACE
    }:
        raise CorpusValidationError(f"{path.name}:{line_number} 命名空间不在允许范围")
    if type(record["redirect"]) is not bool or type(record["stale"]) is not bool:
        raise CorpusValidationError(f"{path.name}:{line_number} 布尔字段无效")
    if not isinstance(record["wikitext"], str):
        raise CorpusValidationError(f"{path.name}:{line_number} wikitext 无效")
    if not isinstance(record["content_model"], str) or not record["content_model"]:
        raise CorpusValidationError(f"{path.name}:{line_number} content_model 无效")
    expected = "sha256:" + sha256(record["wikitext"].encode("utf-8")).hexdigest()
    if record["content_checksum"] != expected:
        raise CorpusValidationError(f"{path.name}:{line_number} 正文校验和不匹配")
    if not str(record["source_url"]).startswith("https://"):
        raise CorpusValidationError(f"{path.name}:{line_number} 来源 URL 必须为 HTTPS")
    if record["redirect"] and not _parse_redirect_target(record["wikitext"]):
        raise CorpusValidationError(f"{path.name}:{line_number} 无法解析重定向目标")


def _resolve_redirects(db: sqlite3.Connection) -> Counter[str]:
    rows = db.execute(
        "SELECT title_key, redirect_target FROM pages WHERE redirect=1 ORDER BY title_key"
    ).fetchall()
    mapping = {
        row["title_key"]: _title_key(row["redirect_target"])
        for row in rows
        if row["redirect_target"]
    }
    existing = {
        row[0]
        for row in db.execute("SELECT title_key FROM pages")
    }
    stats: Counter[str] = Counter()
    for source_key in sorted(mapping):
        chain = [source_key]
        seen = {source_key}
        current = mapping[source_key]
        status = "resolved"
        final_key: str | None = None
        while True:
            chain.append(current)
            if current in seen:
                status = "cycle"
                break
            seen.add(current)
            target = mapping.get(current)
            if target is None:
                if current in existing:
                    final_key = current
                else:
                    status = "broken"
                break
            current = target
        db.execute(
            "INSERT INTO redirect_resolution VALUES(?,?,?,?)",
            (source_key, final_key, status, json.dumps(chain, ensure_ascii=False)),
        )
        stats[status] += 1
    db.commit()
    return stats


def _write_redirects(db: sqlite3.Connection, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        rows = db.execute(
            """
            SELECT s.title AS source_title, s.page_id AS source_page_id,
                   s.revision_id AS source_revision_id,
                   s.raw_document_id AS source_document_id,
                   s.redirect_target, r.status, r.chain_json,
                   t.title AS final_title, t.page_id AS final_page_id,
                   t.revision_id AS final_revision_id,
                   t.raw_document_id AS final_document_id
            FROM redirect_resolution r
            JOIN pages s ON s.title_key=r.source_key
            LEFT JOIN pages t ON t.title_key=r.final_key
            ORDER BY s.title_key
            """
        )
        for row in rows:
            value = dict(row)
            value["chain"] = json.loads(value.pop("chain_json"))
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_lua_facts(db: sqlite3.Connection, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        rows = db.execute(
            "SELECT * FROM lua_facts ORDER BY module_title, fact_key, fact_value"
        )
        for row in rows:
            value = dict(row)
            value["extraction"] = "static-literal-only"
            value["executed"] = False
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_chunks_and_entities(
    db: sqlite3.Connection,
    chunks_path: Path,
    entities_path: Path,
    overrides: dict[str, Any],
    content_version: str,
    game_version: str,
    notify: callable,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    rows = db.execute(
        "SELECT * FROM pages WHERE namespace=0 AND redirect=0 ORDER BY title_key"
    )
    with chunks_path.open("w", encoding="utf-8", newline="\n") as target:
        for page_number, row in enumerate(rows, start=1):
            redirect_sources = _redirect_sources(db, row["title_key"])
            aliases = [row["title"], *(item["title"] for item in redirect_sources)]
            entity = _classify_entity(dict(row), aliases, overrides)
            sections = normalize_wikitext_sections(row["wikitext"], row["title"])
            if not sections:
                stats["skipped_noise_pages"] += 1
                continue
            document_id = f"huiji:page:{row['page_id']}"
            written_for_document = 0
            section_occurrences: Counter[str] = Counter()
            for section_path, text_blocks in sections:
                section_key = " / ".join(section_path) if section_path else "导语"
                section_hash = sha256(section_key.encode("utf-8")).hexdigest()[:12]
                for semantic_ordinal, text in enumerate(text_blocks, start=1):
                    normalized_text = _normalize_plain_text(text)
                    if len(normalized_text) < 12:
                        continue
                    section_occurrences[section_hash] += 1
                    ordinal = section_occurrences[section_hash]
                    chunk_id = f"{document_id}#sec:{section_hash}:{ordinal:02d}"
                    content_checksum = sha256(
                        _dedupe_text(normalized_text).encode("utf-8")
                    ).hexdigest()
                    simhash = _simhash64(normalized_text)
                    duplicate = _dedupe_decision(
                        db, content_checksum, simhash, len(normalized_text)
                    )
                    if duplicate == "exact":
                        stats["exact_duplicates"] += 1
                        continue
                    if duplicate == "near":
                        stats["near_duplicates"] += 1
                        continue
                    source = {
                        "id": row["raw_document_id"],
                        "title": row["source_title"],
                        "url": row["revision_url"],
                        "canonical_url": row["source_url"],
                        "type": row["source_type"],
                    }
                    chunk: dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "entity_type": entity["entity_type"],
                        "entity_id": entity["entity_id"],
                        "name_zh": entity["name_zh"],
                        "name_en": entity["name_en"],
                        "aliases": entity["aliases"],
                        "title": f"{row['title']} · {section_key}",
                        "section_path": list(section_path),
                        "text": normalized_text,
                        "source": source,
                        "source_material_type": "wiki-page-current-revision",
                        "page_id": row["page_id"],
                        "revision_id": row["revision_id"],
                        "revision_timestamp": row["revision_timestamp"],
                        "acquired_on": row["retrieved_at"][:10],
                        "retrieved_at": row["retrieved_at"],
                        "game_version": game_version,
                        "license_note": row["license_note"],
                        "authorization_ref": row["authorization_ref"],
                        "content_version": content_version,
                        "stale": bool(row["stale"]),
                        "raw_document_id": row["raw_document_id"],
                        "redirect_sources": redirect_sources,
                        "normalization": "wikitext-whitelist-static-v2",
                    }
                    canonical = json.dumps(
                        chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    chunk["checksum"] = "sha256:" + sha256(canonical).hexdigest()
                    target.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
                    _remember_content(db, content_checksum, chunk_id, simhash, len(normalized_text))
                    written_for_document += 1
                    stats["chunks"] += 1
            if written_for_document:
                stats["documents"] += 1
                _merge_entity(db, entity, document_id)
            if page_number % 250 == 0:
                db.commit()
                notify(
                    f"已规范化 {page_number} 个正文页面，生成 {stats['chunks']} 个分块"
                )
        _write_data_chunks(
            db,
            target,
            content_version,
            game_version,
            stats,
            notify,
        )
        db.commit()
    with entities_path.open("w", encoding="utf-8", newline="\n") as target:
        for row in db.execute("SELECT * FROM entities ORDER BY entity_type, entity_id"):
            value = dict(row)
            value["aliases"] = json.loads(value.pop("aliases_json"))
            value["document_ids"] = json.loads(value.pop("document_ids_json"))
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stats["entities"] += 1
    return stats


def _write_data_chunks(
    db: sqlite3.Connection,
    target: TextIO,
    content_version: str,
    game_version: str,
    stats: Counter[str],
    notify: callable,
) -> None:
    """将白名单 Data 表和 ROOM_STB 静态转换为可追溯分块。"""

    item_page = db.execute(
        "SELECT * FROM pages WHERE namespace=? AND title='Data:Item.tabx'",
        (DATA_NAMESPACE,),
    ).fetchone()
    keyword_page = db.execute(
        "SELECT * FROM pages WHERE namespace=? AND title='Data:ItemKeywords.tabx'",
        (DATA_NAMESPACE,),
    ).fetchone()
    if item_page is not None:
        keyword_rows = (
            {str(row.get("page")): row for row in _tabular_rows(keyword_page["wikitext"])}
            if keyword_page is not None
            else {}
        )
        item_rows = _tabular_rows(item_page["wikitext"])
        for ordinal, item in enumerate(item_rows, start=1):
            page_key = str(item.get("page") or "").strip()
            entity_type, entity_id = _item_entity_identity(page_key, item)
            keyword = keyword_rows.get(page_key, {})
            aliases = _data_item_aliases(page_key, item, keyword)
            name_zh = _clean_scalar(item.get("namezh")) or page_key
            name_en = _clean_scalar(item.get("nameen")) or name_zh
            text = _data_item_text(item, entity_type, entity_id)
            entity = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "name_zh": name_zh,
                "name_en": name_en,
                "aliases": aliases,
            }
            document_id = f"huiji:data:{item_page['page_id']}:row:{page_key}"
            if _write_structured_chunk(
                db,
                target,
                item_page,
                entity,
                document_id,
                f"{item_page['title']} · {name_zh}",
                text,
                "wiki-data-tabular-row",
                content_version,
                game_version,
                {"data_row": ordinal, "data_page_key": page_key},
                stats,
            ):
                _merge_entity(db, entity, document_id, prefer_names=True)
                stats["data_documents"] += 1

    entity_page = db.execute(
        "SELECT * FROM pages WHERE namespace=? AND title='Data:Entity.tabx'",
        (DATA_NAMESPACE,),
    ).fetchone()
    if entity_page is not None:
        entity_rows = _tabular_rows(entity_page["wikitext"])
        identities = Counter(_game_entity_key(row) for row in entity_rows)
        for ordinal, row in enumerate(entity_rows, start=1):
            identity = _game_entity_key(row)
            entity_id = "game-entity:" + identity
            name_zh = _clean_scalar(row.get("namezh"))
            name_en = _clean_scalar(row.get("nameen"))
            fallback = name_zh or name_en or identity
            entity = {
                "entity_type": "game_entity",
                "entity_id": entity_id,
                "name_zh": name_zh or fallback,
                "name_en": name_en or fallback,
                "aliases": _unique_aliases(
                    [identity, _clean_scalar(row.get("page")), name_zh, name_en]
                ),
            }
            document_id = f"huiji:data:{entity_page['page_id']}:row:{ordinal:04d}"
            conflict = identities[identity] > 1
            if _write_structured_chunk(
                db,
                target,
                entity_page,
                entity,
                document_id,
                f"{entity_page['title']} · {fallback}",
                _data_entity_text(row, identity, conflict),
                "wiki-data-tabular-row",
                content_version,
                game_version,
                {
                    "data_row": ordinal,
                    "source_conflict": conflict,
                    "conflict_key": identity if conflict else None,
                },
                stats,
            ):
                _merge_entity(db, entity, document_id, prefer_names=True)
                stats["data_documents"] += 1

    room_rows = db.execute(
        "SELECT * FROM pages WHERE namespace=? AND title LIKE 'Data:Rooms/%' "
        "ORDER BY title_key",
        (DATA_NAMESPACE,),
    )
    for ordinal, row in enumerate(room_rows, start=1):
        try:
            room = json.loads(row["wikitext"])
        except json.JSONDecodeError as exc:
            raise CorpusValidationError(f"{row['title']} ROOM_STB JSON 无效") from exc
        if not isinstance(room, dict) or room.get("_type") != "ROOM_STB":
            stats["skipped_data_pages"] += 1
            continue
        identity = _room_identity(room)
        name = _clean_scalar(room.get("name")) or f"房间布局 {identity}"
        entity = {
            "entity_type": "room_layout",
            "entity_id": "room-layout:" + sha256(identity.encode("utf-8")).hexdigest()[:20],
            "name_zh": name,
            "name_en": name,
            "aliases": _room_aliases(room, identity),
        }
        document_id = f"huiji:data:page:{row['page_id']}"
        if _write_structured_chunk(
            db,
            target,
            row,
            entity,
            document_id,
            f"{row['title']} · 结构化房间布局",
            _room_text(room, identity),
            "wiki-data-room-stb",
            content_version,
            game_version,
            {
                "room_file": room.get("_file"),
                "room_type": room.get("type"),
                "room_variant": room.get("variant"),
                "room_subtype": room.get("subtype"),
                "room_shape": room.get("shape"),
            },
            stats,
        ):
            _merge_entity(db, entity, document_id, prefer_names=True)
            stats["data_documents"] += 1
            stats["room_layouts"] += 1
        if ordinal % 1000 == 0:
            db.commit()
            notify(
                f"已静态转换 {ordinal} 个房间页面，Data 分块 {stats['data_chunks']} 个"
            )

    data_total = db.execute(
        "SELECT count(*) FROM pages WHERE namespace=?", (DATA_NAMESPACE,)
    ).fetchone()[0]
    # 这里按原始 Data 页面计数；表格行不是独立页面，因此只把实际白名单页面计入。
    recognized_pages = stats["room_layouts"] + int(item_page is not None) + int(
        keyword_page is not None
    ) + int(entity_page is not None)
    stats["skipped_data_pages"] += max(0, data_total - recognized_pages)
    notify(
        f"Data 静态接入完成：{stats['data_chunks']} 个分块，"
        f"其中 {stats['room_layouts']} 个房间布局；跳过 {stats['skipped_data_pages']} 个噪声/非白名单页面"
    )


def _write_structured_chunk(
    db: sqlite3.Connection,
    target: TextIO,
    row: sqlite3.Row,
    entity: dict[str, Any],
    document_id: str,
    title: str,
    text: str,
    material_type: str,
    content_version: str,
    game_version: str,
    metadata: dict[str, Any],
    stats: Counter[str],
) -> bool:
    normalized_text = _normalize_plain_text(text)
    content_checksum = sha256(_dedupe_text(normalized_text).encode("utf-8")).hexdigest()
    simhash = _simhash64(normalized_text)
    # 结构化记录中的稳定 ID 是语义的一部分；布局相似不等于重复房间。
    # 只拦截完全相同的内容，不对 Data 行和 ROOM_STB 应用近重复删除。
    if db.execute(
        "SELECT 1 FROM seen_content WHERE content_checksum=?", (content_checksum,)
    ).fetchone():
        stats["exact_duplicates"] += 1
        return False
    chunk_id = document_id + "#structured:01"
    source = {
        "id": row["raw_document_id"],
        "title": row["source_title"],
        "url": row["revision_url"],
        "canonical_url": row["source_url"],
        "type": row["source_type"],
    }
    chunk: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": document_id,
        **entity,
        "title": title,
        "section_path": ["结构化数据"],
        "text": normalized_text,
        "source": source,
        "source_material_type": material_type,
        "page_id": row["page_id"],
        "revision_id": row["revision_id"],
        "revision_timestamp": row["revision_timestamp"],
        "acquired_on": row["retrieved_at"][:10],
        "retrieved_at": row["retrieved_at"],
        "game_version": game_version,
        "license_note": row["license_note"],
        "authorization_ref": row["authorization_ref"],
        "content_version": content_version,
        "stale": bool(row["stale"]),
        "raw_document_id": row["raw_document_id"],
        "redirect_sources": [],
        "normalization": "json-whitelist-static-v2.1",
        **metadata,
    }
    canonical = json.dumps(
        chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    chunk["checksum"] = "sha256:" + sha256(canonical).hexdigest()
    target.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    _remember_content(db, content_checksum, chunk_id, simhash, len(normalized_text))
    stats["chunks"] += 1
    stats["data_chunks"] += 1
    stats["documents"] += 1
    return True


def _tabular_rows(value: str) -> list[dict[str, Any]]:
    try:
        table = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CorpusValidationError("Data 表格不是有效 JSON") from exc
    if not isinstance(table, dict) or not isinstance(table.get("schema"), dict):
        raise CorpusValidationError("Data 表格缺少 schema")
    fields_raw = table["schema"].get("fields")
    data = table.get("data")
    if not isinstance(fields_raw, list) or not isinstance(data, list):
        raise CorpusValidationError("Data 表格 fields/data 无效")
    fields = [
        field.get("name") if isinstance(field, dict) else None for field in fields_raw
    ]
    if not fields or not all(isinstance(field, str) and field for field in fields):
        raise CorpusValidationError("Data 表格字段名无效")
    rows: list[dict[str, Any]] = []
    for ordinal, values in enumerate(data, start=1):
        if not isinstance(values, list) or len(values) != len(fields):
            raise CorpusValidationError(f"Data 表格第 {ordinal} 行列数无效")
        rows.append(dict(zip(fields, values)))
    return rows


def _item_entity_identity(
    page_key: str, row: dict[str, Any]
) -> tuple[str, str]:
    match = re.fullmatch(r"([ctkp])(\d+)", page_key.casefold())
    if not match:
        raise CorpusValidationError(f"Data:Item.tabx 页面键无效：{page_key}")
    prefix, raw_id = match.groups()
    number = int(raw_id)
    mapping = {
        "c": ("item", "collectible"),
        "t": ("trinket", "trinket"),
        "k": ("card", "card"),
        "p": ("pill", "pill"),
    }
    entity_type, id_prefix = mapping[prefix]
    return entity_type, f"{id_prefix}:{number}"


def _data_item_aliases(
    page_key: str, item: dict[str, Any], keyword: dict[str, Any]
) -> list[str]:
    aliases: list[Any] = [
        page_key,
        item.get("namezh"),
        item.get("nameen"),
    ]
    for field, separator in (
        (item.get("namelist"), ";"),
        (keyword.get("name_alias"), ";"),
        (keyword.get("PinyinIndex"), ";"),
    ):
        if isinstance(field, str):
            aliases.extend(field.split(separator))
    return _unique_aliases(_clean_scalar(value) for value in aliases)


def _data_item_text(
    item: dict[str, Any], entity_type: str, entity_id: str
) -> str:
    labels = {
        "item": "收藏品",
        "trinket": "饰品",
        "card": "卡牌或符文",
        "pill": "胶囊",
    }
    parts = [
        f"名称：{_clean_scalar(item.get('namezh')) or entity_id}",
        f"英文名：{_clean_scalar(item.get('nameen')) or '未填写'}",
        f"实体类型：{labels[entity_type]}",
        f"稳定 ID：{entity_id}",
    ]
    field_labels = (
        ("desczh", "中文短描述"),
        ("descen", "英文短描述"),
        ("effect", "机制效果"),
        ("quality", "品质"),
        ("charge", "充能"),
        ("unlock", "解锁条件"),
        ("tag", "机制标签"),
        ("source", "游戏来源版本"),
        ("subtype", "子类型"),
    )
    for field, label in field_labels:
        value = _clean_scalar(item.get(field))
        if value and value not in {"-2147483648", "-1"}:
            parts.append(f"{label}：{value}")
    return "；".join(parts) + "。"


def _game_entity_key(row: dict[str, Any]) -> str:
    values: list[int] = []
    for field in ("type", "variant", "subtype"):
        value = row.get(field)
        if type(value) is not int:
            raise CorpusValidationError(f"Data:Entity.tabx {field} 不是整数")
        values.append(value)
    return ".".join(str(value) for value in values)


def _data_entity_text(
    row: dict[str, Any], identity: str, conflict: bool
) -> str:
    parts = [
        f"实体 ID：{identity}",
        f"中文名：{_clean_scalar(row.get('namezh')) or '未填写'}",
        f"英文名：{_clean_scalar(row.get('nameen')) or '未填写'}",
    ]
    for field, label in (
        ("page", "关联页面"),
        ("tag", "标签"),
        ("tips", "机制提示"),
        ("hp", "基础生命值"),
        ("stagehp", "关卡生命值"),
        ("shieldstrength", "护盾强度"),
        ("collisionDamage", "碰撞伤害"),
        ("source", "游戏来源版本"),
    ):
        value = _clean_scalar(row.get(field))
        if value and value != "-1":
            parts.append(f"{label}：{value}")
    if conflict:
        parts.append("源数据冲突：相同 type.variant.subtype 存在多个名称记录，均予保留")
    return "；".join(parts) + "。"


def _room_identity(room: dict[str, Any]) -> str:
    fields = ("_file", "type", "variant", "subtype", "_i")
    values = [_clean_scalar(room.get(field)) for field in fields]
    if not all(values):
        raise CorpusValidationError("ROOM_STB 缺少稳定身份字段")
    return "/".join(values)


def _room_aliases(room: dict[str, Any], identity: str) -> list[str]:
    file_name = _clean_scalar(room.get("_file"))
    values = [
        identity,
        _clean_scalar(room.get("name")),
        f"{file_name} {room.get('_i')}",
        f"room-layout {identity}",
    ]
    return _unique_aliases(values)


def _room_text(room: dict[str, Any], identity: str) -> str:
    doors = room.get("doors") if isinstance(room.get("doors"), list) else []
    spawns = room.get("spawns") if isinstance(room.get("spawns"), list) else []
    active_doors = [
        f"({door.get('x')},{door.get('y')})"
        for door in doors
        if isinstance(door, dict) and door.get("exists") is True
    ]
    entity_counts: Counter[str] = Counter()
    spawn_cells = 0
    for spawn in spawns:
        if not isinstance(spawn, dict):
            continue
        spawn_cells += 1
        entities = spawn.get("entity")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            key = ".".join(
                str(entity.get(field, 0)) for field in ("type", "variant", "subtype")
            )
            entity_counts[key] += 1
    entity_summary = "、".join(
        f"{key}×{count}" for key, count in sorted(entity_counts.items())
    ) or "无"
    parts = [
        f"房间布局 ID：{identity}",
        f"STB 文件：{room.get('_file')}",
        f"房间类型：{room.get('type')}",
        f"变体：{room.get('variant')}，子类型：{room.get('subtype')}",
        f"形状：{room.get('shape')}，尺寸：{room.get('width')}×{room.get('height')}",
        f"难度：{room.get('difficulty')}，权重：{room.get('weight')}",
        f"有效门：{'、'.join(active_doors) if active_doors else '无'}",
        f"生成格数量：{spawn_cells}",
        f"生成实体：{entity_summary}",
    ]
    name = _clean_scalar(room.get("name"))
    if name:
        parts.insert(1, f"房间名称：{name}")
    return "；".join(parts) + "。"


def _clean_scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return _normalize_plain_text(str(value))


def normalize_wikitext_sections(
    wikitext: str, page_title: str, *, target_chars: int = 1800
) -> list[tuple[tuple[str, ...], list[str]]]:
    """按标题层级和段落/列表块分组，不按固定字符硬切。"""

    text = re.sub(r"<!--[\s\S]*?-->", " ", wikitext)
    text = re.sub(r"<ref\b[^>]*>[\s\S]*?</ref\s*>", " ", text, flags=re.I)
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.I)
    for tag in _DROP_TAGS:
        text = re.sub(
            rf"<{tag}\b[^>]*>[\s\S]*?</{tag}\s*>", " ", text, flags=re.I
        )
    text = re.sub(r"<includeonly>|</includeonly>|<noinclude>|</noinclude>", " ", text, flags=re.I)
    heading_re = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
    current_path: list[str] = []
    current_lines: list[str] = []
    raw_sections: list[tuple[tuple[str, ...], str]] = []

    def flush() -> None:
        if current_lines:
            raw_sections.append((tuple(current_path), "\n".join(current_lines)))
            current_lines.clear()

    for line in text.splitlines():
        match = heading_re.match(line.strip())
        if match:
            flush()
            level = len(match.group(1)) - 2
            heading = _normalize_plain_text(_render_wikitext(match.group(2), page_title))
            current_path[:] = current_path[:level]
            current_path.append(heading or "未命名章节")
        else:
            current_lines.append(line)
    flush()
    results: list[tuple[tuple[str, ...], list[str]]] = []
    for section_path, raw_body in raw_sections:
        if section_path and section_path[0].strip().casefold() in _NOISE_SECTIONS:
            continue
        rendered = _render_wikitext(raw_body, page_title)
        rendered = _normalize_block_text(rendered)
        if not rendered:
            continue
        blocks = [block.strip() for block in re.split(r"\n\s*\n", rendered) if block.strip()]
        groups: list[str] = []
        current: list[str] = []
        current_length = 0
        for block in blocks:
            if current and current_length + len(block) + 2 > target_chars:
                groups.append("\n".join(current))
                current = []
                current_length = 0
            if len(block) > target_chars * 2:
                for sentence_group in _sentence_groups(block, target_chars):
                    if current:
                        groups.append("\n".join(current))
                        current = []
                        current_length = 0
                    groups.append(sentence_group)
                continue
            current.append(block)
            current_length += len(block) + 1
        if current:
            groups.append("\n".join(current))
        if groups:
            results.append((section_path, groups))
    return results


def _render_wikitext(text: str, page_title: str) -> str:
    rendered = _replace_balanced_templates(text, page_title)
    rendered = re.sub(
        r"\[\[(?:File|Image|文件):[^\]]+\]\]", " ", rendered, flags=re.I
    )
    rendered = re.sub(
        r"\[\[(?:Category|分类):[^\]]+\]\]", " ", rendered, flags=re.I
    )
    rendered = re.sub(
        r"\[\[([^\]|#]+)(?:#[^\]|]+)?\|([^\]]+)\]\]", r"\2", rendered
    )
    rendered = re.sub(r"\[\[([^\]|#]+)(?:#[^\]]+)?\]\]", r"\1", rendered)
    rendered = re.sub(r"\[(?:https?://)[^\s\]]+(?:\s+([^\]]+))?\]", r"\1", rendered)
    rendered = re.sub(r"'''?|__\w+__", "", rendered)
    rendered = re.sub(r"<br\s*/?>", "\n", rendered, flags=re.I)
    rendered = re.sub(r"</?(?:span|div|small|big|code|pre|blockquote|center|sup|sub|i|b|u|s|font)\b[^>]*>", " ", rendered, flags=re.I)
    rendered = re.sub(r"<[^>]+>", " ", rendered)
    rendered = re.sub(r"^\s*\{\|[\s\S]*?^\s*\|\}\s*$", lambda m: _table_to_text(m.group(0)), rendered, flags=re.M)
    rendered = re.sub(r"(?m)^\s*[-!|}]\s*", "", rendered)
    return html.unescape(rendered)


def _replace_balanced_templates(text: str, page_title: str) -> str:
    value = text
    for _ in range(40):
        stack: list[int] = []
        replacement: tuple[int, int, str] | None = None
        index = 0
        while index < len(value) - 1:
            pair = value[index : index + 2]
            if pair == "{{":
                stack.append(index)
                index += 2
                continue
            if pair == "}}" and stack:
                start = stack.pop()
                replacement = (
                    start,
                    index + 2,
                    _render_template(value[start + 2 : index], page_title),
                )
                break
            index += 1
        if replacement is None:
            break
        start, end, rendered = replacement
        value = value[:start] + rendered + value[end:]
    return re.sub(r"\{\{[\s\S]*?\}\}", " ", value)


def _render_template(body: str, page_title: str) -> str:
    parts = _split_top_level(body, "|")
    if not parts:
        return " "
    name = parts[0].strip().casefold().replace("template:", "").replace("模板:", "")
    positional: list[str] = []
    named: dict[str, str] = {}
    for part in parts[1:]:
        key_value = _split_top_level(part, "=", limit=1)
        if len(key_value) == 2 and key_value[0].strip():
            named[key_value[0].strip().casefold()] = key_value[1].strip()
        else:
            positional.append(part.strip())
    if name in _DROP_TEMPLATES or name.startswith("nav/"):
        return " "
    if name in {"n", "名称"}:
        return page_title
    if name in _DISPLAY_FIRST_ARG:
        if name in {"entity", "实体"}:
            return named.get("name") or named.get("id") or (positional[0] if positional else "")
        return positional[0] if positional else next(iter(named.values()), "")
    if name in {"dif", "difference"}:
        for key in ("dlcr+", "dlcr", "dlc", "default"):
            if key in named:
                return named[key]
        return positional[-1] if positional else " "
    if name.startswith("infobox") or name.startswith("信息框"):
        allowed = {
            "名称", "name", "英文名称", "nameen", "id", "类型", "type",
            "说明文字", "描述", "description", "解锁条件", "unlock", "效果", "effect",
        }
        values = [f"{key}：{value}" for key, value in named.items() if key in allowed and value]
        return "；".join(values)
    if name in {"itemsummary", "entitysummary"}:
        return " "
    if name.startswith("#invoke"):
        # 只保留调用的静态参数，不执行模块。
        module = name.split(":", 1)[-1]
        values = [value for value in positional[1:] if value]
        values.extend(f"{key}={value}" for key, value in named.items() if value)
        return f"{module}：" + "；".join(values) if values else " "
    if len(positional) == 1 and len(positional[0]) <= 80:
        # 未知装饰模板只保留单个短文本参数，避免丢失名称。
        return positional[0]
    return " "


def _split_top_level(value: str, separator: str, *, limit: int = -1) -> list[str]:
    parts: list[str] = []
    start = 0
    curly = square = 0
    splits = 0
    index = 0
    while index < len(value):
        pair = value[index : index + 2]
        if pair == "{{":
            curly += 1
            index += 2
            continue
        if pair == "}}" and curly:
            curly -= 1
            index += 2
            continue
        if pair == "[[":
            square += 1
            index += 2
            continue
        if pair == "]]" and square:
            square -= 1
            index += 2
            continue
        if value.startswith(separator, index) and not curly and not square and (limit < 0 or splits < limit):
            parts.append(value[start:index])
            index += len(separator)
            start = index
            splits += 1
            continue
        index += 1
    parts.append(value[start:])
    return parts


def _classify_entity(
    page: dict[str, Any], aliases: list[str], overrides: dict[str, Any]
) -> dict[str, Any]:
    title = page["title"]
    # 稳定 ID 覆盖只能由页面标题或重定向别名触发，不能由正文中的道具提及触发。
    override = _match_override(title, aliases, overrides)
    all_aliases = _unique_aliases([*aliases, *_extract_intro_aliases(page["wikitext"])])
    if override:
        return {
            "entity_type": override["entity_type"],
            "entity_id": override["entity_id"],
            "name_zh": override["name_zh"],
            "name_en": override["name_en"],
            "aliases": _unique_aliases([*all_aliases, *override.get("aliases", [])]),
        }
    compact = title.replace(" ", "")
    item = re.fullmatch(r"[Cc](\d+)", compact)
    trinket = re.fullmatch(r"[Tt](\d+)", compact)
    card = re.fullmatch(r"[Kk](\d+)", compact)
    character = re.search(r"\{\{\s*infobox\s+character\s*\|\s*(\d+)", page["wikitext"], re.I)
    if item:
        entity_type, entity_id = "item", f"collectible:{int(item.group(1))}"
    elif trinket:
        entity_type, entity_id = "trinket", f"trinket:{int(trinket.group(1))}"
    elif card:
        entity_type, entity_id = "card", f"card:{int(card.group(1))}"
    elif character:
        entity_type, entity_id = "character", f"player:{int(character.group(1))}"
    elif title == "房间" or title.startswith("房间/") or title.endswith(("房", "房间")):
        entity_type, entity_id = "room", f"wiki-room:{page['page_id']}"
    elif (
        title == "章节"
        or title.startswith("章节/")
        or "(章节)" in title
        or "（章节）" in title
        or re.search(r"\[\[(?:分类|Category):章节(?:\]|\|)", page["wikitext"], re.I)
    ):
        entity_type, entity_id = "route", f"wiki-route:{page['page_id']}"
    else:
        entity_type, entity_id = "wiki_page", f"wiki-page:{page['page_id']}"
    name_zh = _best_name(all_aliases, prefer_cjk=True) or title
    name_en = _best_name(all_aliases, prefer_cjk=False) or title
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "name_zh": name_zh,
        "name_en": name_en,
        "aliases": all_aliases,
    }


def _match_override(
    title: str, aliases: list[str], overrides: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = {normalize_alias(title), *(normalize_alias(value) for value in aliases)}
    for entry in overrides.get("entities", []):
        match_values = [entry.get("page_title", ""), *entry.get("match_aliases", [])]
        if any(normalize_alias(value) in candidates for value in match_values if value):
            return entry
    return None


def _extract_intro_aliases(wikitext: str) -> list[str]:
    aliases: list[str] = []
    lead = re.split(r"(?m)^={2,6}[^\n]*={2,6}\s*$", wikitext, maxsplit=1)[0]
    for match in re.finditer(r"\{\{\s*[Ee]n\s*\|\s*([^|}\n]+)", lead):
        aliases.append(match.group(1).strip())
    for match in re.finditer(r"'''([^'\n]{1,80})'''", lead):
        aliases.append(_normalize_plain_text(match.group(1)))
    return aliases


def _best_name(aliases: list[str], *, prefer_cjk: bool) -> str | None:
    candidates: list[str] = []
    for value in aliases:
        if re.fullmatch(r"[CcTtKk]\d+", value.strip()):
            continue
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", value))
        if has_cjk == prefer_cjk and 1 < len(value) <= 80:
            candidates.append(value)
    return min(candidates, key=lambda value: (len(value), value.casefold())) if candidates else None


def _redirect_sources(db: sqlite3.Connection, final_key: str) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT p.title, p.page_id, p.revision_id, p.raw_document_id AS document_id
        FROM redirect_resolution r JOIN pages p ON p.title_key=r.source_key
        WHERE r.final_key=? AND r.status='resolved' ORDER BY p.title_key
        """,
        (final_key,),
    )
    return [dict(row) for row in rows]


def _merge_entity(
    db: sqlite3.Connection,
    entity: dict[str, Any],
    document_id: str,
    *,
    prefer_names: bool = False,
) -> None:
    row = db.execute(
        "SELECT name_zh, name_en, aliases_json, document_ids_json "
        "FROM entities WHERE entity_id=?",
        (entity["entity_id"],),
    ).fetchone()
    if row is None:
        db.execute(
            "INSERT INTO entities VALUES(?,?,?,?,?,?)",
            (
                entity["entity_id"],
                entity["entity_type"],
                entity["name_zh"],
                entity["name_en"],
                json.dumps(entity["aliases"], ensure_ascii=False),
                json.dumps([document_id], ensure_ascii=False),
            ),
        )
        return
    aliases = _unique_aliases([*json.loads(row[2]), *entity["aliases"]])
    documents = sorted({*json.loads(row[3]), document_id})
    name_zh = entity["name_zh"] if prefer_names else row["name_zh"]
    name_en = entity["name_en"] if prefer_names else row["name_en"]
    db.execute(
        "UPDATE entities SET name_zh=?, name_en=?, aliases_json=?, "
        "document_ids_json=? WHERE entity_id=?",
        (
            name_zh,
            name_en,
            json.dumps(aliases, ensure_ascii=False),
            json.dumps(documents),
            entity["entity_id"],
        ),
    )


def _dedupe_decision(
    db: sqlite3.Connection, content_checksum: str, simhash: int, text_length: int
) -> str | None:
    if db.execute(
        "SELECT 1 FROM seen_content WHERE content_checksum=?", (content_checksum,)
    ).fetchone():
        return "exact"
    candidates: set[str] = set()
    for band, band_value in _simhash_bands(simhash):
        for row in db.execute(
            "SELECT content_checksum FROM simhash_bands WHERE band=? AND band_value=?",
            (band, band_value),
        ):
            candidates.add(row[0])
    for checksum in sorted(candidates):
        row = db.execute(
            "SELECT simhash, text_length FROM seen_content WHERE content_checksum=?",
            (checksum,),
        ).fetchone()
        if row is None:
            continue
        other_length = int(row["text_length"])
        if min(text_length, other_length) / max(text_length, other_length) < 0.92:
            continue
        if (simhash ^ int(row["simhash"], 16)).bit_count() <= 3:
            return "near"
    return None


def _remember_content(
    db: sqlite3.Connection,
    content_checksum: str,
    chunk_id: str,
    simhash: int,
    text_length: int,
) -> None:
    db.execute(
        "INSERT INTO seen_content VALUES(?,?,?,?)",
        (content_checksum, chunk_id, f"{simhash:016x}", text_length),
    )
    for band, band_value in _simhash_bands(simhash):
        db.execute(
            "INSERT INTO simhash_bands VALUES(?,?,?)",
            (band, band_value, content_checksum),
        )


def _simhash64(text: str) -> int:
    tokens = re.findall(r"[0-9a-z]+|[\u3400-\u9fff]", text.casefold())
    weights = [0] * 64
    for token in tokens:
        value = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _simhash_bands(value: int) -> Iterator[tuple[int, int]]:
    for band in range(4):
        yield band, (value >> (band * 16)) & 0xFFFF


def _dependency_audit(
    db: sqlite3.Connection,
    templates: Counter[str],
    modules: Counter[str],
    counters: Counter[str],
    redirect_stats: Counter[str],
    lua_fact_count: int,
) -> dict[str, Any]:
    titles = {row[0] for row in db.execute("SELECT title FROM pages")}
    titles_casefold = {_title_key(title).casefold() for title in titles}
    missing_templates = sorted(_REQUIRED_TEMPLATES - titles)
    missing_modules = sorted(_REQUIRED_MODULES - titles)
    unresolved_modules = sorted(
        name
        for name in modules
        if f"模块:{name}".casefold() not in titles_casefold
        and f"Module:{name}".casefold() not in titles_casefold
    )
    data_dependencies: Counter[str] = Counter()
    for row in db.execute("SELECT wikitext FROM pages WHERE namespace IN (10,828)"):
        for match in re.finditer(
            r"(?:Data:|数据:)([A-Za-z0-9_./ -]+(?:\.tabx?)?)", row[0], re.I
        ):
            dependency = "Data:" + match.group(1).strip().rstrip("./")
            if len(dependency) <= 160:
                data_dependencies[dependency] += 1
    missing_data_snapshots = sorted(
        dependency
        for dependency in data_dependencies
        if not _data_dependency_present(dependency, titles_casefold)
    )
    return {
        "schema_version": 1,
        "raw_records": counters["raw_records"],
        "redirects": dict(redirect_stats),
        "required_templates_present": not missing_templates,
        "required_modules_present": not missing_modules,
        "missing_required_templates": missing_templates,
        "missing_required_modules": missing_modules,
        "template_invocations": _casefold_counter(templates),
        "module_invocations": _casefold_counter(modules),
        "unresolved_invoked_modules": unresolved_modules,
        "data_dependencies": dict(data_dependencies.most_common()),
        "missing_data_snapshots": missing_data_snapshots,
        "data_dependency_limitations": (
            []
            if not missing_data_snapshots
            else ["部分静态 Data 依赖仍未包含在当前快照"]
        ),
        "data_namespace_present": counters[f"namespace_{DATA_NAMESPACE}"] > 0,
        "data_records": counters[f"namespace_{DATA_NAMESPACE}"],
        "lua_static_fact_count": lua_fact_count,
        "lua_execution": False,
        "category_namespace_required": False,
        "category_namespace_reason": (
            "正文内分类标记足以进行首轮实体分类；当前未发现必须读取分类页正文的依赖"
        ),
    }


def _data_dependency_present(dependency: str, titles_casefold: set[str]) -> bool:
    candidates = {_title_key(dependency).casefold()}
    if not dependency.casefold().endswith((".tab", ".tabx", ".json")):
        candidates.add(_title_key(dependency + ".tabx").casefold())
    return bool(candidates & titles_casefold)


def _casefold_counter(values: Counter[str]) -> dict[str, int]:
    combined: dict[str, tuple[str, int]] = {}
    for name, count in values.items():
        key = name.casefold()
        existing = combined.get(key)
        display = min(name, existing[0], key=str.casefold) if existing else name
        combined[key] = (display, count + (existing[1] if existing else 0))
    return {
        display: count
        for display, count in sorted(combined.values(), key=lambda item: item[0].casefold())
    }


def _extract_lua_literals(wikitext: str) -> list[tuple[str, str]]:
    facts: set[tuple[str, str]] = set()
    # 仅提取表中的字符串/数字/布尔字面量；忽略函数、表达式和数据库查询结果。
    pattern = re.compile(
        r"\[\s*['\"]([^'\"]{1,100})['\"]\s*\]\s*=\s*"
        r"(?:['\"]([^'\"]{0,200})['\"]|(-?\d+(?:\.\d+)?)|(true|false))"
    )
    for match in pattern.finditer(_strip_lua_comments(wikitext)):
        value = next((item for item in match.groups()[1:] if item is not None), "")
        facts.add((match.group(1), value))
    return sorted(facts)


def _strip_lua_comments(value: str) -> str:
    text = re.sub(r"--\[\[[\s\S]*?\]\]", " ", value)
    return re.sub(r"--[^\n]*", " ", text)


def _template_dependencies(wikitext: str) -> Iterator[str]:
    for match in re.finditer(r"\{\{\s*([^{}|\n]+)", wikitext):
        name = match.group(1).strip()
        if name and not name.startswith("#"):
            yield name


def _module_dependencies(wikitext: str) -> Iterator[str]:
    for match in re.finditer(r"\{\{\s*#invoke\s*:\s*([^|}\n]+)", wikitext, re.I):
        yield match.group(1).strip()


def _parse_redirect_target(wikitext: str) -> str | None:
    match = re.search(
        r"^\s*#(?:重定向|redirect)\s*:?[\s\u200e\u200f]*\[\[([^\]|#]+)",
        wikitext,
        re.I,
    )
    return match.group(1).strip() if match else None


def _title_key(value: str) -> str:
    # MediaWiki 默认规范化页面名首字符；有命名空间时处理冒号后的首字符。
    # 其余字符大小写仍可区分真实页面（如 Boss/BOSS）。
    text = re.sub(r"[ _]+", " ", value.strip())
    if not text:
        return text
    if ":" in text:
        namespace, title = text.split(":", 1)
        return f"{namespace}:{title[:1].upper()}{title[1:]}"
    return text[:1].upper() + text[1:]


def _normalize_plain_text(value: str) -> str:
    text = value.replace("\u200e", "").replace("\u200f", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip(" \n|;：")


def _normalize_block_text(value: str) -> str:
    lines: list[str] = []
    for line in value.splitlines():
        text = re.sub(r"^\s*[#*:;]+\s*", "", line).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            lines.append(text)
        elif lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


def _table_to_text(value: str) -> str:
    cells = re.split(r"\n\s*(?:\|-|\||!)", value)
    return "\n".join(_normalize_plain_text(cell) for cell in cells if _normalize_plain_text(cell))


def _sentence_groups(block: str, target_chars: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[。！？；.!?;])\s*", block) if part.strip()]
    groups: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + len(sentence) > target_chars:
            groups.append("".join(current))
            current = []
            length = 0
        current.append(sentence)
        length += len(sentence)
    if current:
        groups.append("".join(current))
    return groups


def _dedupe_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())


def _unique_aliases(values: Iterable[str]) -> list[str]:
    by_key: dict[str, str] = {}
    for value in values:
        clean = _normalize_plain_text(str(value))
        key = normalize_alias(clean)
        if key and len(clean) <= 120:
            existing = by_key.get(key)
            if existing is None or (len(clean), clean.casefold()) < (len(existing), existing.casefold()):
                by_key[key] = clean
    return sorted(by_key.values(), key=lambda value: (value.casefold(), value))


def _load_overrides(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusValidationError(f"无法读取 rag-v2 实体覆盖：{path}") from exc
    if value.get("schema_version") != 1 or not isinstance(value.get("entities"), list):
        raise CorpusValidationError("rag-v2 实体覆盖格式无效")
    required = {"entity_type", "entity_id", "name_zh", "name_en", "aliases"}
    for entry in value["entities"]:
        if not isinstance(entry, dict) or not required <= set(entry):
            raise CorpusValidationError("rag-v2 实体覆盖条目无效")
    return value


def _temporary_output(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False
    )
    path = Path(handle.name)
    handle.close()
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _working_set_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    except (ImportError, AttributeError, OSError):
        return None
