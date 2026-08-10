"""阶段 1 的小型、可追溯本地道具资料库。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class KnowledgeError(ValueError):
    """本地资料格式无效。"""


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class ItemKnowledge:
    collectible_id: int
    name_zh: str
    name_en: str
    facts: tuple[str, ...]
    advice_hint: str
    sources: tuple[Source, ...]

    def prompt_context(self) -> dict[str, Any]:
        return {
            "collectible_id": self.collectible_id,
            "name": self.name_zh,
            "english_name": self.name_en,
            "facts": list(self.facts),
            "advice_hint": self.advice_hint,
            "allowed_source_ids": [source.id for source in self.sources],
        }


class LocalItemKnowledgeBase:
    """按道具 ID 精确召回；阶段 1 不引入向量库或完整 Wiki。"""

    def __init__(self, items: dict[int, ItemKnowledge]) -> None:
        self._items = dict(items)

    @classmethod
    def load(cls, path: Path) -> "LocalItemKnowledgeBase":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeError(f"无法读取本地道具资料：{path}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise KnowledgeError("本地道具资料 schema_version 必须为 1")
        entries = raw.get("items")
        if not isinstance(entries, list):
            raise KnowledgeError("本地道具资料 items 必须是数组")

        items: dict[int, ItemKnowledge] = {}
        for entry in entries:
            item = _parse_item(entry)
            if item.collectible_id in items:
                raise KnowledgeError(f"道具 ID 重复：{item.collectible_id}")
            items[item.collectible_id] = item
        return cls(items)

    def find(self, collectible_id: int) -> ItemKnowledge | None:
        return self._items.get(collectible_id)

    def known_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._items))


def _parse_item(value: Any) -> ItemKnowledge:
    if not isinstance(value, dict):
        raise KnowledgeError("道具条目必须是对象")
    collectible_id = value.get("collectible_id")
    if type(collectible_id) is not int or collectible_id < 1:
        raise KnowledgeError("collectible_id 必须是正整数")
    name_zh = _required_string(value, "name_zh")
    name_en = _required_string(value, "name_en")
    advice_hint = _required_string(value, "advice_hint")
    facts_raw = value.get("facts")
    if not isinstance(facts_raw, list) or not facts_raw:
        raise KnowledgeError(f"道具 {collectible_id} 必须至少包含一条事实")
    facts = tuple(_plain_string(item, "facts") for item in facts_raw)
    sources_raw = value.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise KnowledgeError(f"道具 {collectible_id} 必须至少包含一个来源")
    sources = tuple(_parse_source(item) for item in sources_raw)
    return ItemKnowledge(
        collectible_id, name_zh, name_en, facts, advice_hint, sources
    )


def _parse_source(value: Any) -> Source:
    if not isinstance(value, dict):
        raise KnowledgeError("来源必须是对象")
    source = Source(
        _required_string(value, "id"),
        _required_string(value, "title"),
        _required_string(value, "url"),
    )
    if not source.url.startswith("https://"):
        raise KnowledgeError("来源 URL 必须使用 HTTPS")
    return source


def _required_string(value: dict[str, Any], name: str) -> str:
    return _plain_string(value.get(name), name)


def _plain_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f"{name} 必须是非空字符串")
    return value.strip()
