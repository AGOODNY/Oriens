"""Oriens 游戏日志事件协议。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

EVENT_PREFIX = "[ORIENS_EVENT]"
SCHEMA_VERSION = 1


class EventParseError(ValueError):
    """日志行不是合法 Oriens 事件。"""


@dataclass(frozen=True, slots=True)
class GameEvent:
    schema_version: int
    seq: int
    run_id: str
    type: str
    game_frame: int
    context: Mapping[str, Any]
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GameEvent":
        required = ("schema_version", "seq", "run_id", "type", "game_frame")
        missing = [key for key in required if key not in value]
        if missing:
            raise EventParseError(f"事件缺少字段：{', '.join(missing)}")

        schema_version = value["schema_version"]
        seq = value["seq"]
        run_id = value["run_id"]
        event_type = value["type"]
        game_frame = value["game_frame"]
        context = value.get("context", {})
        payload = value.get("payload", {})

        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            raise EventParseError(f"不支持的 schema_version：{schema_version!r}")
        if type(seq) is not int or seq < 1:
            raise EventParseError("seq 必须是大于等于 1 的整数")
        if not isinstance(run_id, str) or not run_id:
            raise EventParseError("run_id 必须是非空字符串")
        if not isinstance(event_type, str) or not event_type:
            raise EventParseError("type 必须是非空字符串")
        if type(game_frame) is not int or game_frame < 0:
            raise EventParseError("game_frame 必须是非负整数")
        if not isinstance(context, Mapping):
            raise EventParseError("context 必须是对象")
        if not isinstance(payload, Mapping):
            raise EventParseError("payload 必须是对象")

        return cls(
            schema_version=schema_version,
            seq=seq,
            run_id=run_id,
            type=event_type,
            game_frame=game_frame,
            context=dict(context),
            payload=dict(payload),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "run_id": self.run_id,
            "type": self.type,
            "game_frame": self.game_frame,
            "context": dict(self.context),
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))


def parse_event_line(line: str) -> GameEvent | None:
    """解析游戏日志行或录制文件中的纯 JSON 行。

    非 Oriens 日志行返回 ``None``；带前缀但内容损坏时抛出
    :class:`EventParseError`，这样监听器可以单独统计协议错误。
    """

    text = line.strip()
    if not text:
        return None

    prefix_index = text.find(EVENT_PREFIX)
    if prefix_index >= 0:
        text = text[prefix_index + len(EVENT_PREFIX) :].strip()
    elif not text.startswith("{"):
        return None

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventParseError(f"JSON 无效：{exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise EventParseError("事件根节点必须是对象")
    return GameEvent.from_mapping(raw)

