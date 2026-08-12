"""统一玩家文本问答入口；语音转写与键盘问题共用同一闭环。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from threading import Event
from typing import Any

from .budget import BudgetTracker, CostInfo
from .knowledge import Source
from .modeling import ModelCancelled, ModelError, ModelRequest, ModelRouter
from .rag import RagFilters, RagHit, RagResult, RagService
from .state import GameState


class QueryError(RuntimeError):
    pass


class QueryValidationError(QueryError):
    pass


@dataclass(frozen=True, slots=True)
class QueryToken:
    request_id: str
    run_id: str | None
    state_seq: int
    room_index: int | None
    room_spawn_seed: int | None

    @classmethod
    def from_state(cls, request_id: str, state: GameState) -> "QueryToken":
        return cls(
            request_id,
            state.run_id,
            state.last_seq,
            _optional_int(state.context.get("room_index")),
            _optional_int(state.context.get("room_spawn_seed")),
        )

    def is_current(self, state: GameState, request_id: str) -> bool:
        return (
            request_id == self.request_id
            and state.run_id == self.run_id
            and state.last_seq >= self.state_seq
            and _optional_int(state.context.get("room_index")) == self.room_index
            and _optional_int(state.context.get("room_spawn_seed")) == self.room_spawn_seed
        )


@dataclass(frozen=True, slots=True)
class QueryResponse:
    answer: str
    confidence: float
    sources: tuple[Source, ...]
    state_seq: int
    cost: CostInfo
    simulated: bool
    delivery_note: str | None
    rag_hits: tuple[RagHit, ...]
    retrieval_latency_ms: float
    retrieval_degraded: bool
    retrieval_corpus_version: str
    retrieval_degradation_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": [asdict(item) for item in self.sources],
            "state_seq": self.state_seq,
            "cost": asdict(self.cost),
            "simulated": self.simulated,
            "delivery_note": self.delivery_note,
            "rag_hits": [hit.as_dict() for hit in self.rag_hits],
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "retrieval_degraded": self.retrieval_degraded,
            "retrieval_corpus_version": self.retrieval_corpus_version,
            "retrieval_degradation_reason": self.retrieval_degradation_reason,
        }


class QueryEngine:
    MODEL_ROLE = "advice"

    def __init__(
        self,
        rag: RagService,
        router: ModelRouter,
        budget: BudgetTracker,
        *,
        game_version: str,
    ) -> None:
        self.rag = rag
        self.router = router
        self.budget = budget
        self.game_version = game_version

    def ask(
        self,
        question: str,
        state: GameState,
        request_id: str,
        cancel: Event | None = None,
    ) -> tuple[QueryResponse, QueryToken]:
        cancel_event = cancel or Event()
        text = question.strip()
        if not text:
            raise QueryError("问题不能为空")
        if len(text) > 300:
            raise QueryError("问题过长，请缩短后重试")
        token = QueryToken.from_state(request_id, state)
        result = self.rag.retrieve(
            text,
            filters=RagFilters(game_version=self.game_version),
            top_k=5,
        )
        if cancel_event.is_set():
            raise ModelCancelled("文本请求已取消")
        if result.no_answer:
            raise QueryError("本地资料不足，无法可靠回答这个问题。")
        sources = _sources_from_result(result)
        context = {
            "question": text,
            "game_state": {
                "run_id": state.run_id,
                "state_seq": state.last_seq,
                "context": dict(state.context),
                "players": dict(state.players),
            },
            "evidence": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "entity_type": hit.chunk.entity_type,
                    "entity_id": hit.chunk.entity_id,
                    "name_zh": hit.chunk.name_zh,
                    "text": hit.chunk.text,
                    "source_id": hit.chunk.source.id,
                }
                for hit in result.hits
            ],
            "state_seq": state.last_seq,
        }
        fallback = _fallback_answer(result)
        request = ModelRequest(
            "你是 Oriens 游戏助手。仅依据本次提供的本地检索证据，用简体中文回答玩家问题。"
            "必须输出字段严格为 advice、reason、confidence、sources、state_seq 的 JSON 对象；"
            "advice 是不超过 120 字的直接回答，reason 是不超过 160 字的证据说明。"
            "不得使用外部知识或编造来源。",
            json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            {
                "fallback_advice": fallback,
                "fallback_reason": "回答来自本次本地 RAG 召回结果。",
                "allowed_source_ids": [source.id for source in sources],
                "state_seq": state.last_seq,
            },
        )
        note: str | None = None
        if self.router.online and not self.budget.can_call_online():
            routed = self.router.complete_offline(self.MODEL_ROLE, request, cancel_event)
            note = "本局预算上限已达到，已使用本地证据摘要。"
        else:
            try:
                routed = self.router.complete(self.MODEL_ROLE, request, cancel_event)
            except ModelCancelled:
                raise
            except ModelError:
                routed = self.router.complete_offline(self.MODEL_ROLE, request, cancel_event)
                note = "网络模型不可用，已使用本地证据摘要。"
        draft = _validate_query_json(
            routed.content,
            expected_state_seq=state.last_seq,
            allowed_sources={item.id for item in sources},
        )
        source_map = {item.id: item for item in sources}
        cost = self.budget.record(routed.display_name, routed.usage, routed.model)
        response = QueryResponse(
            answer=draft[0],
            confidence=draft[1],
            sources=tuple(source_map[source_id] for source_id in draft[2]),
            state_seq=state.last_seq,
            cost=cost,
            simulated=routed.simulated,
            delivery_note=note,
            rag_hits=result.hits,
            retrieval_latency_ms=result.latency_ms,
            retrieval_degraded=result.degraded,
            retrieval_corpus_version=result.corpus_version,
            retrieval_degradation_reason=result.degradation_reason,
        )
        return response, token


def _validate_query_json(
    content: str, *, expected_state_seq: int, allowed_sources: set[str]
) -> tuple[str, float, tuple[str, ...]]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        raise QueryValidationError("回答不是有效 JSON") from None
    required = {"advice", "reason", "confidence", "sources", "state_seq"}
    if not isinstance(value, dict) or set(value) != required:
        raise QueryValidationError("回答字段与 schema 不匹配")
    advice = value.get("advice")
    reason = value.get("reason")
    confidence = value.get("confidence")
    source_ids = value.get("sources")
    if not isinstance(advice, str) or not advice.strip() or len(advice.strip()) > 240:
        raise QueryValidationError("回答正文无效")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 320:
        raise QueryValidationError("回答说明无效")
    if type(confidence) not in {int, float} or not 0 <= confidence <= 1:
        raise QueryValidationError("回答置信度无效")
    if type(value.get("state_seq")) is not int or value["state_seq"] != expected_state_seq:
        raise QueryValidationError("回答状态序号已过期")
    if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 5:
        raise QueryValidationError("回答来源无效")
    unique: list[str] = []
    for source_id in source_ids:
        if not isinstance(source_id, str) or source_id not in allowed_sources:
            raise QueryValidationError("回答引用了本次检索以外的来源")
        if source_id not in unique:
            unique.append(source_id)
    return advice.strip(), float(confidence), tuple(unique)


def _sources_from_result(result: RagResult) -> tuple[Source, ...]:
    values: list[Source] = []
    for hit in result.hits:
        source = Source(hit.chunk.source.id, hit.chunk.source.title, hit.chunk.source.url)
        if source.id not in {item.id for item in values}:
            values.append(source)
    return tuple(values)


def _fallback_answer(result: RagResult) -> str:
    first = result.hits[0].chunk
    text = " ".join(first.text.split())
    if len(text) > 120:
        text = text[:119].rstrip() + "…"
    return text


def _optional_int(value: Any) -> int | None:
    return value if type(value) is int else None
