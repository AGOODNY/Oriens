"""道具房建议闭环：事件 -> 本地资料 -> 模型 -> 校验 -> 费用。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from threading import Event
from typing import Any

from .budget import BudgetTracker, CostInfo
from .knowledge import ItemKnowledge, LocalItemKnowledgeBase, Source
from .modeling import ModelCancelled, ModelError, ModelRequest, ModelRouter
from .protocol import GameEvent
from .state import GameState


class AdviceError(RuntimeError):
    """建议无法生成或无法通过校验。"""


class AdviceValidationError(AdviceError):
    """模型结构化输出不可信，禁止展示。"""


@dataclass(frozen=True, slots=True)
class StateToken:
    run_id: str
    state_seq: int
    room_index: int | None
    room_spawn_seed: int | None
    collectible_id: int

    @classmethod
    def from_event(cls, event: GameEvent, collectible_id: int) -> "StateToken":
        return cls(
            run_id=event.run_id,
            state_seq=event.seq,
            room_index=_optional_int(event.context.get("room_index")),
            room_spawn_seed=_optional_int(event.context.get("room_spawn_seed")),
            collectible_id=collectible_id,
        )

    def is_current(self, state: GameState) -> bool:
        """心跳可推进 seq；换局或换房会使建议过期。"""

        return (
            state.run_id == self.run_id
            and state.last_seq >= self.state_seq
            and _optional_int(state.context.get("room_index")) == self.room_index
            and _optional_int(state.context.get("room_spawn_seed"))
            == self.room_spawn_seed
        )


@dataclass(frozen=True, slots=True)
class AdviceDraft:
    advice: str
    reason: str
    confidence: float
    source_ids: tuple[str, ...]
    state_seq: int


@dataclass(frozen=True, slots=True)
class AdviceResponse:
    advice: str
    reason: str
    confidence: float
    sources: tuple[Source, ...]
    state_seq: int
    cost: CostInfo
    collectible_id: int
    item_name: str
    simulated: bool
    delivery_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "advice": self.advice,
            "reason": self.reason,
            "confidence": self.confidence,
            "sources": [asdict(source) for source in self.sources],
            "state_seq": self.state_seq,
            "cost": asdict(self.cost),
            "collectible_id": self.collectible_id,
            "item_name": self.item_name,
            "simulated": self.simulated,
            "delivery_note": self.delivery_note,
        }


class AdviceEngine:
    MODEL_ROLE = "advice"

    def __init__(
        self,
        knowledge: LocalItemKnowledgeBase,
        router: ModelRouter,
        budget: BudgetTracker,
    ) -> None:
        self.knowledge = knowledge
        self.router = router
        self.budget = budget

    def supports(self, event: GameEvent) -> bool:
        return self.knowledge_for(event) is not None

    def knowledge_for(self, event: GameEvent) -> ItemKnowledge | None:
        if event.type != "collectible_spawned":
            return None
        collectible_id = event.payload.get("collectible_id")
        if type(collectible_id) is not int:
            return None
        # ROOM_TREASURE = 4。阶段 1 只做“进入道具房”切片。
        if event.context.get("room_type") != 4:
            return None
        return self.knowledge.find(collectible_id)

    def generate(
        self,
        event: GameEvent,
        cancel: Event | None = None,
    ) -> tuple[AdviceResponse, StateToken]:
        cancel_event = cancel or Event()
        item = self.knowledge_for(event)
        if item is None:
            raise AdviceError("当前事件不是已覆盖的道具房道具")
        token = StateToken.from_event(event, item.collectible_id)
        model_request = self._make_request(event, item)
        delivery_note: str | None = None

        if self.router.online and not self.budget.can_call_online():
            routed = self.router.complete_offline(self.MODEL_ROLE, model_request, cancel_event)
            delivery_note = "本局预算上限已达到，已改用本地模拟建议。"
        else:
            try:
                routed = self.router.complete(self.MODEL_ROLE, model_request, cancel_event)
            except ModelCancelled:
                raise
            except ModelError:
                routed = self.router.complete_offline(
                    self.MODEL_ROLE, model_request, cancel_event
                )
                delivery_note = "网络模型不可用，已改用本地模拟建议。"

        allowed_sources = {source.id: source for source in item.sources}
        draft = validate_advice_draft(
            routed.content,
            expected_state_seq=event.seq,
            allowed_source_ids=set(allowed_sources),
        )
        cost = self.budget.record(routed.display_name, routed.usage, routed.model)
        sources = tuple(allowed_sources[source_id] for source_id in draft.source_ids)
        response = AdviceResponse(
            advice=draft.advice,
            reason=draft.reason,
            confidence=draft.confidence,
            sources=sources,
            state_seq=draft.state_seq,
            cost=cost,
            collectible_id=item.collectible_id,
            item_name=f"{item.name_zh} / {item.name_en}",
            simulated=routed.simulated,
            delivery_note=delivery_note,
        )
        validate_advice_response(response)
        return response, token

    @staticmethod
    def _make_request(event: GameEvent, item: ItemKnowledge) -> ModelRequest:
        system_prompt = (
            "你是 Oriens 游戏助手。仅依据提供的本地资料生成简体中文短建议。"
            "必须输出一个 JSON 对象，字段严格为 advice、reason、confidence、sources、"
            "state_seq。不得编造来源；建议和理由各不超过 80 个汉字。"
        )
        player_state = {
            "context": dict(event.context),
            "item": item.prompt_context(),
            "state_seq": event.seq,
        }
        user_prompt = json.dumps(player_state, ensure_ascii=False, separators=(",", ":"))
        return ModelRequest(
            system_prompt,
            user_prompt,
            {
                "fallback_advice": item.advice_hint,
                "fallback_reason": item.facts[0],
                "allowed_source_ids": [source.id for source in item.sources],
                "state_seq": event.seq,
            },
        )


def validate_advice_draft(
    content: str,
    *,
    expected_state_seq: int,
    allowed_source_ids: set[str],
) -> AdviceDraft:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdviceValidationError("模型输出不是有效 JSON") from exc
    if not isinstance(value, dict):
        raise AdviceValidationError("模型输出根节点必须是对象")
    required = {"advice", "reason", "confidence", "sources", "state_seq"}
    if set(value) != required:
        raise AdviceValidationError("模型输出字段与 schema 不匹配")

    advice = _bounded_text(value["advice"], "advice", 160)
    reason = _bounded_text(value["reason"], "reason", 240)
    confidence = value["confidence"]
    if type(confidence) not in {int, float} or not 0 <= confidence <= 1:
        raise AdviceValidationError("confidence 必须是 0 到 1 的数字")
    sources = value["sources"]
    if not isinstance(sources, list) or not sources or len(sources) > 5:
        raise AdviceValidationError("sources 必须包含 1 到 5 个来源")
    source_ids: list[str] = []
    for source_id in sources:
        if not isinstance(source_id, str) or source_id not in allowed_source_ids:
            raise AdviceValidationError("模型引用了未检索或不存在的来源")
        if source_id not in source_ids:
            source_ids.append(source_id)
    state_seq = value["state_seq"]
    if type(state_seq) is not int or state_seq != expected_state_seq:
        raise AdviceValidationError("模型返回的状态序号不匹配")
    return AdviceDraft(advice, reason, float(confidence), tuple(source_ids), state_seq)


def validate_advice_response(response: AdviceResponse) -> None:
    """展示前的最终 schema 校验，包含程序计算的费用字段。"""

    _bounded_text(response.advice, "advice", 160)
    _bounded_text(response.reason, "reason", 240)
    if not response.sources:
        raise AdviceValidationError("最终建议缺少来源")
    if response.state_seq < 1:
        raise AdviceValidationError("最终建议状态序号无效")
    cost = response.cost
    if (
        cost.input_tokens < 0
        or cost.output_tokens < 0
        or cost.estimated_cost_cny < 0
        or cost.run_total_cny < 0
        or cost.currency != "CNY"
    ):
        raise AdviceValidationError("最终建议费用信息无效")


def _bounded_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdviceValidationError(f"{name} 必须是非空字符串")
    text = value.strip()
    if len(text) > limit:
        raise AdviceValidationError(f"{name} 超过长度限制")
    return text


def _optional_int(value: Any) -> int | None:
    return value if type(value) is int else None
