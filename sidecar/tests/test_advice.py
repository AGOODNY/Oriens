from __future__ import annotations

import json
from threading import Event
import unittest

from oriens.advice import AdviceEngine, AdviceValidationError, StateToken
from oriens.budget import BudgetTracker
from sidecar.tests.test_support import load_test_config as load_config
from oriens.knowledge import LocalItemKnowledgeBase
from oriens.memory import MemoryContext, MemoryItem
from oriens.modeling import (
    AdapterResponse,
    ModelAdapter,
    ModelCancelled,
    ModelError,
    ModelRequest,
    ModelRouter,
    ModelUsage,
)
from oriens.protocol import GameEvent
from oriens.state import StateStore


def item_event(seq: int = 10) -> GameEvent:
    return GameEvent(
        schema_version=1,
        seq=seq,
        run_id="TEST RUN:0",
        type="collectible_spawned",
        game_frame=200,
        context={
            "stage": 1,
            "room_index": 4,
            "room_type": 4,
            "room_spawn_seed": 555,
        },
        payload={"collectible_id": 350, "init_seed": 123, "price": 0},
    )


class StaticAdapter(ModelAdapter):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.requests = []

    def complete(self, model, model_request: ModelRequest, cancel: Event) -> AdapterResponse:
        self.calls += 1
        self.requests.append(model_request)
        return AdapterResponse(self.content, ModelUsage(100, 20))


class FlakyAdapter(StaticAdapter):
    def complete(self, model, model_request: ModelRequest, cancel: Event) -> AdapterResponse:
        self.calls += 1
        if self.calls == 1:
            raise ModelError("临时失败")
        return AdapterResponse(self.content, ModelUsage(100, 20))


class AlwaysFailAdapter(StaticAdapter):
    def complete(self, model, model_request: ModelRequest, cancel: Event) -> AdapterResponse:
        self.calls += 1
        raise ModelError("网络失败")


class AdviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.knowledge = LocalItemKnowledgeBase.load(self.config.app.knowledge_path)

    def _engine(self, router: ModelRouter) -> AdviceEngine:
        budget = BudgetTracker(self.config.budget.run_limit_cny)
        budget.set_run("TEST RUN:0")
        return AdviceEngine(self.knowledge, router, budget)

    def test_offline_simulation_returns_valid_sourced_zero_cost_advice(self) -> None:
        router = ModelRouter(self.config, online=False, api_key=None)
        response, token = self._engine(router).generate(item_event())
        self.assertTrue(response.simulated)
        self.assertEqual(response.cost.estimated_cost_cny, 0)
        self.assertEqual(response.state_seq, 10)
        self.assertEqual(response.sources[0].id, "wiki.gg:item/toxic-shock")
        self.assertEqual(token.collectible_id, 350)

    def test_rejects_unretrieved_citation(self) -> None:
        content = json.dumps(
            {
                "advice": "建议拾取。",
                "reason": "可以清房。",
                "confidence": 0.8,
                "sources": ["made-up:source"],
                "state_seq": 10,
            }
        )
        adapter = StaticAdapter(content)
        router = ModelRouter(
            self.config,
            online=True,
            api_key="test-only",
            adapters={"advice": adapter},
        )
        with self.assertRaises(AdviceValidationError):
            self._engine(router).generate(item_event())

    def test_rejects_wrong_state_sequence(self) -> None:
        content = json.dumps(
            {
                "advice": "建议拾取。",
                "reason": "可以清房。",
                "confidence": 0.8,
                "sources": ["wiki.gg:item/toxic-shock"],
                "state_seq": 9,
            }
        )
        router = ModelRouter(
            self.config,
            online=True,
            api_key="test-only",
            adapters={"advice": StaticAdapter(content)},
        )
        with self.assertRaises(AdviceValidationError):
            self._engine(router).generate(item_event())

    def test_retries_once_then_accepts_valid_response(self) -> None:
        content = json.dumps(
            {
                "advice": "建议拾取。",
                "reason": "提供开场群体伤害。",
                "confidence": 0.9,
                "sources": ["wiki.gg:item/toxic-shock"],
                "state_seq": 10,
            }
        )
        adapter = FlakyAdapter(content)
        router = ModelRouter(
            self.config,
            online=True,
            api_key="test-only",
            adapters={"advice": adapter},
        )
        response, _token = self._engine(router).generate(item_event())
        self.assertEqual(adapter.calls, 2)
        self.assertFalse(response.simulated)

    def test_structured_memory_cannot_replace_current_event_or_sources(self) -> None:
        content = json.dumps(
            {
                "advice": "建议拾取。",
                "reason": "提供开场群体伤害。",
                "confidence": 0.9,
                "sources": ["wiki.gg:item/toxic-shock"],
                "state_seq": 10,
            }
        )
        adapter = StaticAdapter(content)
        router = ModelRouter(
            self.config, online=True, api_key="test-only",
            adapters={"advice": adapter},
        )
        item = MemoryItem(
            id="memory", kind="guidance_preference", content="解释深度偏好：简短",
            status="active", confidence=1.0, confirmation_level="manual",
            source_summary="用户手动添加", source_session_id=None, source_run_id=None,
            created_at="2026-08-14T00:00:00+00:00",
            updated_at="2026-08-14T00:00:00+00:00", last_used_at=None,
        )
        response, _token = self._engine(router).generate(
            item_event(), memory_context=MemoryContext((item,), len(item.content))
        )
        prompt = json.loads(adapter.requests[0].user_prompt)
        self.assertEqual(prompt["context"]["room_index"], 4)
        self.assertEqual(prompt["item"]["collectible_id"], 350)
        self.assertEqual(prompt["long_term_memory"][0]["content"], item.content)
        self.assertEqual(response.sources[0].id, "wiki.gg:item/toxic-shock")
        self.assertIn("当前事件状态和本地资料始终优先", adapter.requests[0].system_prompt)

    def test_network_failure_falls_back_to_local_simulation(self) -> None:
        adapter = AlwaysFailAdapter("")
        router = ModelRouter(
            self.config,
            online=True,
            api_key="test-only",
            adapters={"advice": adapter},
        )
        response, _token = self._engine(router).generate(item_event())
        self.assertTrue(response.simulated)
        self.assertIn("网络模型不可用", response.delivery_note or "")
        self.assertEqual(adapter.calls, 2)

    def test_cancelled_request_does_not_generate(self) -> None:
        cancel = Event()
        cancel.set()
        router = ModelRouter(self.config, online=False, api_key=None)
        with self.assertRaises(ModelCancelled):
            self._engine(router).generate(item_event(), cancel)

    def test_state_token_allows_heartbeat_but_expires_after_room_change(self) -> None:
        token = StateToken.from_event(item_event(), 350)
        store = StateStore()
        store.apply(item_event())
        heartbeat = GameEvent(
            1,
            11,
            "TEST RUN:0",
            "heartbeat",
            260,
            dict(item_event().context),
            {},
        )
        store.apply(heartbeat)
        self.assertTrue(token.is_current(store.state))
        moved = GameEvent(
            1,
            12,
            "TEST RUN:0",
            "room_entered",
            300,
            {"stage": 1, "room_index": 5, "room_type": 1, "room_spawn_seed": 777},
            {},
        )
        store.apply(moved)
        self.assertFalse(token.is_current(store.state))


if __name__ == "__main__":
    unittest.main()
