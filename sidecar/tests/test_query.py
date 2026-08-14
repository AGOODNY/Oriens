from __future__ import annotations

import json
from pathlib import Path
from threading import Event
import tempfile
import unittest

from oriens.budget import BudgetTracker
from sidecar.tests.test_support import load_test_config as load_config
from oriens.application import MemoryAwareQueryEngine
from oriens.memory import MemoryCandidate, SQLiteMemoryStore
from oriens.modeling import AdapterResponse, ModelRouter, ModelUsage
from oriens.query import QueryEngine, QueryError, QueryValidationError
from oriens.rag import RagService
from oriens.rag_pipeline import build_corpus, build_keyword_index
from oriens.state import GameState


class StaticAdapter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, model, model_request, cancel: Event):
        self.requests.append(model_request)
        return AdapterResponse(self.content, ModelUsage(20, 10))


class GroundedAdapter:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, model, model_request, cancel: Event):
        self.requests.append(model_request)
        return AdapterResponse(
            json.dumps({
                "advice": "当前道具是安卡十字。",
                "reason": "按当前房间道具的本地证据回答。",
                "confidence": 0.9,
                "sources": [model_request.metadata["allowed_source_ids"][0]],
                "state_seq": model_request.metadata["state_seq"],
            }, ensure_ascii=False),
            ModelUsage(20, 10),
        )


class DynamicGroundedAdapter:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, model, model_request, cancel: Event):
        self.requests.append(model_request)
        return AdapterResponse(
            json.dumps({
                "advice": "按当前游戏状态和本地证据回答。",
                "reason": "长期记忆只用于表达偏好。",
                "confidence": 0.9,
                "sources": [model_request.metadata["allowed_source_ids"][0]],
                "state_seq": model_request.metadata["state_seq"],
            }, ensure_ascii=False),
            ModelUsage(20, 10),
        )


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        chunks = build_corpus(cls.config.rag.source_path, root / "chunks.jsonl", root / "manifest.json")
        cls.index = root / "rag.sqlite"
        build_keyword_index(chunks, cls.index)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _engine(self, router: ModelRouter | None = None) -> QueryEngine:
        budget = BudgetTracker(self.config.budget.run_limit_cny)
        budget.set_run("QUERY:0")
        return QueryEngine(
            RagService(self.index),
            router or ModelRouter(self.config, online=False, api_key=None),
            budget,
            game_version=self.config.rag.game_version,
        )

    def test_player_question_uses_same_rag_and_only_current_citations(self) -> None:
        state = GameState(run_id="QUERY:0", active=True, last_seq=8, context={"room_index": 4, "room_spawn_seed": 9})
        response, token = self._engine().ask("硫磺火有什么效果", state, "request-1")
        retrieved = {hit.chunk.source.id for hit in response.rag_hits}
        self.assertTrue({source.id for source in response.sources} <= retrieved)
        self.assertTrue(token.is_current(state, "request-1"))
        self.assertFalse(token.is_current(state, "old-request"))

    def test_forged_citation_is_rejected(self) -> None:
        content = json.dumps({
            "advice": "回答", "reason": "说明", "confidence": 0.8,
            "sources": ["forged"], "state_seq": 8,
        })
        router = ModelRouter(
            self.config, online=True, api_key="test-only",
            adapters={"advice": StaticAdapter(content)},
        )
        state = GameState(run_id="QUERY:0", active=True, last_seq=8, context={"room_index": 4})
        with self.assertRaises(QueryValidationError):
            self._engine(router).ask("硫磺火", state, "request-2")

    def test_room_change_expires_question_token(self) -> None:
        state = GameState(run_id="QUERY:0", active=True, last_seq=8, context={"room_index": 4, "room_spawn_seed": 9})
        _response, token = self._engine().ask("硫磺火", state, "request-3")
        moved = GameState(run_id="QUERY:0", active=True, last_seq=9, context={"room_index": 5, "room_spawn_seed": 10})
        self.assertFalse(token.is_current(moved, "request-3"))

    def test_current_item_reference_uses_exact_room_item_and_resolved_state_names(self) -> None:
        live_config = load_config(Path("config/rag-v2.1-faiss.toml"))
        adapter = GroundedAdapter()
        router = ModelRouter(
            live_config, online=True, api_key="test-only",
            adapters={"advice": adapter},
        )
        state = GameState(
            run_id="QUERY:0",
            active=True,
            last_seq=17,
            context={"room_index": 4, "room_spawn_seed": 9},
            players={
                "0": {
                    "controller_index": 0,
                    "player_type": 0,
                    "inventory": {"active_item": 105},
                }
            },
            room_collectibles=[
                {"collectible_id": 161, "init_seed": 123, "taken": False}
            ],
        )

        budget = BudgetTracker(live_config.budget.run_limit_cny)
        budget.set_run("QUERY:0")
        engine = QueryEngine(
            RagService(live_config.rag.index_path), router, budget,
            game_version=live_config.rag.game_version,
        )
        response, _token = engine.ask("这个道具拿吗？", state, "request-4")

        self.assertEqual({hit.chunk.entity_id for hit in response.rag_hits}, {"collectible:161"})
        prompt = json.loads(adapter.requests[0].user_prompt)
        self.assertEqual(prompt["question_subject"]["entity_id"], "collectible:161")
        self.assertEqual(prompt["question_subject"]["name_zh"], "安卡十字")
        player = prompt["game_state"]["players"]["0"]
        self.assertEqual(player["resolved_identity"]["name_zh"], "以撒")
        self.assertEqual(
            player["inventory"]["resolved_active_item"]["name_zh"], "六面骰"
        )

    def test_current_item_reference_without_room_item_is_not_guessed(self) -> None:
        state = GameState(
            run_id="QUERY:0", active=True, last_seq=8,
            context={"room_index": 4, "room_spawn_seed": 9},
        )
        with self.assertRaisesRegex(QueryError, "请在问题中说出道具名称"):
            self._engine().ask("这个道具值得拿吗", state, "request-5")

    def test_memory_closed_loop_is_structured_bounded_and_does_not_replace_rag(self) -> None:
        adapter = DynamicGroundedAdapter()
        router = ModelRouter(
            self.config, online=True, api_key="test-only",
            adapters={"advice": adapter},
        )
        state = GameState(
            run_id="MEMORY QUERY:0", active=True, last_seq=12,
            context={"room_index": 4, "room_spawn_seed": 9},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            raw = self._engine(router)
            engine = MemoryAwareQueryEngine(
                raw, store, recall_max_items=2, recall_max_chars=80
            )
            first, _token = engine.ask(
                "我喜欢硫磺火，硫磺火有什么效果", state, "memory-request-1"
            )
            self.assertTrue(first.sources)
            self.assertEqual(store.list_items()[0].content, "喜欢硫磺火")
            second, _token = engine.ask(
                "硫磺火有什么效果", state, "memory-request-2"
            )
            prompt = json.loads(adapter.requests[-1].user_prompt)
            self.assertEqual(prompt["long_term_memory"][0]["content"], "喜欢硫磺火")
            self.assertLessEqual(len(prompt["long_term_memory"]), 2)
            self.assertIn("不得覆盖用户当前表达", adapter.requests[-1].system_prompt)
            self.assertTrue({source.id for source in second.sources} <= {
                hit.chunk.source.id for hit in second.rag_hits
            })
            store.close()

    def test_disabled_and_pending_memories_never_enter_query_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            disabled = store.add(MemoryCandidate("喜欢硫磺火", "手动"))
            store.set_item_enabled(disabled.id, False)
            store.add(MemoryCandidate(
                "可能喜欢高风险路线", "行为推断", confidence=0.5,
                confirmation_level="inferred",
            ))
            self.assertEqual(store.recall("硫磺火高风险路线").items, ())
            store.close()

    def test_wrong_character_and_active_item_names_are_rejected_to_local_summary(self) -> None:
        live_config = load_config(Path("config/rag-v2.1-faiss.toml"))
        invalid = json.dumps({
            "advice": "当前角色为以实玛利，携带主动道具节拍器，建议拿取。",
            "reason": "错误地猜测了裸数字 ID。",
            "confidence": 0.9,
            "sources": ["huiji:isaac:page:3370:rev:166661"],
            "state_seq": 17,
        }, ensure_ascii=False)
        router = ModelRouter(
            live_config, online=True, api_key="test-only",
            adapters={"advice": StaticAdapter(invalid)},
        )
        state = GameState(
            run_id="QUERY:0", active=True, last_seq=17,
            context={"room_index": 4, "room_spawn_seed": 9},
            players={"0": {
                "controller_index": 0, "player_type": 0,
                "inventory": {"active_item": 105},
            }},
            room_collectibles=[{"collectible_id": 161, "taken": False}],
        )
        budget = BudgetTracker(live_config.budget.run_limit_cny)
        budget.set_run("QUERY:0")
        engine = QueryEngine(
            RagService(live_config.rag.index_path), router, budget,
            game_version=live_config.rag.game_version,
        )

        response, _token = engine.ask("这个道具拿吗？", state, "request-6")

        self.assertTrue(response.simulated)
        self.assertNotIn("以实玛利", response.answer)
        self.assertNotIn("节拍器", response.answer)
        self.assertIn("未通过本地游戏状态校验", response.delivery_note or "")


if __name__ == "__main__":
    unittest.main()
