from __future__ import annotations

import json
from pathlib import Path
from threading import Event
import tempfile
import unittest

from oriens.budget import BudgetTracker
from oriens.config import load_config
from oriens.modeling import AdapterResponse, ModelRouter, ModelUsage
from oriens.query import QueryEngine, QueryValidationError
from oriens.rag import RagService
from oriens.rag_pipeline import build_corpus, build_keyword_index
from oriens.state import GameState


class StaticAdapter:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, model, model_request, cancel: Event):
        return AdapterResponse(self.content, ModelUsage(20, 10))


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


if __name__ == "__main__":
    unittest.main()
