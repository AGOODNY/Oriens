from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from oriens.advice import AdviceEngine
from oriens.budget import BudgetTracker
from oriens.config import load_config
from oriens.knowledge import LocalItemKnowledgeBase
from oriens.modeling import ModelRouter
from oriens.protocol import GameEvent
from oriens.rag import RagFilters, RagService
from oriens.rag_eval import evaluate
from oriens.rag_pipeline import build_corpus, build_keyword_index


class _UnavailableVector:
    available = False
    unavailable_reason = "测试 Worker 不可用"

    def search(self, query: str, limit: int):
        raise AssertionError("不可用 Worker 不应收到查询")


class _FakeVector:
    available = True
    unavailable_reason = None

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def search(self, query: str, limit: int):
        self.calls += 1
        return self.rows[:limit]


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.chunks_path = root / "chunks.jsonl"
        cls.manifest_path = root / "manifest.json"
        cls.index_path = root / "rag.sqlite"
        chunks = build_corpus(
            cls.config.rag.source_path, cls.chunks_path, cls.manifest_path
        )
        build_keyword_index(chunks, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_pipeline_records_required_governance_fields_and_checksums(self) -> None:
        chunks = [json.loads(line) for line in self.chunks_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(chunks), 37)
        required = {
            "chunk_id", "document_id", "entity_type", "entity_id", "name_zh",
            "name_en", "aliases", "source", "acquired_on", "game_version",
            "license_note", "content_version", "checksum", "stale",
        }
        for chunk in chunks:
            self.assertTrue(required <= set(chunk))
            self.assertTrue(chunk["checksum"].startswith("sha256:"))
            self.assertTrue(chunk["source"]["url"].startswith("https://"))

    def test_exact_id_chinese_english_and_alias_queries(self) -> None:
        service = RagService(self.index_path)
        for query in ("118", "硫磺火", "Brimstone", "brim"):
            result = service.retrieve(query, top_k=3)
            self.assertFalse(result.no_answer, query)
            self.assertEqual(result.hits[0].chunk.entity_id, "collectible:118", query)
            self.assertIn("exact", result.hits[0].methods)

    def test_synergy_character_route_and_room_aliases(self) -> None:
        service = RagService(self.index_path)
        expected = {
            "豆浆和石底": "synergy:collectible:330+562",
            "双子": "player:19",
            "怎么去祸兽": "route:beast",
            "恶魔交易": "room:14",
        }
        for query, entity_id in expected.items():
            self.assertEqual(service.retrieve(query).hits[0].chunk.entity_id, entity_id)

    def test_metadata_filters_and_version_mismatch(self) -> None:
        service = RagService(self.index_path)
        item = service.retrieve("硫磺火", filters=RagFilters(entity_types=("item",)))
        self.assertTrue(item.hits)
        self.assertTrue(all(hit.chunk.entity_type == "item" for hit in item.hits))
        mismatch = service.retrieve(
            "硫磺火", filters=RagFilters(game_version="Repentance 1.7.9")
        )
        self.assertTrue(mismatch.no_answer)
        wrong_source = service.retrieve(
            "Brimstone", filters=RagFilters(source_types=("official",))
        )
        self.assertTrue(wrong_source.no_answer)

    def test_unavailable_vector_degrades_to_keyword_without_crashing(self) -> None:
        result = RagService(self.index_path, _UnavailableVector()).retrieve("妈刀")
        self.assertTrue(result.degraded)
        self.assertEqual(result.hits[0].chunk.entity_id, "collectible:114")

    def test_vector_candidates_are_filtered_and_merged_deterministically(self) -> None:
        baseline = RagService(self.index_path).retrieve("剖腹产").hits[0].chunk.chunk_id
        vector = _FakeVector([(baseline, 0.91)])
        result = RagService(self.index_path, vector).retrieve("剖腹产")
        self.assertEqual(result.hits[0].chunk.chunk_id, baseline)
        self.assertEqual(result.hits[0].methods, ("exact", "vector"))

    def test_low_confidence_vector_only_candidate_is_treated_as_no_answer(self) -> None:
        known_chunk = RagService(self.index_path).retrieve("Brimstone").hits[0].chunk.chunk_id
        vector = _FakeVector([(known_chunk, 0.40)])
        result = RagService(self.index_path, vector).retrieve("不存在的语义问题")
        self.assertTrue(result.no_answer)
        self.assertEqual(vector.calls, 1)

    def test_unknown_explicit_entity_id_does_not_use_semantic_fallback(self) -> None:
        known_chunk = RagService(self.index_path).retrieve("Brimstone").hits[0].chunk.chunk_id
        vector = _FakeVector([(known_chunk, 0.99)])
        result = RagService(self.index_path, vector).retrieve("collectible:999999")
        self.assertTrue(result.no_answer)

    def test_unknown_extended_stable_id_does_not_use_fts_fallback(self) -> None:
        service = RagService(self.index_path)
        result = service.retrieve("room-layout:not-real")
        self.assertTrue(result.no_answer)

    def test_explicit_third_party_mod_query_is_out_of_scope(self) -> None:
        service = RagService(self.index_path)
        result = service.retrieve("Fiend Folio 的某个模组道具")
        self.assertTrue(result.no_answer)

    def test_fixed_offline_eval_meets_recorded_thresholds(self) -> None:
        report = evaluate(RagService(self.index_path), self.config.rag.eval_path)
        failures = [case["id"] for case in report.cases if not case["passed"]]
        self.assertTrue(report.passed, failures)
        self.assertGreaterEqual(report.recall_at_k, 0.90)
        self.assertEqual(report.no_answer_accuracy, 1.0)

    def test_advice_for_new_item_only_cites_retrieved_evidence(self) -> None:
        knowledge = LocalItemKnowledgeBase.load(self.config.app.knowledge_path)
        budget = BudgetTracker(self.config.budget.run_limit_cny)
        budget.set_run("RAG TEST:0")
        engine = AdviceEngine(
            knowledge,
            ModelRouter(self.config, online=False, api_key=None),
            budget,
            rag=RagService(self.index_path),
            game_version=self.config.rag.game_version,
        )
        event = GameEvent(
            1, 1, "RAG TEST:0", "collectible_spawned", 100,
            {"stage": 1, "room_index": 4, "room_type": 4, "room_spawn_seed": 7},
            {"collectible_id": 118, "init_seed": 1, "price": 0},
        )
        response, _token = engine.generate(event)
        retrieved_sources = {hit.chunk.source.id for hit in response.rag_hits}
        self.assertEqual(response.collectible_id, 118)
        self.assertTrue(retrieved_sources)
        self.assertTrue({source.id for source in response.sources} <= retrieved_sources)
        self.assertEqual(response.retrieval_corpus_version, "1.0.0")
        self.assertTrue(response.retrieval_degraded)
        self.assertIsNotNone(response.retrieval_degradation_reason)


if __name__ == "__main__":
    unittest.main()
