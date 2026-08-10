"""完全离线、固定实体与证据目标的检索评测。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .rag import RagFilters, RagService


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    eval_id: str
    mode: str
    case_count: int
    recall_at_k: float
    mrr: float
    no_answer_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    passed: bool
    cases: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "mode": self.mode,
            "case_count": self.case_count,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "no_answer_accuracy": self.no_answer_accuracy,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "passed": self.passed,
            "cases": list(self.cases),
        }


def evaluate(service: RagService, eval_path: Path) -> EvaluationReport:
    suite = json.loads(eval_path.read_text(encoding="utf-8"))
    top_k = int(suite["top_k"])
    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    no_answer_checks: list[float] = []
    latencies: list[float] = []
    used_vector = False
    for case in suite["cases"]:
        filters_raw = case.get("filters", {})
        filters = RagFilters(
            entity_types=tuple(filters_raw.get("entity_types", ())),
            game_version=filters_raw.get("game_version"),
            source_types=tuple(filters_raw.get("source_types", ())),
        )
        result = service.retrieve(case["query"], filters=filters, top_k=top_k)
        used_vector = used_vector or not result.degraded
        actual_entities = [hit.chunk.entity_id for hit in result.hits]
        actual_sources = [hit.chunk.source.id for hit in result.hits]
        expected_entities = list(case["expected_entities"])
        expected_sources = list(case["expected_sources"])
        expect_no_answer = bool(case.get("expect_no_answer", False))
        if expect_no_answer:
            no_answer_checks.append(float(result.no_answer))
            recall = 1.0 if result.no_answer else 0.0
            rr = recall
        else:
            entity_matches = set(expected_entities) & set(actual_entities)
            source_matches = set(expected_sources) & set(actual_sources)
            recall = min(
                len(entity_matches) / max(1, len(expected_entities)),
                len(source_matches) / max(1, len(expected_sources)),
            )
            ranks = [actual_entities.index(entity) + 1 for entity in expected_entities if entity in actual_entities]
            rr = 1.0 / min(ranks) if ranks else 0.0
        recalls.append(recall)
        reciprocal_ranks.append(rr)
        latencies.append(result.latency_ms)
        details.append(
            {
                "id": case["id"],
                "passed": recall == 1.0,
                "expected_entities": expected_entities,
                "actual_entities": actual_entities,
                "actual_sources": actual_sources,
                "latency_ms": result.latency_ms,
                "degraded": result.degraded,
            }
        )
    recall_at_k = mean(recalls)
    mrr = mean(reciprocal_ranks)
    no_answer_accuracy = mean(no_answer_checks) if no_answer_checks else 1.0
    ordered = sorted(latencies)
    p95 = ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))]
    thresholds = suite["thresholds"]
    latency_threshold = (
        thresholds["hybrid_p95_latency_ms"]
        if used_vector
        else thresholds["keyword_p95_latency_ms"]
    )
    passed = (
        recall_at_k >= thresholds["recall_at_k"]
        and mrr >= thresholds["mrr"]
        and no_answer_accuracy >= thresholds["no_answer_accuracy"]
        and p95 <= latency_threshold
    )
    return EvaluationReport(
        suite["eval_id"], "hybrid" if used_vector else "keyword", len(details), recall_at_k, mrr, no_answer_accuracy,
        mean(latencies), p95, passed, tuple(details)
    )
