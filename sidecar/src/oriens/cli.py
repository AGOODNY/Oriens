"""Oriens 命令行入口。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import sys
import time
from typing import TextIO

from .advice import AdviceEngine
from .budget import BudgetTracker
from .config import ConfigError, OriensConfig, load_api_key, load_config
from .knowledge import KnowledgeError, LocalItemKnowledgeBase
from .modeling import ModelRouter
from .protocol import EventParseError, GameEvent, parse_event_line
from .rag import RagError, RagFilters, RagService
from .rag_eval import evaluate
from .rag_pipeline import (
    CorpusValidationError,
    build_corpus,
    build_keyword_index,
    load_chunks,
)
from .rag_worker import VectorWorkerClient
from .state import EventOrderError, StateStore
from .tailer import LogTailer


def default_log_path() -> Path:
    return (
        Path.home()
        / "Documents"
        / "My Games"
        / "Binding of Isaac Repentance+"
        / "log.txt"
    )


def _json_output(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_doctor(args: argparse.Namespace) -> int:
    log_path = args.log.resolve()
    _json_output(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "log_path": str(log_path),
            "log_exists": log_path.exists(),
            "log_size": log_path.stat().st_size if log_path.exists() else None,
        }
    )
    return 0 if log_path.exists() else 1


def _handle_line(
    line: str,
    store: StateStore,
    recorder: TextIO | None,
    *,
    verbose: bool,
) -> None:
    try:
        event = parse_event_line(line)
    except EventParseError as exc:
        store.mark_invalid()
        print(f"协议错误：{exc}", file=sys.stderr)
        return
    if event is None:
        store.mark_ignored()
        return
    try:
        store.apply(event)
    except EventOrderError as exc:
        print(f"顺序错误：{exc}", file=sys.stderr)
        return
    if recorder is not None:
        recorder.write(event.to_json() + "\n")
        recorder.flush()
    if verbose:
        print(
            f"事件 seq={event.seq} run={event.run_id} "
            f"frame={event.game_frame} type={event.type}"
        )


def run_listen(args: argparse.Namespace) -> int:
    log_path = args.log.resolve()
    if args.record is not None:
        args.record.resolve().parent.mkdir(parents=True, exist_ok=True)
        record_context = args.record.resolve().open("a", encoding="utf-8", newline="\n")
    else:
        record_context = nullcontext(None)

    tailer = LogTailer(log_path, from_start=args.from_start)
    store = StateStore()
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    print(f"正在监听：{log_path}")
    if args.record is not None:
        print(f"正在录制：{args.record.resolve()}")

    try:
        with record_context as recorder:
            while deadline is None or time.monotonic() < deadline:
                poll = tailer.poll()
                if poll.reopened:
                    store.diagnostics.log_reopens += 1
                for line in poll.lines:
                    _handle_line(line, store, recorder, verbose=not args.quiet)
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("监听已停止。")
    finally:
        tailer.close()

    _json_output(store.snapshot())
    diagnostics = store.diagnostics
    return 2 if diagnostics.invalid_events or diagnostics.out_of_order_events else 0


def run_replay(args: argparse.Namespace) -> int:
    store = StateStore()
    with args.recording.resolve().open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            _handle_line(line, store, None, verbose=args.verbose)
    _json_output(store.snapshot())
    diagnostics = store.diagnostics
    return 2 if diagnostics.invalid_events or diagnostics.out_of_order_events else 0


def _make_advice_services(
    config_path: Path | None,
    *,
    online: bool,
    enable_vector: bool = False,
) -> tuple[OriensConfig, LocalItemKnowledgeBase, BudgetTracker, AdviceEngine, bool]:
    config = load_config(config_path)
    provider, _model = config.provider_for("advice")
    api_key = load_api_key(provider.api_key_env)
    router = ModelRouter(config, online=online, api_key=api_key)
    knowledge = LocalItemKnowledgeBase.load(config.app.knowledge_path)
    _ensure_rag_index(config)
    vector = _make_vector_client(config) if enable_vector else None
    if vector is not None:
        vector.start()
    rag = RagService(
        config.rag.index_path,
        vector,
        vector_min_similarity=config.rag.vector_min_similarity,
    )
    budget = BudgetTracker(config.budget.run_limit_cny)
    engine = AdviceEngine(
        knowledge, router, budget, rag=rag, game_version=config.rag.game_version
    )
    return config, knowledge, budget, engine, bool(api_key)


def _ensure_rag_index(config: OriensConfig) -> None:
    needs_build = not config.rag.index_path.is_file()
    if config.rag.source_path.is_file() and config.rag.index_path.is_file():
        needs_build = config.rag.source_path.stat().st_mtime_ns > config.rag.index_path.stat().st_mtime_ns
    if needs_build:
        chunks = build_corpus(
            config.rag.source_path, config.rag.chunks_path, config.rag.manifest_path
        )
        build_keyword_index(chunks, config.rag.index_path)


def _make_vector_client(config: OriensConfig) -> VectorWorkerClient | None:
    if not config.rag.vector_enabled:
        return None
    return VectorWorkerClient(
        index_path=config.rag.index_path,
        model_path=config.rag.vector_model_path,
        dimension=config.rag.vector_dimension,
        request_timeout_seconds=config.rag.vector_query_timeout_seconds,
    )


def _demo_event(collectible_id: int, seq: int = 1) -> GameEvent:
    return GameEvent(
        schema_version=1,
        seq=seq,
        run_id="ORIENS DEMO:0",
        type="collectible_spawned",
        game_frame=100,
        context={
            "stage": 1,
            "room_index": 4,
            "room_type": 4,
            "room_spawn_seed": 20260810,
            "room_clear": True,
        },
        payload={"collectible_id": collectible_id, "price": 0, "init_seed": 1},
    )


def run_advice_demo(args: argparse.Namespace) -> int:
    try:
        _config, knowledge, budget, engine, _key_available = _make_advice_services(
            args.config, online=False
        )
        if knowledge.find(args.collectible_id) is None:
            print(f"本地资料未覆盖道具 ID：{args.collectible_id}", file=sys.stderr)
            return 1
        budget.set_run("ORIENS DEMO:0")
        response, _token = engine.generate(_demo_event(args.collectible_id))
    except (ConfigError, KnowledgeError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    _json_output(response.as_dict())
    return 0


def run_api_smoke(args: argparse.Namespace) -> int:
    if not args.confirm_charge:
        print(
            "已阻止联网调用：请在了解可能产生少量费用后添加 --confirm-charge。",
            file=sys.stderr,
        )
        return 1
    try:
        config, knowledge, budget, engine, key_available = _make_advice_services(
            args.config, online=True
        )
        if not key_available:
            print("未找到 DASHSCOPE_API_KEY，未发起联网请求。", file=sys.stderr)
            return 1
        if knowledge.find(args.collectible_id) is None:
            print(f"本地资料未覆盖道具 ID：{args.collectible_id}", file=sys.stderr)
            return 1
        _provider, model = config.provider_for("advice")
        print(
            f"将调用 {model.display_name} 一次；预计 5–15 秒，通常费用低于人民币 0.001 元。"
        )
        budget.set_run("ORIENS API SMOKE:0")
        response, _token = engine.generate(_demo_event(args.collectible_id))
    except (ConfigError, KnowledgeError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    _json_output(response.as_dict())
    return 0 if not response.simulated else 2


def run_ui(args: argparse.Namespace) -> int:
    try:
        config, knowledge, budget, engine, key_available = _make_advice_services(
            args.config, online=args.online, enable_vector=True
        )
    except (ConfigError, KnowledgeError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    try:
        from .ui import run_overlay
    except ImportError:
        print(
            "缺少 PySide6。请按 README 的项目内虚拟环境步骤安装依赖。",
            file=sys.stderr,
        )
        return 1
    return run_overlay(
        config=config,
        log_path=args.log.resolve(),
        knowledge=knowledge,
        advice_engine=engine,
        budget=budget,
        from_start=args.from_start,
        online_requested=args.online,
        api_key_available=key_available,
    )


def run_rag_build(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        chunks = build_corpus(
            config.rag.source_path, config.rag.chunks_path, config.rag.manifest_path
        )
        build_keyword_index(chunks, config.rag.index_path)
        report: dict[str, object] = {
            "documents": len(chunks),
            "chunks": len(chunks),
            "keyword_index": str(config.rag.index_path),
            "vector_index": "未请求",
        }
        if args.with_vectors:
            worker = _make_vector_client(config)
            if worker is None:
                raise RagError("配置已关闭向量索引")
            try:
                vector_report = worker.build(chunks)
                report["vector_index"] = {
                    "count": vector_report["vector_count"],
                    "model_init_ms": worker.init_latency_ms,
                    "build_latency_ms": vector_report["build_latency_ms"],
                }
            finally:
                worker.close()
    except (ConfigError, CorpusValidationError, RagError, RuntimeError, TimeoutError) as exc:
        print(f"RAG 构建失败：{exc}", file=sys.stderr)
        return 1
    _json_output(report)
    return 0


def run_rag_query(args: argparse.Namespace) -> int:
    worker: VectorWorkerClient | None = None
    try:
        config = load_config(args.config)
        _ensure_rag_index(config)
        if args.with_vectors:
            worker = _make_vector_client(config)
            if worker is not None:
                worker.start()
                worker.wait_ready(args.worker_timeout)
        service = RagService(
            config.rag.index_path,
            worker,
            vector_min_similarity=config.rag.vector_min_similarity,
        )
        filters = RagFilters(
            entity_types=tuple(args.entity_type or ()),
            game_version=args.game_version,
            source_types=tuple(args.source_type or ()),
        )
        result = service.retrieve(args.query, filters=filters, top_k=args.top_k)
        _json_output(result.as_dict())
        return 2 if result.no_answer else 0
    except (ConfigError, RagError, CorpusValidationError) as exc:
        print(f"RAG 查询失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()


def run_rag_eval(args: argparse.Namespace) -> int:
    worker: VectorWorkerClient | None = None
    try:
        config = load_config(args.config)
        _ensure_rag_index(config)
        if args.with_vectors:
            worker = _make_vector_client(config)
            if worker is not None:
                worker.start()
                worker.wait_ready(args.worker_timeout)
        service = RagService(
            config.rag.index_path,
            worker,
            vector_min_similarity=config.rag.vector_min_similarity,
        )
        report = evaluate(service, config.rag.eval_path)
        _json_output(report.as_dict())
        return 0 if report.passed else 2
    except (ConfigError, RagError, CorpusValidationError) as exc:
        print(f"RAG 评测失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()


def run_rag_benchmark(args: argparse.Namespace) -> int:
    worker: VectorWorkerClient | None = None
    try:
        import psutil

        config = load_config(args.config)
        worker = _make_vector_client(config)
        if worker is None:
            raise RagError("配置已关闭向量 Worker")
        started = time.perf_counter()
        worker.start()
        if not worker.wait_ready(args.worker_timeout):
            raise RagError(worker.unavailable_reason or "向量 Worker 不可用")
        init_ms = (time.perf_counter() - started) * 1000
        process = psutil.Process(worker.pid)
        latencies: list[float] = []
        cpu_before = process.cpu_times()
        process.cpu_percent(None)
        query_batch_started = time.perf_counter()
        for query in ("硫磺火", "豆浆和石底", "怎么去祸兽", "Jacob and Esau"):
            query_started = time.perf_counter()
            worker.search(query, 5)
            latencies.append((time.perf_counter() - query_started) * 1000)
        query_batch_seconds = time.perf_counter() - query_batch_started
        cpu_percent = process.cpu_percent(None)
        cpu_after = process.cpu_times()
        cpu_seconds = (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        )
        memory = process.memory_info().rss
        _json_output(
            {
                "model": config.rag.vector_model_id,
                "worker_pid": worker.pid,
                "initialization_ms": round(init_ms, 3),
                "query_latency_ms": [round(value, 3) for value in latencies],
                "mean_query_latency_ms": round(sum(latencies) / len(latencies), 3),
                "query_batch_seconds": round(query_batch_seconds, 3),
                "query_cpu_seconds": round(cpu_seconds, 3),
                "query_cpu_percent": round(cpu_percent, 1),
                "worker_rss_bytes": memory,
                "worker_rss_mib": round(memory / 1024 / 1024, 2),
            }
        )
        return 0
    except (ImportError, ConfigError, RagError, RuntimeError, TimeoutError) as exc:
        print(f"RAG 性能测试失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oriens 游戏日志伴侣")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查日志路径和 Python 环境")
    doctor.add_argument("--log", type=Path, default=default_log_path())
    doctor.set_defaults(handler=run_doctor)

    listen = subparsers.add_parser("listen", help="监听游戏日志并重建状态")
    listen.add_argument("--log", type=Path, default=default_log_path())
    listen.add_argument("--record", type=Path)
    listen.add_argument("--from-start", action="store_true")
    listen.add_argument("--duration", type=float, default=0.0, help="运行秒数；0 表示持续监听")
    listen.add_argument("--poll-interval", type=float, default=0.1)
    listen.add_argument("--quiet", action="store_true")
    listen.set_defaults(handler=run_listen)

    replay = subparsers.add_parser("replay", help="离线回放 JSONL 录制")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--verbose", action="store_true")
    replay.set_defaults(handler=run_replay)

    ui = subparsers.add_parser("ui", help="启动阶段 1 简体中文悬浮窗")
    ui.add_argument("--log", type=Path, default=default_log_path())
    ui.add_argument("--config", type=Path)
    ui.add_argument("--from-start", action="store_true")
    ui.add_argument(
        "--online",
        action="store_true",
        help="显式启用百炼；未指定或缺少密钥时使用零费用模拟模型",
    )
    ui.set_defaults(handler=run_ui)

    demo = subparsers.add_parser("advice-demo", help="使用模拟模型生成固定道具建议")
    demo.add_argument("collectible_id", type=int, nargs="?", default=350)
    demo.add_argument("--config", type=Path)
    demo.set_defaults(handler=run_advice_demo)

    smoke = subparsers.add_parser("api-smoke", help="执行一次最小额度百炼联网烟雾测试")
    smoke.add_argument("collectible_id", type=int, nargs="?", default=350)
    smoke.add_argument("--config", type=Path)
    smoke.add_argument("--confirm-charge", action="store_true")
    smoke.set_defaults(handler=run_api_smoke)

    rag_build = subparsers.add_parser("rag-build", help="清洗语料并构建本地检索索引")
    rag_build.add_argument("--config", type=Path)
    rag_build.add_argument("--with-vectors", action="store_true", help="使用 BGE-M3 构建 sqlite-vec 索引")
    rag_build.set_defaults(handler=run_rag_build)

    rag_query = subparsers.add_parser("rag-query", help="查询本地 RAG")
    rag_query.add_argument("query")
    rag_query.add_argument("--config", type=Path)
    rag_query.add_argument("--entity-type", action="append")
    rag_query.add_argument("--source-type", action="append")
    rag_query.add_argument("--game-version")
    rag_query.add_argument("--top-k", type=int, default=5)
    rag_query.add_argument("--with-vectors", action="store_true")
    rag_query.add_argument("--worker-timeout", type=float, default=600.0)
    rag_query.set_defaults(handler=run_rag_query)

    rag_eval = subparsers.add_parser("rag-eval", help="运行完全离线的固定检索评测")
    rag_eval.add_argument("--config", type=Path)
    rag_eval.add_argument("--with-vectors", action="store_true")
    rag_eval.add_argument("--worker-timeout", type=float, default=600.0)
    rag_eval.set_defaults(handler=run_rag_eval)

    rag_benchmark = subparsers.add_parser("rag-benchmark", help="测量 BGE-M3 Worker 的本机性能")
    rag_benchmark.add_argument("--config", type=Path)
    rag_benchmark.add_argument("--worker-timeout", type=float, default=600.0)
    rag_benchmark.set_defaults(handler=run_rag_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
