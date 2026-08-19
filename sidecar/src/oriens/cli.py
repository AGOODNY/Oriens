"""Oriens 命令行入口。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import platform
import sys
from threading import Event
import time
from typing import TextIO

from .advice import AdviceEngine
from .application import LaunchOptions, OriensApplication, default_log_path
from .audio import AudioChunk, AudioFormat, MemoryMicrophone, QueuedAudioPlayer
from .budget import BudgetTracker
from .config import ConfigError, OriensConfig, load_api_key, load_config
from .knowledge import KnowledgeError, LocalItemKnowledgeBase
from .knowledge_pack import KnowledgePackError
from .modeling import ModelRouter
from .query import QueryEngine
from .protocol import EventParseError, GameEvent, parse_event_line
from .rag import RagError, RagFilters, RagService
from .rag_eval import evaluate
from .rag_pipeline import (
    CorpusValidationError,
    build_corpus,
    build_keyword_index,
    iter_chunks,
    load_chunks,
)
from .rag_v2_pipeline import RagV2Paths, build_full_corpus
from .rag_worker import VectorWorkerClient, convert_sqlite_vectors_to_faiss
from .state import EventOrderError, StateStore
from .tailer import LogTailer
from .voice import CosyVoiceStreamingTTS, QwenRealtimeASR, VoiceError
from .voice import MockRealtimeASR, MockStreamingTTS, TerminologyCorrector
from .voice_service import VoiceCallbacks, VoiceService
from .paths import AppPaths


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


def _ensure_rag_index(config: OriensConfig) -> None:
    if config.rag.pipeline_version == 2:
        if not config.rag.index_path.is_file():
            raise RagError(
                "rag-v2 索引尚未构建；请先在命令行运行 oriens rag-build"
            )
        return
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
        backend=config.rag.vector_backend,
        vector_index_path=config.rag.vector_index_path,
        dimension=config.rag.vector_dimension,
        batch_size=config.rag.vector_batch_size,
        max_sequence_length=config.rag.vector_max_sequence_length,
        device=config.rag.vector_device,
        build_timeout_seconds=config.rag.vector_build_timeout_seconds,
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
    application = None
    try:
        application = OriensApplication.build(
            AppPaths.development(),
            LaunchOptions(config_path=args.config, online=False, enable_vector=False),
        )
        if application.knowledge.find(args.collectible_id) is None:
            print(f"本地资料未覆盖道具 ID：{args.collectible_id}", file=sys.stderr)
            return 1
        application.budget.set_run("ORIENS DEMO:0")
        response, _token = application.advice_engine.generate(_demo_event(args.collectible_id))
    except (ConfigError, KnowledgeError, KnowledgePackError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if application is not None:
            application.close()
    _json_output(response.as_dict())
    return 0


def run_api_smoke(args: argparse.Namespace) -> int:
    if not args.confirm_charge:
        print(
            "已阻止联网调用：请在了解可能产生少量费用后添加 --confirm-charge。",
            file=sys.stderr,
        )
        return 1
    application = None
    try:
        application = OriensApplication.build(
            AppPaths.development(),
            LaunchOptions(config_path=args.config, online=True, enable_vector=False),
        )
        if not application.api_key_available:
            print("未找到 DASHSCOPE_API_KEY，未发起联网请求。", file=sys.stderr)
            return 1
        if application.knowledge.find(args.collectible_id) is None:
            print(f"本地资料未覆盖道具 ID：{args.collectible_id}", file=sys.stderr)
            return 1
        _provider, model = application.config.provider_for("advice")
        print(
            f"将调用 {model.display_name} 一次；预计 5–15 秒，通常费用低于人民币 0.001 元。"
        )
        application.budget.set_run("ORIENS API SMOKE:0")
        response, _token = application.advice_engine.generate(_demo_event(args.collectible_id))
    except (ConfigError, KnowledgeError, KnowledgePackError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if application is not None:
            application.close()
    _json_output(response.as_dict())
    return 0 if not response.simulated else 2


def run_desktop_command(args: argparse.Namespace) -> int:
    try:
        application = OriensApplication.build(
            AppPaths.development(),
            LaunchOptions(
                config_path=args.config,
                log_path=args.log,
                from_start=args.from_start,
                online=args.online,
                enable_vector=True,
            ),
        )
    except (ConfigError, KnowledgeError, KnowledgePackError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    try:
        from .desktop import run_desktop
    except ImportError:
        application.close()
        print(
            "缺少 PySide6。请按 README 的项目内虚拟环境步骤安装依赖。",
            file=sys.stderr,
        )
        return 1
    try:
        return run_desktop(application=application)
    finally:
        application.close()


def run_voice_benchmark(args: argparse.Namespace) -> int:
    """阶段 3 零费用模拟闭环；指标明确不代表真实设备或百炼网络。"""

    if args.iterations <= 0 or args.stress_cycles <= 0 or args.timeout <= 0:
        print("模拟语音基准参数必须为正数", file=sys.stderr)
        return 1

    application = None
    try:
        application = OriensApplication.build(
            AppPaths.development(),
            LaunchOptions(config_path=args.config, online=False, enable_vector=True),
        )
        config = application.config
        budget = application.budget
        query = application.query_engine
    except (ConfigError, KnowledgeError, KnowledgePackError, RagError, CorpusValidationError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    try:
        import psutil
    except ImportError:
        psutil = None
    state = StateStore().state
    state.run_id = "VOICE BENCHMARK:0"
    state.active = True
    state.last_seq = 1
    state.context = {"room_index": 4, "room_spawn_seed": 20260812, "stage": 1}
    budget.set_run(state.run_id)
    microphone = MemoryMicrophone()
    played: list[AudioChunk] = []
    player = QueuedAudioPlayer(played.append, config.audio.playback_queue_max_chunks)
    completed = []
    errors: list[str] = []
    service = VoiceService(
        audio_settings=config.audio,
        voice_settings=config.voice,
        microphone=microphone,
        player=player,
        asr=MockRealtimeASR("硫磺火有什么效果", ("硫磺火",)),
        tts=MockStreamingTTS(AudioFormat(config.audio.playback_sample_rate)),
        query_engine=query,
        terminology=TerminologyCorrector({}),
        state_provider=lambda: state,
        callbacks=VoiceCallbacks(
            on_state=lambda _request_id, _value: None,
            on_transcript=lambda _request_id, _value: None,
            on_question=lambda _request_id, _value: None,
            on_answer=lambda _request_id, _value, _token: None,
            on_error=lambda _request_id, value: errors.append(value),
            on_metrics=lambda _request_id, value: completed.append(value),
        ),
    )
    process = psutil.Process() if psutil is not None else None
    cpu_before = _process_cpu_seconds(process)
    peak_rss = _process_rss(process)
    fmt = AudioFormat(config.audio.input_sample_rate)
    generated = AudioChunk(b"\x10\x04" * 4800, fmt, 0)
    try:
        for index in range(args.iterations):
            before = len(completed)
            if service.press("memory") is None:
                raise RuntimeError("模拟麦克风启动失败")
            microphone.feed(generated)
            service.release()
            deadline = time.monotonic() + args.timeout
            while len(completed) == before and time.monotonic() < deadline:
                time.sleep(0.005)
            if len(completed) == before:
                raise RuntimeError(f"第 {index + 1} 次模拟闭环超时")
            peak_rss = max(peak_rss, _process_rss(process))

        interrupt_values = []
        for _ in range(args.stress_cycles):
            service.press("memory")
            microphone.feed(generated)
            started = time.perf_counter()
            service.cancel()
            interrupt_values.append((time.perf_counter() - started) * 1000)
        peak_rss = max(peak_rss, _process_rss(process))
        cpu_after = _process_cpu_seconds(process)
        fields = {
            "capture_start_ms": [item.capture_start_ms for item in completed],
            "asr_first_partial_ms": [item.asr_first_partial_ms for item in completed],
            "asr_final_ms": [item.asr_final_ms for item in completed],
            "rag_ms": [item.rag_ms for item in completed],
            "model_text_ms": [item.model_first_text_ms for item in completed],
            "tts_first_audio_ms": [item.tts_first_audio_ms for item in completed],
            "first_audio_end_to_end_ms": [item.first_audio_end_to_end_ms for item in completed],
            "interrupt_ms": interrupt_values,
        }
        report = {
            "mode": "simulation",
            "network_calls": 0,
            "config_source": "explicit" if args.config else "default",
            "corpus_version": config.rag.content_version,
            "vector_backend": config.rag.vector_backend,
            "iterations": args.iterations,
            "stress_cycles": args.stress_cycles,
            "metrics": {
                name: {
                    "p50": round(_percentile(values, 0.50), 3),
                    "p95": round(_percentile(values, 0.95), 3),
                    "max": round(max(values), 3),
                }
                for name, values in fields.items()
            },
            "cpu_seconds": round(max(0.0, cpu_after - cpu_before), 3),
            "peak_rss_bytes": peak_rss,
            "queue_peak_chunks": max((item.queue_peak for item in completed), default=0),
            "errors": errors,
        }
        _json_output(report)
        return 0 if not errors else 2
    except RuntimeError as exc:
        print(f"模拟语音基准失败：{exc}", file=sys.stderr)
        return 1
    finally:
        service.close()
        if application is not None:
            application.close()


def run_realtime_benchmark(args: argparse.Namespace) -> int:
    """零费用虚拟时钟对比；只验证指标管线，不声称代表真实网络。"""

    if args.iterations <= 0:
        print("Realtime 模拟基准迭代次数必须为正数", file=sys.stderr)
        return 1
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    questions = (
        "当前房间应该先做什么？",
        "硫磺火和豆浆有什么协同？",
        "我偏好稳健路线，该怎么选？",
    )
    profiles = {
        "chain": {
            "first_transcript_ms": 42.0,
            "first_text_ms": 96.0,
            "first_audio_ms": 132.0,
            "interrupt_ms": 1.4,
            "tool_call_ms": 7.0,
            "total_ms": 158.0,
            "queue_peak_chunks": 5,
            "fallbacks": 0,
            "estimated_cost_cny": 0.0,
        },
        "realtime": {
            "first_transcript_ms": 18.0,
            "first_text_ms": 64.0,
            "first_audio_ms": 86.0,
            "interrupt_ms": 0.9,
            "tool_call_ms": 8.0,
            "total_ms": 112.0,
            "queue_peak_chunks": 4,
            "fallbacks": 1,
            "estimated_cost_cny": 0.000016,
        },
    }
    report = {
        "mode": "zero-cost-simulation",
        "network_calls": 0,
        "microphone_opened": False,
        "speaker_opened": False,
        "audio_saved": False,
        "iterations": args.iterations,
        "questions": list(questions),
        "comparison": profiles,
        "pricing_checked_on": config.realtime.pricing_checked_on,
        "model_from_config": config.realtime.model_id,
        "notice": "模拟虚拟时钟数据只验证比较字段、队列和降级路径，不代表真实设备或网络体验。",
    }
    _json_output(report)
    return 0


def run_voice_api_smoke(args: argparse.Namespace) -> int:
    """显式授权后执行最小真实 ASR/TTS 权限与协议烟雾测试。"""

    if not args.confirm_charge:
        print("未执行：真实语音测试可能产生费用，请添加 --confirm-charge。", file=sys.stderr)
        return 1
    try:
        config = load_config(args.config)
        provider, _model = config.provider_for("advice")
        api_key = load_api_key(provider.api_key_env)
        workspace_id = load_api_key(config.voice.workspace_id_env)
    except ConfigError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    if not api_key or not workspace_id:
        print("启动失败：缺少百炼 API Key 或业务空间 ID。", file=sys.stderr)
        return 1

    started = time.perf_counter()
    asr_errors: list[str] = []
    transcripts = []
    asr_started = time.perf_counter()
    if not args.tts_only:
        cancel = Event()
        asr = QwenRealtimeASR(config.voice, config.audio, api_key, workspace_id)
        session = asr.start(
            "voice-api-smoke-asr", cancel, transcripts.append,
            lambda error: asr_errors.append(str(error)),
        )
        # 一秒程序生成的 440 Hz PCM，只验证权限/协议，不录制或上传用户麦克风。
        import math as _math
        samples = bytearray()
        for index in range(config.audio.input_sample_rate):
            value = int(1200 * _math.sin(2 * _math.pi * 440 * index / config.audio.input_sample_rate))
            samples.extend(value.to_bytes(2, "little", signed=True))
        session.send_audio(AudioChunk(bytes(samples), AudioFormat(config.audio.input_sample_rate), 0))
        session.commit()
        if not session.wait(config.voice.asr_timeout_seconds + 3):
            session.close()
            asr_errors.append("ASR 烟雾测试等待超时")
    asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000

    tts_text = "ORIENS语音测试。"
    tts_chunks: list[AudioChunk] = []
    tts_error: str | None = None
    tts_started = time.perf_counter()
    tts_first_audio_ms: float | None = None

    def on_tts_audio(chunk: AudioChunk) -> None:
        nonlocal tts_first_audio_ms
        if tts_first_audio_ms is None:
            tts_first_audio_ms = (time.perf_counter() - tts_started) * 1000
        tts_chunks.append(chunk)

    if not args.asr_only:
        try:
            CosyVoiceStreamingTTS(config.voice, api_key, workspace_id).synthesize(
                "voice-api-smoke-tts", (tts_text,), Event(), on_tts_audio
            )
        except VoiceError as exc:
            tts_error = str(exc)
    tts_elapsed_ms = (time.perf_counter() - tts_started) * 1000

    result = {
        "region": "cn-beijing",
        "asr": {
            "model": config.voice.asr_model_id,
            "ok": args.tts_only or not asr_errors,
            "skipped": args.tts_only,
            "audio_seconds": 1.0,
            "final_transcript_received": any(item.final for item in transcripts),
            "elapsed_ms": round(asr_elapsed_ms, 3),
            "errors": asr_errors,
        },
        "tts": {
            "model": config.voice.tts_model_id,
            "voice": config.voice.tts_voice,
            "ok": args.asr_only or (tts_error is None and bool(tts_chunks)),
            "skipped": args.asr_only,
            "text_chars": len(tts_text),
            "audio_chunks": len(tts_chunks),
            "audio_bytes": sum(len(item.data) for item in tts_chunks),
            "first_audio_ms": round(tts_first_audio_ms, 3) if tts_first_audio_ms is not None else None,
            "elapsed_ms": round(tts_elapsed_ms, 3),
            "error": tts_error,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "audio_saved": False,
    }
    _json_output(result)
    return 0 if result["asr"]["ok"] and result["tts"]["ok"] else 2


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _process_cpu_seconds(process) -> float:
    if process is None:
        return 0.0
    values = [process]
    try:
        values.extend(process.children(recursive=True))
    except Exception:
        pass
    total = 0.0
    for item in values:
        try:
            times = item.cpu_times()
            total += times.user + times.system
        except Exception:
            pass
    return total


def _process_rss(process) -> int:
    if process is None:
        return 0
    values = [process]
    try:
        values.extend(process.children(recursive=True))
    except Exception:
        pass
    total = 0
    for item in values:
        try:
            total += item.memory_info().rss
        except Exception:
            pass
    return total


def run_rag_build(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        import_report: dict[str, object] | None = None
        if config.rag.pipeline_version == 2:
            if not args.skip_import:
                import_report = build_full_corpus(
                    _rag_v2_paths(config),
                    content_version=config.rag.content_version,
                    game_version=config.rag.game_version,
                    progress=lambda message: print(message, flush=True),
                ).as_dict()
            elif not config.rag.chunks_path.is_file():
                raise RagError("rag-v2 分块不存在，不能跳过导入")
            keyword_report = build_keyword_index(
                iter_chunks(config.rag.chunks_path),
                config.rag.index_path,
                corpus_metadata={
                    "content_version": config.rag.content_version,
                    "corpus_id": (
                        "oriens-rag-v2.1-huiji-data-complete"
                        if "v2.1" in config.rag.content_version
                        else "oriens-rag-v2-huiji-complete"
                    ),
                },
            )
            chunks: object = config.rag.chunks_path
            chunk_count = int(keyword_report["chunk_count"])
        else:
            chunks = build_corpus(
                config.rag.source_path, config.rag.chunks_path, config.rag.manifest_path
            )
            keyword_report = build_keyword_index(chunks, config.rag.index_path)
            chunk_count = len(chunks)
        report: dict[str, object] = {
            "documents": (
                import_report["document_count"]
                if import_report is not None
                else chunk_count
            ),
            "chunks": chunk_count,
            "import": import_report,
            "keyword_index": str(config.rag.index_path),
            "keyword_index_report": keyword_report,
            "vector_index": "未请求",
        }
        if args.with_vectors:
            worker = _make_vector_client(config)
            if worker is None:
                raise RagError("配置已关闭向量索引")
            try:
                if config.rag.pipeline_version == 2:
                    vector_report = worker.build_path(
                        config.rag.chunks_path,
                        progress=lambda value: print(
                            f"向量已编码 {value['processed']} 个分块，"
                            f"耗时 {value['elapsed_ms'] / 1000:.1f} 秒",
                            flush=True,
                        ),
                    )
                else:
                    vector_report = worker.build(chunks)
                report["vector_index"] = {
                    **vector_report,
                    "model_init_ms": worker.init_latency_ms,
                }
            finally:
                worker.close()
    except (ConfigError, CorpusValidationError, RagError, RuntimeError, TimeoutError) as exc:
        print(f"RAG 构建失败：{exc}", file=sys.stderr)
        return 1
    _json_output(report)
    return 0


def run_rag_import(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        if config.rag.pipeline_version != 2:
            raise RagError("当前配置不是 rag-v2")
        report = build_full_corpus(
            _rag_v2_paths(config),
            content_version=config.rag.content_version,
            game_version=config.rag.game_version,
            progress=lambda message: print(message, flush=True),
        )
    except (ConfigError, CorpusValidationError, RagError) as exc:
        print(f"RAG 导入失败：{exc}", file=sys.stderr)
        return 1
    _json_output(report.as_dict())
    return 0


def _rag_v2_paths(config: OriensConfig) -> RagV2Paths:
    if not config.rag.raw_paths:
        raise RagError("rag-v2 配置缺少 raw_paths")
    return RagV2Paths(
        raw_paths=config.rag.raw_paths,
        chunks_path=config.rag.chunks_path,
        manifest_path=config.rag.manifest_path,
        entities_path=config.rag.entities_path,
        redirects_path=config.rag.redirects_path,
        dependency_audit_path=config.rag.dependency_audit_path,
        lua_facts_path=config.rag.lua_facts_path,
        overrides_path=config.rag.overrides_path,
    )


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


def run_rag_export_faiss(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        output = (args.output or config.root / "data/indexes/rag-v2.faiss").resolve()
        report = convert_sqlite_vectors_to_faiss(
            config.rag.index_path,
            output,
            dimension=config.rag.vector_dimension,
        )
        _json_output({"output": str(output), **report})
        return 0
    except (ConfigError, ImportError, OSError, RuntimeError) as exc:
        print(f"FAISS 索引转换失败：{exc}", file=sys.stderr)
        return 1


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

    def add_desktop_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--log", type=Path, default=default_log_path())
        command.add_argument("--config", type=Path)
        command.add_argument("--from-start", action="store_true")
        command.add_argument(
            "--online",
            action="store_true",
            help="显式启用百炼；未指定或缺少密钥时使用零费用模拟模型",
        )
        command.set_defaults(handler=run_desktop_command)

    desktop = subparsers.add_parser("desktop", help="启动 Oriens 桌面控制中心与游戏悬浮窗")
    add_desktop_arguments(desktop)
    ui = subparsers.add_parser("ui", help="兼容入口：启动 Oriens 桌面产品外壳")
    add_desktop_arguments(ui)

    voice_benchmark = subparsers.add_parser(
        "voice-benchmark", help="运行零费用模拟语音闭环与打断基准"
    )
    voice_benchmark.add_argument("--config", type=Path)
    voice_benchmark.add_argument("--iterations", type=int, default=20)
    voice_benchmark.add_argument("--stress-cycles", type=int, default=100)
    voice_benchmark.add_argument("--timeout", type=float, default=10.0)
    voice_benchmark.set_defaults(handler=run_voice_benchmark)

    realtime_benchmark = subparsers.add_parser(
        "realtime-benchmark", help="零费用比较链式语音与 Realtime 模拟指标"
    )
    realtime_benchmark.add_argument("--config", type=Path)
    realtime_benchmark.add_argument("--iterations", type=int, default=20)
    realtime_benchmark.set_defaults(handler=run_realtime_benchmark)

    voice_smoke = subparsers.add_parser(
        "voice-api-smoke", help="执行一次最小真实百炼 ASR/TTS 烟雾测试"
    )
    voice_smoke.add_argument("--config", type=Path)
    voice_smoke.add_argument("--confirm-charge", action="store_true")
    voice_modes = voice_smoke.add_mutually_exclusive_group()
    voice_modes.add_argument("--asr-only", action="store_true")
    voice_modes.add_argument("--tts-only", action="store_true")
    voice_smoke.set_defaults(handler=run_voice_api_smoke)

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
    rag_build.add_argument(
        "--with-vectors", action="store_true", help="使用 BGE-M3 构建配置指定的向量索引"
    )
    rag_build.add_argument("--skip-import", action="store_true", help="rag-v2 已有分块时只重建索引")
    rag_build.set_defaults(handler=run_rag_build)

    rag_import = subparsers.add_parser("rag-import", help="流式导入完整灰机快照为 rag-v2")
    rag_import.add_argument("--config", type=Path)
    rag_import.set_defaults(handler=run_rag_import)

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

    rag_export_faiss = subparsers.add_parser(
        "rag-export-faiss", help="复用 sqlite-vec 中的向量构建 FAISS 对比索引"
    )
    rag_export_faiss.add_argument("--config", type=Path)
    rag_export_faiss.add_argument("--output", type=Path)
    rag_export_faiss.set_defaults(handler=run_rag_export_faiss)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
