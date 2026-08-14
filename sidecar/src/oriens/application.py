"""Oriens 应用启动、依赖装配与生命周期边界。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Event, Lock
import time
from typing import Callable

from .advice import AdviceEngine, AdviceResponse, StateToken
from .audio import (
    AudioDeviceUnavailable,
    AudioFormat,
    NullAudioPlayer,
    QtAudioPlayer,
    QtMicrophoneInput,
    UnavailableMicrophone,
)
from .budget import BudgetTracker
from .config import ConfigService, OriensConfig
from .credentials import CredentialService
from .knowledge import LocalItemKnowledgeBase
from .knowledge_pack import InstalledKnowledgePack, KnowledgePackError, KnowledgePackManager
from .memory import (
    MemoryCandidate,
    MemoryContext,
    MemoryItem,
    MemoryStore,
    NullMemoryStore,
    SQLiteMemoryStore,
    extract_explicit_candidates,
)
from .modeling import ModelRouter
from .paths import AppPaths, RuntimeMode
from .protocol import EventParseError, GameEvent, parse_event_line
from .query import QueryEngine
from .rag import RagError, RagService
from .rag_pipeline import build_corpus, build_keyword_index
from .rag_worker import VectorWorkerClient
from .state import EventOrderError, StateStore
from .tailer import LogTailer
from .voice import CosyVoiceStreamingTTS, QwenRealtimeASR, TerminologyCorrector
from .voice_service import VoiceCallbacks, VoiceService


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    config_path: Path | None = None
    log_path: Path | None = None
    from_start: bool = False
    online: bool = False
    enable_vector: bool = True


@dataclass(frozen=True, slots=True)
class VoiceAssembly:
    service: VoiceService
    asr_available: bool
    unavailable_reason: str | None = None


class ApplicationPhase(str, Enum):
    STARTING = "应用启动中"
    READY = "已就绪"
    EXITING = "正在退出"


class ListeningState(str, Enum):
    LISTENING = "游戏日志监听中"
    PAUSED = "监听已暂停"
    STOPPED = "监听已停止"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    phase: ApplicationPhase
    listening: ListeningState
    online: bool
    online_available: bool
    config_source: str
    knowledge_name: str
    knowledge_version: str
    knowledge_capability: str
    log_connection: str
    rag_status: str
    voice_status: str
    memory_status: str
    run_cost_cny: float
    run_budget_cny: float


class GameSession:
    """短期游戏状态与阶段 4 会话生命周期的稳定归属。"""

    def __init__(self, memory: MemoryStore) -> None:
        self._store = StateStore()
        self._memory = memory
        self._active_memory_session: str | None = None

    @property
    def state(self):
        return self._store.state

    @property
    def diagnostics(self):
        return self._store.diagnostics

    def snapshot(self):
        return self._store.snapshot()

    def mark_invalid(self) -> None:
        self._store.mark_invalid()

    def mark_ignored(self) -> None:
        self._store.mark_ignored()

    def apply(self, event: GameEvent) -> None:
        previous = self._store.state.run_id
        self._store.apply(event)
        current = self._store.state.run_id
        if current and current != previous:
            if self._active_memory_session is not None:
                try:
                    self._memory.end_session(self._active_memory_session)
                except Exception:
                    pass
            try:
                self._memory.begin_session(current)
            except Exception:
                self._active_memory_session = None
            else:
                self._active_memory_session = current
        if event.type == "run_ended" and self._active_memory_session is not None:
            try:
                self._memory.end_session(self._active_memory_session)
            except Exception:
                pass
            self._active_memory_session = None

    def close(self) -> None:
        if self._active_memory_session is not None:
            try:
                self._memory.end_session(self._active_memory_session)
            except Exception:
                pass
            self._active_memory_session = None


class MemoryAwareQueryEngine:
    """在稳定边界内执行有预算召回和确定性候选提交。"""

    def __init__(
        self,
        query: QueryEngine,
        memory: MemoryStore,
        *,
        recall_max_items: int = 3,
        recall_max_chars: int = 360,
    ) -> None:
        self._query = query
        self._memory = memory
        self._recall_max_items = recall_max_items
        self._recall_max_chars = recall_max_chars

    def ask(self, question, state, request_id, cancel=None):
        try:
            context = self._memory.recall(
                question,
                max_items=self._recall_max_items,
                max_chars=self._recall_max_chars,
            )
        except Exception:
            # 记忆永远不是文字问答的可用性前提。
            context = None
        result = self._query.ask(
            question, state, request_id, cancel, memory_context=context
        )
        try:
            candidates = extract_explicit_candidates(
                question,
                session_id=state.run_id,
                run_id=state.run_id,
            )
            self._memory.submit_candidates(candidates)
        except Exception:
            # 候选过滤或写入失败不能使已经完成的回答失败。
            pass
        return result


class OriensApplication:
    """桌面控制中心、悬浮窗和未来托盘将共享的运行时。"""

    def __init__(
        self,
        *,
        paths: AppPaths,
        config: OriensConfig,
        options: LaunchOptions,
        knowledge_pack: InstalledKnowledgePack | None,
        knowledge: LocalItemKnowledgeBase,
        budget: BudgetTracker,
        router: ModelRouter,
        rag: RagService,
        advice_engine: AdviceEngine,
        query_engine: MemoryAwareQueryEngine,
        session: GameSession,
        tailer: LogTailer,
        memory: MemoryStore,
        api_key: str | None,
        workspace_id: str | None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.options = options
        self.knowledge_pack = knowledge_pack
        self.knowledge = knowledge
        self.budget = budget
        self.router = router
        self.rag = rag
        self.advice_engine = advice_engine
        self.query_engine = query_engine
        self.session = session
        self.tailer = tailer
        self.memory = memory
        self.api_key_available = bool(api_key)
        self.workspace_id_available = bool(workspace_id)
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._voice_services: list[VoiceService] = []
        self._voice_status = "尚未初始化；文字功能可用"
        if isinstance(memory, NullMemoryStore):
            self._memory_status = "长期记忆已关闭"
        else:
            self._memory_status = "长期记忆已启用，仅保存在本机"
        self._phase = ApplicationPhase.READY
        self._listening = ListeningState.LISTENING
        self._last_event_at: float | None = None
        self._log_error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oriens-model")
        self._close_lock = Lock()
        self._closed = False

    @classmethod
    def build(
        cls,
        paths: AppPaths,
        options: LaunchOptions,
        *,
        memory: MemoryStore | None = None,
    ) -> "OriensApplication":
        config = ConfigService(paths).load(options.config_path)
        manager = KnowledgePackManager(paths.knowledge_dir)
        selected: InstalledKnowledgePack | None = None
        if options.config_path is None:
            selected = manager.current()
        if selected is not None:
            config = _config_for_pack(config, selected, paths)
        elif paths.mode is RuntimeMode.INSTALLED:
            raise KnowledgePackError("尚未选择可用的知识包。")
        _ensure_rag_index(config)
        vector = _make_vector_client(config) if options.enable_vector else None
        try:
            if vector is not None:
                vector.start()
            rag = RagService(
                config.rag.index_path,
                vector,
                vector_min_similarity=config.rag.vector_min_similarity,
            )
        except Exception:
            if vector is not None:
                vector.close()
            raise
        memory_store: MemoryStore | None = None
        tailer: LogTailer | None = None
        try:
            knowledge = LocalItemKnowledgeBase.load(config.app.knowledge_path)
            budget = BudgetTracker(config.budget.run_limit_cny)
            credentials = CredentialService(paths)
            provider, _model = config.provider_for("advice")
            api_key = credentials.read(provider.api_key_env)
            workspace_id = credentials.read(config.voice.workspace_id_env)
            router = ModelRouter(config, online=options.online, api_key=api_key)
            advice = AdviceEngine(
                knowledge, router, budget, rag=rag, game_version=config.rag.game_version
            )
            raw_query = QueryEngine(rag, router, budget, game_version=config.rag.game_version)
            memory_store, memory_status = _make_memory_store(paths, config, memory)
            query = MemoryAwareQueryEngine(
                raw_query,
                memory_store,
                recall_max_items=config.memory.recall_max_items,
                recall_max_chars=config.memory.recall_max_chars,
            )
            session = GameSession(memory_store)
            log_path = (options.log_path or default_log_path()).resolve()
            tailer = LogTailer(log_path, from_start=options.from_start)
            application = cls(
                paths=paths, config=config, options=options, knowledge_pack=selected,
                knowledge=knowledge, budget=budget, router=router, rag=rag,
                advice_engine=advice, query_engine=query, session=session,
                tailer=tailer, memory=memory_store, api_key=api_key,
                workspace_id=workspace_id,
            )
            application._memory_status = memory_status
            return application
        except Exception:
            if tailer is not None:
                tailer.close()
            if memory_store is not None:
                memory_store.close()
            rag.close()
            raise

    def create_voice_service(
        self,
        callbacks: VoiceCallbacks,
    ) -> VoiceAssembly:
        """在 QApplication 已存在后创建 Qt 音频基础设施。"""

        unavailable: str | None = None
        try:
            microphone = QtMicrophoneInput(
                AudioFormat(self.config.audio.input_sample_rate),
                self.config.audio.chunk_duration_ms,
            )
            player = QtAudioPlayer(
                AudioFormat(self.config.audio.playback_sample_rate),
                self.config.audio.playback_queue_max_chunks,
            )
        except AudioDeviceUnavailable as exc:
            microphone = UnavailableMicrophone()
            player = NullAudioPlayer()
            unavailable = str(exc) + " 文字提问仍可使用。"
        asr = None
        tts = None
        if (
            self.config.voice.enabled
            and self.options.online
            and self._api_key
            and self._workspace_id
        ):
            asr = QwenRealtimeASR(
                self.config.voice, self.config.audio, self._api_key, self._workspace_id
            )
            tts = CosyVoiceStreamingTTS(
                self.config.voice, self._api_key, self._workspace_id
            )
        service = VoiceService(
            audio_settings=self.config.audio,
            voice_settings=self.config.voice,
            microphone=microphone,
            player=player,
            asr=asr,
            tts=tts,
            query_engine=self.query_engine,
            terminology=TerminologyCorrector.from_entities(self.config.rag.entities_path),
            state_provider=lambda: self.session.state,
            callbacks=callbacks,
        )
        self._voice_services.append(service)
        if asr is not None:
            self._voice_status = "语音可用"
        elif unavailable:
            self._voice_status = unavailable
        elif not self.config.voice.enabled:
            self._voice_status = "语音已在设置中关闭；文字功能可用"
        elif not self.options.online:
            self._voice_status = "离线模式下语音不可用；文字功能可用"
        elif not self._api_key:
            self._voice_status = "缺少开发凭据，语音不可用；文字功能可用"
        else:
            self._voice_status = "语音工作区不可用；文字功能可用"
        return VoiceAssembly(service, asr is not None, unavailable)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def listening(self) -> bool:
        return self._listening is ListeningState.LISTENING

    def pause_listening(self) -> None:
        if not self._closed:
            self._listening = ListeningState.PAUSED

    def resume_listening(self) -> None:
        if not self._closed:
            self._listening = ListeningState.LISTENING

    def list_memories(self) -> tuple[MemoryItem, ...]:
        return self.memory.list_items()

    def add_memory(self, kind: str, content: str) -> MemoryItem:
        return self.memory.add(MemoryCandidate(
            content=content,
            source="用户在记忆管理中手动添加",
            kind=kind,
            confidence=1.0,
            confirmation_level="manual",
            evidence="用户手动添加",
            source_session_id=self.session.state.run_id,
            source_run_id=self.session.state.run_id,
        ))

    def update_memory(self, memory_id: str, kind: str, content: str) -> MemoryItem:
        return self.memory.update(memory_id, content=content, kind=kind)

    def set_memory_item_enabled(self, memory_id: str, enabled: bool) -> bool:
        return self.memory.set_item_enabled(memory_id, enabled)

    def delete_memory(self, memory_id: str) -> bool:
        return self.memory.delete(memory_id)

    def clear_memories(self) -> int:
        return self.memory.clear_all()

    def submit_advice(
        self, event: GameEvent, cancel: Event
    ) -> Future[tuple[AdviceResponse, StateToken]]:
        if self._closed:
            raise RuntimeError("Oriens 正在退出，无法创建新任务。")
        return self._executor.submit(self._generate_advice, event, cancel)

    def _generate_advice(
        self, event: GameEvent, cancel: Event
    ) -> tuple[AdviceResponse, StateToken]:
        context: MemoryContext | None = None
        try:
            context = self.memory.recall(
                "道具建议 提示频率 解释深度",
                max_items=self.config.memory.recall_max_items,
                max_chars=self.config.memory.recall_max_chars,
            )
        except Exception:
            pass
        return self.advice_engine.generate(event, cancel, memory_context=context)

    def poll_events(self) -> tuple[GameEvent, ...]:
        """推进日志游标；暂停时丢弃新行，恢复后只处理之后的事件。"""

        if self._closed:
            return ()
        try:
            poll = self.tailer.poll()
        except OSError:
            self._log_error = "游戏日志暂时无法读取"
            return ()
        self._log_error = None
        if poll.reopened:
            self.session.diagnostics.log_reopens += 1
        if self._listening is ListeningState.PAUSED:
            return ()
        events: list[GameEvent] = []
        for line in poll.lines:
            event = self.process_log_line(line)
            if event is not None:
                events.append(event)
        return tuple(events)

    def process_log_line(self, line: str) -> GameEvent | None:
        if self._closed or self._listening is not ListeningState.LISTENING:
            return None
        try:
            event = parse_event_line(line)
        except EventParseError:
            self.session.mark_invalid()
            return None
        if event is None:
            self.session.mark_ignored()
            return None
        previous_room = (
            self.session.state.context.get("room_index"),
            self.session.state.context.get("room_spawn_seed"),
        )
        try:
            self.session.apply(event)
        except EventOrderError:
            return None
        self._last_event_at = time.monotonic()
        self.budget.set_run(self.session.state.run_id)
        current_room = (
            self.session.state.context.get("room_index"),
            self.session.state.context.get("room_spawn_seed"),
        )
        if current_room != previous_room and event.type != "collectible_spawned":
            for service in tuple(self._voice_services):
                service.room_changed()
        return event

    def runtime_snapshot(self) -> RuntimeSnapshot:
        pack = self.knowledge_pack
        if pack is not None:
            knowledge_name = pack.manifest.display_name
            capability = "完整能力" if "vector" in pack.capabilities else "轻量能力"
        else:
            knowledge_name = "开发配置知识包"
            capability = "完整能力" if self.config.rag.vector_enabled else "轻量能力"
        if self._log_error:
            log_connection = self._log_error
        elif self._last_event_at is None:
            log_connection = "等待游戏事件"
        elif time.monotonic() - self._last_event_at > 3:
            log_connection = "日志已连接，等待新事件"
        else:
            log_connection = "游戏已连接"
        vector = getattr(self.rag, "_vector", None)
        if vector is None:
            rag_status = "关键词检索可用"
        elif getattr(vector, "available", False):
            rag_status = "本地混合检索可用"
        else:
            rag_status = "向量能力不可用，已使用关键词检索"
        source = "开发显式配置" if self.options.config_path is not None else "用户设置或默认配置"
        return RuntimeSnapshot(
            self._phase,
            self._listening,
            self.options.online and self.api_key_available,
            self.api_key_available,
            source,
            knowledge_name,
            self.config.rag.content_version,
            capability,
            log_connection,
            rag_status,
            self._voice_status,
            self._memory_status,
            self.budget.run_total_cny,
            self.budget.run_limit_cny,
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._phase = ApplicationPhase.EXITING
            self._listening = ListeningState.STOPPED
        closers = [service.close for service in tuple(self._voice_services)]
        self._voice_services.clear()
        closers.extend(
            (
                lambda: self._executor.shutdown(wait=True, cancel_futures=True),
                self.tailer.close,
                self.session.close,
                self.rag.close,
                self.memory.close,
            )
        )
        for close in closers:
            try:
                close()
            except Exception:
                # 完全退出必须继续释放其余资源；普通用户界面不显示底层异常细节。
                continue


def default_log_path() -> Path:
    return Path.home() / "Documents/My Games/Binding of Isaac Repentance+/log.txt"


def _ensure_rag_index(config: OriensConfig) -> None:
    if config.rag.pipeline_version == 2:
        if not config.rag.index_path.is_file():
            raise RagError("本地知识包索引不可用；请安装有效知识包或运行 oriens rag-build。")
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


def _make_memory_store(
    paths: AppPaths,
    config: OriensConfig,
    injected: MemoryStore | None,
) -> tuple[MemoryStore, str]:
    if injected is not None:
        if injected.enabled:
            return injected, "长期记忆已启用，仅保存在本机"
        return injected, "长期记忆已关闭"
    if not config.memory.enabled:
        return NullMemoryStore(), "长期记忆已关闭"
    try:
        return SQLiteMemoryStore(paths.memory_dir), "长期记忆已启用，仅保存在本机"
    except Exception:
        return NullMemoryStore(), "长期记忆暂时不可用，已安全切换为无记忆模式"


def _config_for_pack(
    config: OriensConfig,
    pack: InstalledKnowledgePack,
    paths: AppPaths,
) -> OriensConfig:
    keyword = pack.file_for("keyword_index")
    entities = pack.file_for("entities")
    vector = pack.file_for("vector_index")
    if keyword is None or entities is None:
        raise KnowledgePackError("知识包缺少运行所需的关键词索引或实体表。")
    rag = replace(
        config.rag,
        index_path=keyword,
        source_path=entities,
        chunks_path=entities,
        entities_path=entities,
        manifest_path=pack.root / "manifest.json",
        content_version=pack.manifest.content_version,
        game_version=pack.manifest.game_version,
        vector_enabled=vector is not None,
        vector_backend="faiss" if vector is not None else config.rag.vector_backend,
        vector_index_path=vector or keyword,
        vector_dimension=pack.manifest.vector_dimension,
        vector_model_id=pack.manifest.embedding_model_id,
        vector_model_path=paths.model_dir_for(pack.manifest.embedding_model_id),
    )
    return replace(config, rag=rag)
