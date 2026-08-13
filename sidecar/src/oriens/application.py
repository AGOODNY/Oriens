"""Oriens 应用启动、依赖装配与生命周期边界。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .advice import AdviceEngine
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
from .memory import MemoryStore, NullMemoryStore
from .modeling import ModelRouter
from .paths import AppPaths, RuntimeMode
from .protocol import GameEvent
from .query import QueryEngine
from .rag import RagError, RagService
from .rag_pipeline import build_corpus, build_keyword_index
from .rag_worker import VectorWorkerClient
from .state import StateStore
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
                self._memory.end_session(self._active_memory_session)
            self._memory.begin_session(current)
            self._active_memory_session = current
        if event.type == "run_ended" and self._active_memory_session is not None:
            self._memory.end_session(self._active_memory_session)
            self._active_memory_session = None

    def close(self) -> None:
        if self._active_memory_session is not None:
            self._memory.end_session(self._active_memory_session)
            self._active_memory_session = None


class MemoryAwareQueryEngine:
    """只标记记忆接入点；空实现不会改变提示词或保存任何内容。"""

    def __init__(self, query: QueryEngine, memory: MemoryStore) -> None:
        self._query = query
        self._memory = memory

    def ask(self, question, state, request_id, cancel=None):
        self._memory.recall(question)
        result = self._query.ask(question, state, request_id, cancel)
        self._memory.submit_candidates(())
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
            memory_store = memory or NullMemoryStore()
            query = MemoryAwareQueryEngine(raw_query, memory_store)
            session = GameSession(memory_store)
            log_path = (options.log_path or default_log_path()).resolve()
            tailer = LogTailer(log_path, from_start=options.from_start)
            return cls(
                paths=paths, config=config, options=options, knowledge_pack=selected,
                knowledge=knowledge, budget=budget, router=router, rag=rag,
                advice_engine=advice, query_engine=query, session=session,
                tailer=tailer, memory=memory_store, api_key=api_key,
                workspace_id=workspace_id,
            )
        except Exception:
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
        if self.options.online and self._api_key and self._workspace_id:
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
        return VoiceAssembly(service, asr is not None, unavailable)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for service in self._voice_services:
            service.close()
        self._voice_services.clear()
        self.tailer.close()
        self.session.close()
        self.rag.close()
        self.memory.close()


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
