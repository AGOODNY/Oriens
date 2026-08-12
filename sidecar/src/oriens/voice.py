"""阶段 3 实时 ASR、流式 TTS 与可取消会话抽象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import base64
import json
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Event, Lock, Thread
import time
from typing import Callable, Iterable, Protocol
from urllib.parse import quote
from uuid import uuid4

from .audio import AudioChunk, AudioError, AudioFormat
from .config import AudioSettings, VoiceSettings
from .websocket import StandardWebSocket, WebSocketClosed, WebSocketError


class VoiceError(RuntimeError):
    """经过清理、可安全显示的语音错误。"""


class VoiceCancelled(VoiceError):
    pass


class VoiceTimeout(VoiceError):
    pass


class VoiceInputRejected(VoiceError):
    pass


class VoiceState(str, Enum):
    IDLE = "未监听"
    LISTENING = "正在聆听"
    RECOGNIZING = "正在识别"
    THINKING = "正在思考"
    SPEAKING = "正在播报"
    CANCELLED = "已取消"
    OFFLINE = "离线不可用"


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    final: bool
    language: str = "zh"


@dataclass(slots=True)
class VoiceMetrics:
    capture_start_ms: float = 0.0
    asr_first_partial_ms: float = 0.0
    asr_final_ms: float = 0.0
    rag_ms: float = 0.0
    model_first_text_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    first_audio_end_to_end_ms: float = 0.0
    interrupt_ms: float = 0.0
    queue_peak: int = 0


class RealtimeASRSession(Protocol):
    def send_audio(self, chunk: AudioChunk) -> None: ...
    def commit(self) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class RealtimeASR(Protocol):
    def start(
        self,
        request_id: str,
        cancel: Event,
        on_transcript: Callable[[Transcript], None],
        on_error: Callable[[VoiceError], None],
    ) -> RealtimeASRSession: ...


class StreamingTTS(Protocol):
    def synthesize(
        self,
        request_id: str,
        text_segments: Iterable[str],
        cancel: Event,
        on_audio: Callable[[AudioChunk], None],
    ) -> None: ...


class TerminologyCorrector:
    def __init__(self, replacements: dict[str, str]) -> None:
        self._replacements = tuple(
            sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
        )

    def correct(self, text: str) -> str:
        result = text.strip()
        for source, target in self._replacements:
            if source.isascii():
                result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)
            else:
                result = result.replace(source, target)
        return result

    @classmethod
    def from_entities(cls, path: Path, *, limit: int = 320) -> "TerminologyCorrector":
        allowed_types = {"item", "character", "room", "route", "trinket", "card"}
        replacements: dict[str, str] = {
            "妈刀": "妈妈的菜刀",
            "硫磺火": "硫磺火",
            "r key": "R键",
            "boss rush": "Boss Rush",
            "恶魔房": "恶魔房",
            "天使房": "天使房",
        }
        try:
            source = path.open("r", encoding="utf-8")
        except OSError:
            return cls(replacements)
        with source:
            for line in source:
                if len(replacements) >= limit:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("entity_type") not in allowed_types:
                    continue
                target = value.get("name_zh") or value.get("name_en")
                aliases = value.get("aliases", [])
                if not isinstance(target, str) or not isinstance(aliases, list):
                    continue
                for alias in aliases:
                    if (
                        isinstance(alias, str)
                        and 2 <= len(alias) <= 40
                        and alias.casefold() != target.casefold()
                    ):
                        replacements.setdefault(alias, target)
                        if len(replacements) >= limit:
                            break
        return cls(replacements)


def validate_recording(
    chunks: Iterable[AudioChunk], settings: AudioSettings
) -> tuple[AudioChunk, ...]:
    values = tuple(chunks)
    if not values or not any(chunk.data for chunk in values):
        raise VoiceInputRejected("没有检测到音频，请按住说话后重试。")
    duration = sum(chunk.duration_ms for chunk in values)
    if duration < settings.min_recording_ms:
        raise VoiceInputRejected("说话时间太短，请按住按键说完整一句。")
    if duration > settings.max_recording_seconds * 1000:
        raise VoiceInputRejected("录音超过时长上限，请缩短问题。")
    peak = max(chunk.rms for chunk in values)
    if peak < settings.silence_rms_threshold:
        raise VoiceInputRejected("只检测到静音，请检查麦克风。")
    active = sum(chunk.rms >= settings.noise_rms_threshold for chunk in values)
    if active == 0:
        raise VoiceInputRejected("没有检测到清晰语音，请靠近麦克风重试。")
    return values


def segment_tts_text(text: str, max_chars: int) -> tuple[str, ...]:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"\b(?:chunk|entity|request|state)[-_ ]?id\b[^，。；]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ()
    segments: list[str] = []
    for sentence in re.split(r"(?<=[。！？；])", cleaned):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = max(sentence.rfind("，", 0, max_chars), sentence.rfind("、", 0, max_chars))
            if cut < max_chars // 2:
                cut = max_chars
            segments.append(sentence[: cut + (cut < max_chars)].strip())
            sentence = sentence[cut + (cut < max_chars) :].strip()
        if sentence:
            segments.append(sentence)
    return tuple(segments)


class MockASRSession:
    def __init__(
        self,
        cancel: Event,
        on_transcript: Callable[[Transcript], None],
        partials: tuple[str, ...],
        final: str,
    ) -> None:
        self._cancel = cancel
        self._on_transcript = on_transcript
        self._partials = partials
        self._final = final
        self._chunks: list[AudioChunk] = []
        self.closed = False

    def send_audio(self, chunk: AudioChunk) -> None:
        if self._cancel.is_set():
            raise VoiceCancelled("语音识别已取消")
        self._chunks.append(chunk)
        index = min(len(self._chunks) - 1, len(self._partials) - 1)
        if self._partials:
            self._on_transcript(Transcript(self._partials[index], False))

    def commit(self) -> None:
        if self._cancel.is_set():
            raise VoiceCancelled("语音识别已取消")
        self._on_transcript(Transcript(self._final, True))

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.closed = True


class MockRealtimeASR:
    def __init__(self, final: str, partials: tuple[str, ...] = ()) -> None:
        self.final = final
        self.partials = partials
        self.sessions: list[MockASRSession] = []

    def start(self, request_id, cancel, on_transcript, on_error) -> MockASRSession:
        session = MockASRSession(cancel, on_transcript, self.partials, self.final)
        self.sessions.append(session)
        return session


class MockStreamingTTS:
    def __init__(self, audio_format: AudioFormat) -> None:
        self.format = audio_format
        self.requests: list[tuple[str, tuple[str, ...]]] = []

    def synthesize(self, request_id, text_segments, cancel, on_audio) -> None:
        segments = tuple(text_segments)
        self.requests.append((request_id, segments))
        for index, segment in enumerate(segments):
            if cancel.is_set():
                raise VoiceCancelled("语音合成已取消")
            # 20 ms 项目自有程序生成静音块；只验证流式与队列，不冒充真人音色。
            on_audio(AudioChunk(b"\x00\x00" * 480, self.format, index))


class _Transport(Protocol):
    def connect(self) -> None: ...
    def send_json(self, value: dict) -> None: ...
    def receive(self, cancel: Event) -> str | bytes | None: ...
    def close(self) -> None: ...


TransportFactory = Callable[[str, dict[str, str], float], _Transport]


class QwenRealtimeASR:
    def __init__(
        self,
        settings: VoiceSettings,
        audio: AudioSettings,
        api_key: str,
        workspace_id: str,
        *,
        transport_factory: TransportFactory = StandardWebSocket,
    ) -> None:
        self._settings = settings
        self._audio = audio
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._factory = transport_factory

    def start(self, request_id, cancel, on_transcript, on_error) -> "_QwenASRSession":
        session = _QwenASRSession(
            self._settings,
            self._audio,
            self._api_key,
            self._workspace_id,
            cancel,
            on_transcript,
            on_error,
            self._factory,
        )
        session.start()
        return session


class _QwenASRSession:
    _COMMIT = object()

    def __init__(self, settings, audio, api_key, workspace_id, cancel, on_transcript, on_error, factory):
        self._settings = settings
        self._audio = audio
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._cancel = cancel
        self._on_transcript = on_transcript
        self._on_error = on_error
        self._factory = factory
        self._outbound: Queue[AudioChunk | object] = Queue()
        self._buffer: list[AudioChunk] = []
        self._transport: _Transport | None = None
        self._transport_lock = Lock()
        self._thread = Thread(target=self._run, name="oriens-qwen-asr", daemon=True)
        self._committed = Event()
        self._committed_at: float | None = None

    def start(self) -> None:
        self._thread.start()

    def send_audio(self, chunk: AudioChunk) -> None:
        if self._cancel.is_set():
            raise VoiceCancelled("语音识别已取消")
        self._buffer.append(chunk)
        self._outbound.put(chunk)

    def commit(self) -> None:
        self._committed.set()
        self._committed_at = time.monotonic()
        self._outbound.put(self._COMMIT)

    def cancel(self) -> None:
        self._cancel.set()
        with self._transport_lock:
            transport = self._transport
        if transport is not None:
            transport.close()

    def close(self) -> None:
        self.cancel()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        retries = self._settings.asr_max_retries
        for attempt in range(retries + 1):
            if self._cancel.is_set():
                return
            try:
                self._run_once(replay=attempt > 0)
                return
            except VoiceCancelled:
                return
            except (WebSocketError, VoiceError):
                if attempt >= retries:
                    self._on_error(VoiceError("实时语音识别连接失败，文本功能仍可使用。"))
                    return
                if self._cancel.wait(min(0.15 * (attempt + 1), 0.5)):
                    return

    def _run_once(self, *, replay: bool) -> None:
        endpoint = self._settings.asr_endpoint.format(
            workspace_id=quote(self._workspace_id, safe="")
        )
        separator = "&" if "?" in endpoint else "?"
        endpoint += separator + "model=" + quote(self._settings.asr_model_id, safe="")
        transport = self._factory(
            endpoint,
            {"Authorization": "Bearer " + self._api_key, "User-Agent": "Oriens/0.2"},
            self._settings.asr_timeout_seconds,
        )
        with self._transport_lock:
            self._transport = transport
        transport.connect()
        transport.send_json({
            "event_id": "event_" + uuid4().hex,
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": self._audio.input_sample_rate,
                "input_audio_transcription": {"language": self._settings.asr_language},
                "turn_detection": None,
            },
        })
        if replay:
            self._discard_outbound()
            for chunk in self._buffer:
                self._send_chunk(transport, chunk)
            if self._committed.is_set():
                self._send_commit(transport)
        finished = False
        try:
            while not self._cancel.is_set() and not finished:
                if (
                    self._committed_at is not None
                    and time.monotonic() - self._committed_at
                    > self._settings.asr_timeout_seconds
                ):
                    raise VoiceTimeout("实时语音识别超时")
                while True:
                    try:
                        item = self._outbound.get_nowait()
                    except Empty:
                        break
                    if item is self._COMMIT:
                        self._send_commit(transport)
                    else:
                        assert isinstance(item, AudioChunk)
                        self._send_chunk(transport, item)
                incoming = transport.receive(self._cancel)
                if incoming is None:
                    continue
                if isinstance(incoming, bytes):
                    continue
                try:
                    event = json.loads(incoming)
                except json.JSONDecodeError:
                    raise VoiceError("语音识别返回格式无效") from None
                kind = event.get("type")
                if kind == "conversation.item.input_audio_transcription.delta":
                    text = str(event.get("text", "")) + str(event.get("stash", ""))
                    if text:
                        self._on_transcript(Transcript(text, False, str(event.get("language", "zh"))))
                elif kind == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        self._on_transcript(Transcript(transcript.strip(), True, str(event.get("language", "zh"))))
                elif kind == "conversation.item.input_audio_transcription.failed" or kind == "error":
                    raise VoiceError("实时语音识别失败")
                elif kind == "session.finished":
                    finished = True
        except WebSocketClosed:
            if not self._cancel.is_set():
                raise
        finally:
            with self._transport_lock:
                if self._transport is transport:
                    self._transport = None
            transport.close()

    @staticmethod
    def _send_chunk(transport: _Transport, chunk: AudioChunk) -> None:
        transport.send_json({
            "event_id": "event_" + uuid4().hex,
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk.data).decode("ascii"),
        })

    @staticmethod
    def _send_commit(transport: _Transport) -> None:
        transport.send_json({"event_id": "event_" + uuid4().hex, "type": "input_audio_buffer.commit"})
        transport.send_json({"event_id": "event_" + uuid4().hex, "type": "session.finish"})

    def _discard_outbound(self) -> None:
        while True:
            try:
                self._outbound.get_nowait()
            except Empty:
                return


class CosyVoiceStreamingTTS:
    def __init__(self, settings: VoiceSettings, api_key: str, workspace_id: str, *, transport_factory: TransportFactory = StandardWebSocket) -> None:
        self._settings = settings
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._factory = transport_factory

    def synthesize(self, request_id, text_segments, cancel, on_audio) -> None:
        segments = tuple(text_segments)
        last_error: Exception | None = None
        for attempt in range(self._settings.tts_max_retries + 1):
            try:
                self._synthesize_once(segments, cancel, on_audio)
                return
            except VoiceCancelled:
                raise
            except (WebSocketError, VoiceError) as exc:
                last_error = exc
                if attempt < self._settings.tts_max_retries:
                    if cancel.wait(min(0.15 * (attempt + 1), 0.5)):
                        raise VoiceCancelled("语音合成已取消") from None
        raise VoiceError("语音合成连接失败，文字回答仍可查看。") from last_error

    def _synthesize_once(self, segments, cancel, on_audio) -> None:
        endpoint = self._settings.tts_endpoint.format(workspace_id=quote(self._workspace_id, safe=""))
        transport = self._factory(
            endpoint,
            {"Authorization": "Bearer " + self._api_key, "User-Agent": "Oriens/0.2"},
            self._settings.tts_timeout_seconds,
        )
        task_id = str(uuid4())
        audio_format = AudioFormat(self._settings.tts_sample_rate)
        sequence = 0
        watcher_done = Event()

        def watch_cancel() -> None:
            while not watcher_done.wait(0.05):
                if cancel.is_set():
                    transport.close()
                    return

        watcher = Thread(target=watch_cancel, name="oriens-tts-cancel", daemon=True)
        watcher.start()
        try:
            transport.connect()
            transport.send_json({
                "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {
                    "task_group": "audio", "task": "tts", "function": "SpeechSynthesizer",
                    "model": self._settings.tts_model_id,
                    "parameters": {
                        "text_type": "PlainText", "voice": self._settings.tts_voice,
                        "format": self._settings.tts_format, "sample_rate": self._settings.tts_sample_rate,
                        "volume": self._settings.tts_volume, "rate": self._settings.tts_rate,
                        "pitch": 1.0, "enable_ssml": False, "language_hints": ["zh"],
                    },
                    "input": {},
                },
            })
            started = False
            start_deadline = time.monotonic() + self._settings.tts_timeout_seconds
            while not started:
                if time.monotonic() > start_deadline:
                    raise VoiceTimeout("语音合成任务启动超时")
                event = self._receive_json(transport, cancel)
                if event is None:
                    continue
                kind = event.get("header", {}).get("event")
                if kind == "task-started":
                    started = True
                elif kind == "task-failed":
                    raise VoiceError("语音合成任务启动失败")
            for segment in segments:
                if cancel.is_set():
                    raise VoiceCancelled("语音合成已取消")
                transport.send_json({
                    "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {"text": segment}},
                })
            transport.send_json({
                "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"input": {}},
            })
            finished = False
            deadline = time.monotonic() + self._settings.tts_timeout_seconds
            while not finished and not cancel.is_set():
                if time.monotonic() > deadline:
                    raise VoiceTimeout("语音合成超时")
                incoming = transport.receive(cancel)
                if incoming is None:
                    continue
                if isinstance(incoming, bytes):
                    on_audio(AudioChunk(incoming, audio_format, sequence))
                    sequence += 1
                    continue
                try:
                    event = json.loads(incoming)
                except json.JSONDecodeError:
                    raise VoiceError("语音合成返回格式无效") from None
                kind = event.get("header", {}).get("event")
                if kind == "task-finished":
                    finished = True
                elif kind == "task-failed":
                    raise VoiceError("语音合成任务失败")
            if cancel.is_set():
                raise VoiceCancelled("语音合成已取消")
        finally:
            watcher_done.set()
            transport.close()
            watcher.join(timeout=0.2)

    @staticmethod
    def _receive_json(transport: _Transport, cancel: Event) -> dict | None:
        incoming = transport.receive(cancel)
        if incoming is None:
            return None
        if not isinstance(incoming, str):
            raise VoiceError("语音合成事件顺序无效")
        try:
            value = json.loads(incoming)
        except json.JSONDecodeError:
            raise VoiceError("语音合成返回格式无效") from None
        if not isinstance(value, dict):
            raise VoiceError("语音合成返回格式无效")
        return value


@dataclass(slots=True)
class VoiceSession:
    request_id: str
    cancel_event: Event = field(default_factory=Event)
    state: VoiceState = VoiceState.IDLE
    chunks: list[AudioChunk] = field(default_factory=list)
    final_transcript: str = ""
    metrics: VoiceMetrics = field(default_factory=VoiceMetrics)
    created_at: float = field(default_factory=time.perf_counter)
    asr: RealtimeASRSession | None = None


class SessionRegistry:
    """原子替换当前交互，并将取消向所有已注册下游传播。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._current: VoiceSession | None = None
        self._cancellers: list[Callable[[], None]] = []

    @property
    def current(self) -> VoiceSession | None:
        with self._lock:
            return self._current

    def replace(self) -> VoiceSession:
        self.cancel_current()
        session = VoiceSession(uuid4().hex)
        with self._lock:
            self._current = session
            self._cancellers = []
        return session

    def register_cancel(self, request_id: str, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._current is not None and self._current.request_id == request_id:
                self._cancellers.append(callback)

    def is_current(self, request_id: str) -> bool:
        with self._lock:
            return self._current is not None and self._current.request_id == request_id and not self._current.cancel_event.is_set()

    def cancel_current(self) -> None:
        with self._lock:
            session = self._current
            cancellers = tuple(self._cancellers)
            self._cancellers = []
        if session is None:
            return
        started = time.perf_counter()
        session.cancel_event.set()
        session.state = VoiceState.CANCELLED
        for callback in cancellers:
            try:
                callback()
            except Exception:
                pass
        session.metrics.interrupt_ms = (time.perf_counter() - started) * 1000

    def finish(self, request_id: str) -> None:
        with self._lock:
            if self._current is not None and self._current.request_id == request_id:
                self._cancellers = []
