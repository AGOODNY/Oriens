"""按键说话的阶段 3 编排器：采集 -> ASR -> 统一问答 -> TTS -> 播放。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import time
from typing import Callable

from .audio import AudioChunk, AudioDeviceUnavailable, AudioError, AudioPlayer, MicrophoneInput
from .config import AudioSettings, VoiceSettings
from .modeling import ModelCancelled
from .query import QueryEngine, QueryError, QueryResponse, QueryToken
from .state import GameState
from .voice import (
    RealtimeASR,
    SessionRegistry,
    StreamingTTS,
    TerminologyCorrector,
    Transcript,
    VoiceCancelled,
    VoiceError,
    VoiceInputRejected,
    VoiceSession,
    VoiceState,
    segment_tts_text,
    validate_recording,
)


class VoiceCallbacks:
    def __init__(
        self,
        *,
        on_state: Callable[[str, VoiceState], None],
        on_transcript: Callable[[str, Transcript], None],
        on_question: Callable[[str, str], None],
        on_answer: Callable[[str, QueryResponse, QueryToken], None],
        on_error: Callable[[str, str], None],
        on_metrics: Callable[[str, object], None],
    ) -> None:
        self.on_state = on_state
        self.on_transcript = on_transcript
        self.on_question = on_question
        self.on_answer = on_answer
        self.on_error = on_error
        self.on_metrics = on_metrics


class VoiceService:
    def __init__(
        self,
        *,
        audio_settings: AudioSettings,
        voice_settings: VoiceSettings,
        microphone: MicrophoneInput,
        player: AudioPlayer,
        asr: RealtimeASR | None,
        tts: StreamingTTS | None,
        query_engine: QueryEngine,
        terminology: TerminologyCorrector,
        state_provider: Callable[[], GameState],
        callbacks: VoiceCallbacks,
    ) -> None:
        self.audio_settings = audio_settings
        self.voice_settings = voice_settings
        self.microphone = microphone
        self.player = player
        self.asr = asr
        self.tts = tts
        self.query_engine = query_engine
        self.terminology = terminology
        self.state_provider = state_provider
        self.callbacks = callbacks
        self.registry = SessionRegistry()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oriens-voice")
        self._query_future: Future | None = None
        self._tts_future: Future | None = None
        self._closed = False

    def devices(self) -> tuple[tuple[str, str], ...]:
        return self.microphone.devices()

    def press(self, device_id: str | None) -> str | None:
        if self._closed:
            return None
        self.cancel()
        if self.asr is None:
            self.callbacks.on_error("", "实时语音离线不可用；文字提问和游戏建议仍可使用。")
            self.callbacks.on_state("", VoiceState.OFFLINE)
            return None
        session = self.registry.replace()
        started = time.perf_counter()
        self.player.interrupt()
        try:
            session.asr = self.asr.start(
                session.request_id,
                session.cancel_event,
                lambda transcript: self._on_transcript(session, transcript),
                lambda error: self._on_error(session, error),
            )
            self.registry.register_cancel(session.request_id, session.asr.cancel)
            self.registry.register_cancel(session.request_id, self.player.interrupt)
            self.registry.register_cancel(session.request_id, self.microphone.stop)
            self.microphone.start(device_id, lambda chunk: self._on_audio(session, chunk))
        except (AudioDeviceUnavailable, VoiceError) as exc:
            self.registry.cancel_current()
            self.callbacks.on_error(session.request_id, str(exc))
            return None
        session.metrics.capture_start_ms = (time.perf_counter() - started) * 1000
        self._set_state(session, VoiceState.LISTENING)
        return session.request_id

    def release(self) -> None:
        session = self.registry.current
        if session is None or session.cancel_event.is_set() or session.asr is None:
            return
        self.microphone.stop()
        try:
            validate_recording(session.chunks, self.audio_settings)
        except VoiceInputRejected as exc:
            self.registry.cancel_current()
            self.callbacks.on_error(session.request_id, str(exc))
            self._set_state(session, VoiceState.CANCELLED)
            return
        self._set_state(session, VoiceState.RECOGNIZING)
        session.asr.commit()

    def ask_text(self, question: str, *, speak: bool = False) -> str | None:
        if self._closed:
            return None
        self.cancel()
        text = question.strip()
        if not text:
            self.callbacks.on_error("", "请输入问题。")
            return None
        session = self.registry.replace()
        session.final_transcript = text
        self.player.interrupt()
        self.callbacks.on_question(session.request_id, text)
        self._submit_query(session, text, speak=speak)
        return session.request_id

    def speak_validated(self, text: str) -> str | None:
        """只接受已经通过业务 schema 和过期校验的短文本。"""

        if self._closed or self.tts is None or not text.strip():
            return None
        self.cancel()
        session = self.registry.replace()
        self.player.interrupt()
        self.registry.register_cancel(session.request_id, self.player.interrupt)
        self._submit_tts(session, text.strip())
        return session.request_id

    def cancel(self) -> None:
        session = self.registry.current
        self.registry.cancel_current()
        if self._query_future is not None:
            self._query_future.cancel()
        if self._tts_future is not None:
            self._tts_future.cancel()
        self.player.interrupt()
        self.microphone.stop()
        if session is not None:
            self.callbacks.on_state(session.request_id, VoiceState.CANCELLED)

    def room_changed(self) -> None:
        self.cancel()

    def is_current(self, request_id: str) -> bool:
        return self.registry.is_current(request_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel()
        self.microphone.close()
        self.player.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _on_audio(self, session: VoiceSession, chunk: AudioChunk) -> None:
        if not self.registry.is_current(session.request_id) or session.asr is None:
            return
        session.chunks.append(chunk)
        duration = sum(item.duration_ms for item in session.chunks)
        if duration > self.audio_settings.max_recording_seconds * 1000:
            self.registry.cancel_current()
            self.callbacks.on_error(session.request_id, "录音已达到时长上限，已停止。")
            return
        try:
            session.asr.send_audio(chunk)
        except VoiceError as exc:
            self._on_error(session, exc)

    def _on_transcript(self, session: VoiceSession, transcript: Transcript) -> None:
        if not self.registry.is_current(session.request_id):
            return
        corrected = Transcript(
            self.terminology.correct(transcript.text), transcript.final, transcript.language
        )
        elapsed = (time.perf_counter() - session.created_at) * 1000
        if corrected.final:
            if corrected.text == session.final_transcript:
                return
            session.final_transcript = corrected.text
            session.metrics.asr_final_ms = elapsed
            self.callbacks.on_transcript(session.request_id, corrected)
            self.callbacks.on_question(session.request_id, corrected.text)
            self._submit_query(session, corrected.text, speak=True)
        else:
            if session.metrics.asr_first_partial_ms == 0:
                session.metrics.asr_first_partial_ms = elapsed
            self.callbacks.on_transcript(session.request_id, corrected)

    def _submit_query(self, session: VoiceSession, question: str, *, speak: bool) -> None:
        if not self.registry.is_current(session.request_id):
            return
        state = deepcopy(self.state_provider())
        self._set_state(session, VoiceState.THINKING)
        started = time.perf_counter()
        self._query_future = self._executor.submit(
            self.query_engine.ask,
            question,
            state,
            session.request_id,
            session.cancel_event,
        )

        def done(future: Future) -> None:
            try:
                response, token = future.result()
            except (VoiceCancelled, ModelCancelled):
                return
            except QueryError as exc:
                self._on_error(session, VoiceError(str(exc)))
                return
            except Exception:
                self._on_error(session, VoiceError("回答生成失败，已安全忽略本次结果。"))
                return
            if not self.registry.is_current(session.request_id):
                return
            current = self.state_provider()
            if not token.is_current(current, session.request_id):
                self.cancel()
                return
            total_ms = (time.perf_counter() - started) * 1000
            session.metrics.rag_ms = response.retrieval_latency_ms
            session.metrics.model_first_text_ms = max(0.0, total_ms - response.retrieval_latency_ms)
            self.callbacks.on_answer(session.request_id, response, token)
            if speak and self.tts is not None:
                self._submit_tts(session, response.answer)
            else:
                self._set_state(session, VoiceState.IDLE)
                self.callbacks.on_metrics(session.request_id, session.metrics)

        self._query_future.add_done_callback(done)

    def _submit_tts(self, session: VoiceSession, answer: str) -> None:
        segments = segment_tts_text(answer, self.voice_settings.tts_max_segment_chars)
        if not segments or not self.registry.is_current(session.request_id):
            self._set_state(session, VoiceState.IDLE)
            return
        started = time.perf_counter()
        first = True

        def on_audio(chunk: AudioChunk) -> None:
            nonlocal first
            if not self.registry.is_current(session.request_id):
                return
            if first:
                first = False
                elapsed = (time.perf_counter() - started) * 1000
                session.metrics.tts_first_audio_ms = elapsed
                session.metrics.first_audio_end_to_end_ms = (
                    time.perf_counter() - session.created_at
                ) * 1000
                self._set_state(session, VoiceState.SPEAKING)
            self.player.enqueue(chunk, session.request_id)
            session.metrics.queue_peak = max(
                session.metrics.queue_peak, int(getattr(self.player, "peak_size", 0))
            )

        def work() -> None:
            assert self.tts is not None
            self.tts.synthesize(
                session.request_id, segments, session.cancel_event, on_audio
            )
            self.player.wait_until_idle(session.cancel_event, self.voice_settings.tts_timeout_seconds)

        self._tts_future = self._executor.submit(work)

        def done(future: Future) -> None:
            try:
                future.result()
            except VoiceCancelled:
                return
            except (VoiceError, AudioError):
                if self.registry.is_current(session.request_id):
                    self.callbacks.on_error(
                        session.request_id, "语音播报失败，文字回答仍可查看。"
                    )
            except Exception:
                # Worker 边界必须收口意外的网络/音频异常，避免 Future 回调打印堆栈。
                if self.registry.is_current(session.request_id):
                    self.callbacks.on_error(
                        session.request_id, "语音播报失败，文字回答仍可查看。"
                    )
            if self.registry.is_current(session.request_id):
                self._set_state(session, VoiceState.IDLE)
                self.callbacks.on_metrics(session.request_id, session.metrics)
                self.registry.finish(session.request_id)

        self._tts_future.add_done_callback(done)

    def _on_error(self, session: VoiceSession, error: VoiceError) -> None:
        if not self.registry.is_current(session.request_id):
            return
        self.callbacks.on_error(session.request_id, str(error))
        self.registry.cancel_current()

    def _set_state(self, session: VoiceSession, state: VoiceState) -> None:
        session.state = state
        self.callbacks.on_state(session.request_id, state)
