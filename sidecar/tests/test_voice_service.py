from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from oriens.audio import AudioChunk, AudioDeviceUnavailable, AudioFormat, MemoryMicrophone, QueuedAudioPlayer
from oriens.budget import BudgetTracker
from sidecar.tests.test_support import load_test_config as load_config
from oriens.modeling import AdapterResponse, ModelRouter, ModelUsage
from oriens.query import QueryEngine
from oriens.rag import RagService
from oriens.rag_pipeline import build_corpus, build_keyword_index
from oriens.state import GameState
from oriens.voice import MockRealtimeASR, MockStreamingTTS, TerminologyCorrector, VoiceError, VoiceState
from oriens.voice_service import VoiceCallbacks, VoiceService


class ForgedCitationAdapter:
    def complete(self, model, model_request, cancel):
        return AdapterResponse(
            json.dumps({
                "advice": "这条在线回答带有错误引用。",
                "reason": "模拟模型错误。",
                "confidence": 0.9,
                "sources": ["forged"],
                "state_seq": model_request.metadata["state_seq"],
            }, ensure_ascii=False),
            ModelUsage(20, 10),
        )


class VoiceServiceTests(unittest.TestCase):
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

    def _service(self, asr=None, router=None):
        state = GameState(run_id="VOICE:0", active=True, last_seq=1, context={"room_index": 4, "room_spawn_seed": 9})
        microphone = MemoryMicrophone()
        played = []
        player = QueuedAudioPlayer(played.append, 16)
        budget = BudgetTracker(self.config.budget.run_limit_cny)
        budget.set_run("VOICE:0")
        query = QueryEngine(
            RagService(self.index),
            router or ModelRouter(self.config, online=False, api_key=None),
            budget, game_version=self.config.rag.game_version,
        )
        events = {"states": [], "transcripts": [], "questions": [], "answers": [], "errors": [], "metrics": []}
        callbacks = VoiceCallbacks(
            on_state=lambda rid, value: events["states"].append(value),
            on_transcript=lambda rid, value: events["transcripts"].append(value),
            on_question=lambda rid, value: events["questions"].append(value),
            on_answer=lambda rid, value, token: events["answers"].append(value),
            on_error=lambda rid, value: events["errors"].append(value),
            on_metrics=lambda rid, value: events["metrics"].append(value),
        )
        service = VoiceService(
            audio_settings=self.config.audio, voice_settings=self.config.voice,
            microphone=microphone, player=player,
            asr=asr or MockRealtimeASR("硫磺火有什么效果", ("硫磺",)),
            tts=MockStreamingTTS(AudioFormat(self.config.audio.playback_sample_rate)), query_engine=query,
            terminology=TerminologyCorrector({}), state_provider=lambda: state,
            callbacks=callbacks,
        )
        return service, microphone, state, events, played

    def test_mock_asr_rag_model_tts_full_loop_and_metrics(self) -> None:
        service, microphone, state, events, played = self._service()
        request_id = service.press("memory")
        self.assertIsNotNone(request_id)
        microphone.feed(AudioChunk(b"\x10\x04" * 4800, AudioFormat(16000), 0))
        service.release()
        deadline = time.monotonic() + 2
        while not events["metrics"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(events["answers"])
        self.assertTrue(played)
        self.assertIn(VoiceState.LISTENING, events["states"])
        self.assertIn(VoiceState.RECOGNIZING, events["states"])
        self.assertIn(VoiceState.THINKING, events["states"])
        self.assertIn(VoiceState.SPEAKING, events["states"])
        self.assertEqual(events["errors"], [])
        service.close()
        self.assertTrue(microphone.closed)

    def test_new_question_cancels_old_and_room_change_clears_playback(self) -> None:
        service, microphone, state, events, played = self._service()
        old = service.press("memory")
        microphone.feed(AudioChunk(b"\x10\x04" * 4800, AudioFormat(16000), 0))
        new = service.press("memory")
        self.assertNotEqual(old, new)
        self.assertFalse(service.is_current(old))
        state.context = {"room_index": 5, "room_spawn_seed": 10}
        service.room_changed()
        self.assertFalse(service.is_current(new))
        self.assertIn(VoiceState.CANCELLED, events["states"])
        service.close()

    def test_missing_asr_is_fully_offline_without_starting_microphone(self) -> None:
        service, microphone, state, events, played = self._service()
        service.asr = None
        self.assertIsNone(service.press("memory"))
        self.assertIn(VoiceState.OFFLINE, events["states"])
        self.assertIsNone(microphone._callback)
        service.close()

    def test_audio_device_failure_cancels_asr_and_keeps_text_service_available(self) -> None:
        class FailingMicrophone(MemoryMicrophone):
            def start(self, device_id, on_chunk):
                raise AudioDeviceUnavailable("测试设备不可用")

        service, microphone, state, events, played = self._service()
        failing = FailingMicrophone()
        service.microphone = failing
        self.assertIsNone(service.press("missing"))
        self.assertIn("测试设备不可用", events["errors"])
        self.assertIsNotNone(service.ask_text("硫磺火有什么效果"))
        service.close()

    def test_tts_network_failure_preserves_validated_text_answer(self) -> None:
        class FailingTTS:
            def synthesize(self, request_id, text_segments, cancel, on_audio):
                raise VoiceError("模拟网络中断")

        service, microphone, state, events, played = self._service()
        service.tts = FailingTTS()
        service.ask_text("硫磺火有什么效果", speak=True)
        deadline = time.monotonic() + 2
        while not events["errors"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(events["answers"])
        self.assertIn("硫磺火", events["answers"][0].answer)
        self.assertIn("语音播报失败，文字回答仍可查看。", events["errors"])
        service.close()

    def test_invalid_online_citation_falls_back_and_still_speaks(self) -> None:
        router = ModelRouter(
            self.config,
            online=True,
            api_key="test-only",
            adapters={"advice": ForgedCitationAdapter()},
        )
        service, _microphone, _state, events, played = self._service(router=router)

        service.ask_text("硫磺火有什么效果", speak=True)
        deadline = time.monotonic() + 2
        while not events["metrics"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(events["answers"])
        self.assertTrue(events["answers"][0].simulated)
        self.assertIn("格式或来源校验", events["answers"][0].delivery_note or "")
        self.assertTrue(played)
        self.assertEqual(events["errors"], [])
        self.assertIn(VoiceState.SPEAKING, events["states"])
        service.close()

    def test_unexpected_tts_worker_error_is_safely_reported(self) -> None:
        class FailingTTS:
            def synthesize(self, request_id, text_segments, cancel, on_audio):
                raise RuntimeError("模拟底层 TLS 异常")

        service, microphone, state, events, played = self._service()
        service.tts = FailingTTS()
        service.ask_text("硫磺火有什么效果", speak=True)
        deadline = time.monotonic() + 2
        while not events["errors"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn("语音播报失败，文字回答仍可查看。", events["errors"])
        service.close()


if __name__ == "__main__":
    unittest.main()
