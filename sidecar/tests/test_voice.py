from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from threading import Event
import time
import unittest

from oriens.audio import AudioChunk, AudioFormat
from oriens.config import load_config
from oriens.voice import (
    CosyVoiceStreamingTTS,
    QwenRealtimeASR,
    TerminologyCorrector,
    Transcript,
    VoiceCancelled,
    VoiceError,
)


class ScriptedTransport:
    scripts: list[list[str | bytes | Exception]] = []
    instances: list["ScriptedTransport"] = []

    def __init__(self, url, headers, timeout):
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.sent = []
        self.closed = False
        self.script = list(self.scripts.pop(0))
        self.instances.append(self)

    def connect(self):
        if self.script and isinstance(self.script[0], Exception):
            raise self.script.pop(0)

    def send_json(self, value):
        self.sent.append(value)

    def receive(self, cancel):
        if cancel.is_set():
            raise VoiceCancelled("已取消")
        if not self.script:
            raise AssertionError("测试传输事件耗尽")
        # ASR 服务只会在收到音频后返回转写；阻止测试脚本制造不可能的时序。
        if any(
            isinstance(item, str) and "input_audio_transcription" in item
            for item in self.script[:1]
        ):
            if not any(value.get("type") == "input_audio_buffer.append" for value in self.sent):
                time.sleep(0.001)
                return None
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class VoiceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        ScriptedTransport.scripts.clear()
        ScriptedTransport.instances.clear()
        self.config = load_config()

    def test_terminology_correction_loads_only_finite_entity_dictionary(self) -> None:
        corrector = TerminologyCorrector.from_entities(
            Path("data/knowledge/rag-v2.1/entities.jsonl"), limit=40
        )
        self.assertEqual(corrector.correct("我想去boss rush再拿妈刀"), "我想去Boss Rush再拿妈妈的菜刀")

    def test_qwen_protocol_emits_partial_final_and_never_puts_secret_in_url(self) -> None:
        ScriptedTransport.scripts.append([
            json.dumps({"type": "session.updated"}),
            json.dumps({"type": "conversation.item.input_audio_transcription.text", "text": "硫", "stash": "磺火"}),
            json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "硫磺火"}),
            json.dumps({"type": "session.finished"}),
        ])
        transcripts: list[Transcript] = []
        errors = []
        adapter = QwenRealtimeASR(
            self.config.voice, self.config.audio, "test-secret", "workspace-test",
            transport_factory=ScriptedTransport,
        )
        cancel = Event()
        session = adapter.start("r1", cancel, transcripts.append, errors.append)
        session.send_audio(AudioChunk(b"\x01\x00" * 1600, AudioFormat(16000), 0))
        session.commit()
        session._thread.join(timeout=2)  # internal worker must terminate in bounded time
        self.assertEqual([(item.text, item.final) for item in transcripts], [("硫磺火", False), ("硫磺火", True)])
        transport = ScriptedTransport.instances[0]
        self.assertNotIn("test-secret", transport.url)
        self.assertEqual(transport.headers["Authorization"], "Bearer test-secret")
        self.assertEqual(transport.headers["OpenAI-Beta"], "realtime=v1")
        sent_types = [item.get("type") for item in transport.sent]
        self.assertIn("input_audio_buffer.append", sent_types)
        self.assertIn("input_audio_buffer.commit", sent_types)
        self.assertIn("session.finish", sent_types)
        self.assertFalse(errors)

    def test_qwen_reconnects_and_replays_buffer(self) -> None:
        from oriens.websocket import WebSocketError

        ScriptedTransport.scripts.extend([
            [WebSocketError("断线")],
            [json.dumps({"type": "session.updated"}), json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "重连成功"}), json.dumps({"type": "session.finished"})],
        ])
        values = []
        adapter = QwenRealtimeASR(
            self.config.voice, self.config.audio, "secret", "workspace",
            transport_factory=ScriptedTransport,
        )
        session = adapter.start("r2", Event(), values.append, lambda error: self.fail(str(error)))
        session.send_audio(AudioChunk(b"\x01\x00" * 1600, AudioFormat(16000), 0))
        session.commit()
        session._thread.join(timeout=2)
        self.assertEqual(values[-1].text, "重连成功")
        replay_types = [item.get("type") for item in ScriptedTransport.instances[-1].sent]
        self.assertIn("input_audio_buffer.append", replay_types)

    def test_qwen_commit_timeout_is_bounded_and_sanitized(self) -> None:
        class HangingTransport(ScriptedTransport):
            def receive(self, cancel):
                if cancel.is_set():
                    raise VoiceCancelled("已取消")
                time.sleep(0.001)
                return None

        errors: list[VoiceError] = []
        ScriptedTransport.scripts.append([])
        settings = replace(self.config.voice, asr_timeout_seconds=0.02, asr_max_retries=0)
        adapter = QwenRealtimeASR(
            settings, self.config.audio, "secret", "workspace",
            transport_factory=HangingTransport,
        )
        session = adapter.start("timeout", Event(), lambda value: None, errors.append)
        session.send_audio(AudioChunk(b"\x01\x00" * 1600, AudioFormat(16000), 0))
        session.commit()
        session._thread.join(timeout=1)
        self.assertFalse(session._thread.is_alive())
        self.assertIn("实时语音识别超时", str(errors[0]))

    def test_cosyvoice_streams_binary_audio_and_uses_one_task_id(self) -> None:
        ScriptedTransport.scripts.append([
            json.dumps({"header": {"event": "task-started"}}),
            json.dumps({"header": {"event": "result-generated"}}),
            b"\x00\x00" * 100,
            json.dumps({"header": {"event": "task-finished"}}),
        ])
        chunks = []
        adapter = CosyVoiceStreamingTTS(
            self.config.voice, "secret", "workspace", transport_factory=ScriptedTransport
        )
        adapter.synthesize("r3", ("第一段。", "第二段。"), Event(), chunks.append)
        self.assertEqual(len(chunks), 1)
        sent = ScriptedTransport.instances[0].sent
        task_ids = {item["header"]["task_id"] for item in sent}
        self.assertEqual(len(task_ids), 1)
        self.assertEqual(sent[0]["payload"]["parameters"]["language_hints"], ["zh"])
        self.assertEqual([item["header"]["action"] for item in sent], ["run-task", "continue-task", "continue-task", "finish-task"])


if __name__ == "__main__":
    unittest.main()
