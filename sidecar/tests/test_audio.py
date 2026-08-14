from __future__ import annotations

import math
from threading import Event
import time
import unittest

from oriens.audio import AudioChunk, AudioError, AudioFormat, QueuedAudioPlayer, chunk_pcm
from sidecar.tests.test_support import load_test_config as load_config
from oriens.voice import VoiceInputRejected, segment_tts_text, validate_recording


def tone(duration_ms: int, amplitude: int = 1200, sample_rate: int = 16000) -> bytes:
    samples = sample_rate * duration_ms // 1000
    values = bytearray()
    for index in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
        values.extend(value.to_bytes(2, "little", signed=True))
    return bytes(values)


class AudioTests(unittest.TestCase):
    def test_pcm_chunks_keep_sample_boundaries_and_duration(self) -> None:
        fmt = AudioFormat(16000)
        chunks = chunk_pcm(tone(250), fmt, 100)
        self.assertEqual([round(item.duration_ms) for item in chunks], [100, 100, 50])
        self.assertEqual([item.sequence for item in chunks], [0, 1, 2])
        with self.assertRaises(AudioError):
            chunk_pcm(b"\x00", fmt, 100)

    def test_empty_short_silent_and_noise_inputs_are_rejected(self) -> None:
        settings = load_config().audio
        fmt = AudioFormat(16000)
        with self.assertRaises(VoiceInputRejected):
            validate_recording((), settings)
        with self.assertRaises(VoiceInputRejected):
            validate_recording((AudioChunk(tone(100), fmt, 0),), settings)
        with self.assertRaises(VoiceInputRejected):
            validate_recording((AudioChunk(b"\x00\x00" * 4800, fmt, 0),), settings)
        with self.assertRaises(VoiceInputRejected):
            validate_recording((AudioChunk(tone(300, amplitude=100), fmt, 0),), settings)
        self.assertEqual(len(validate_recording((AudioChunk(tone(300), fmt, 0),), settings)), 1)

    def test_playback_queue_is_bounded_and_interrupt_clears_old_generation(self) -> None:
        played: list[int] = []
        player = QueuedAudioPlayer(lambda chunk: (time.sleep(0.03), played.append(chunk.sequence)), 2)
        fmt = AudioFormat(24000)
        player.enqueue(AudioChunk(b"\0\0" * 480, fmt, 1), "old")
        player.enqueue(AudioChunk(b"\0\0" * 480, fmt, 2), "old")
        with self.assertRaises(AudioError):
            player.enqueue(AudioChunk(b"\0\0" * 480, fmt, 3), "old")
        player.interrupt()
        player.enqueue(AudioChunk(b"\0\0" * 480, fmt, 9), "new")
        self.assertTrue(player.wait_until_idle(Event(), 1.0))
        player.close()
        self.assertIn(9, played)
        self.assertNotIn(2, played)

    def test_tts_segmentation_removes_urls_and_internal_ids(self) -> None:
        values = segment_tts_text(
            "建议先拿道具。来源 https://example.com chunk_id=secret。然后继续。", 12
        )
        joined = "".join(values)
        self.assertNotIn("http", joined)
        self.assertNotIn("chunk_id", joined.lower())
        self.assertTrue(all(len(item) <= 13 for item in values))


if __name__ == "__main__":
    unittest.main()
