"""阶段 3 音频格式、设备抽象和可中断有界播放队列。"""

from __future__ import annotations

from dataclasses import dataclass
from array import array
import math
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable, Protocol


class AudioError(RuntimeError):
    """可安全展示的音频错误。"""


class AudioDeviceUnavailable(AudioError):
    pass


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int = 1
    sample_width_bytes: int = 2
    encoding: str = "pcm_s16le"

    def validate(self) -> None:
        if self.encoding != "pcm_s16le" or self.channels != 1 or self.sample_width_bytes != 2:
            raise AudioError("阶段 3 仅支持单声道 16-bit PCM")
        if self.sample_rate not in {8000, 16000, 22050, 24000, 44100, 48000}:
            raise AudioError("不支持的音频采样率")

    def bytes_for_ms(self, duration_ms: int) -> int:
        if duration_ms <= 0:
            raise AudioError("音频块时长必须为正数")
        return self.sample_rate * self.channels * self.sample_width_bytes * duration_ms // 1000


@dataclass(frozen=True, slots=True)
class AudioChunk:
    data: bytes
    format: AudioFormat
    sequence: int

    @property
    def duration_ms(self) -> float:
        denominator = self.format.sample_rate * self.format.channels * self.format.sample_width_bytes
        return len(self.data) * 1000.0 / denominator

    @property
    def rms(self) -> int:
        if not self.data:
            return 0
        samples = array("h")
        samples.frombytes(self.data)
        return math.isqrt(sum(value * value for value in samples) // len(samples))


def chunk_pcm(data: bytes, audio_format: AudioFormat, chunk_duration_ms: int) -> tuple[AudioChunk, ...]:
    audio_format.validate()
    size = audio_format.bytes_for_ms(chunk_duration_ms)
    frame_size = audio_format.channels * audio_format.sample_width_bytes
    if len(data) % frame_size:
        raise AudioError("PCM 数据没有按样本边界对齐")
    return tuple(
        AudioChunk(data[offset : offset + size], audio_format, index)
        for index, offset in enumerate(range(0, len(data), size))
        if data[offset : offset + size]
    )


class MicrophoneInput(Protocol):
    def devices(self) -> tuple[tuple[str, str], ...]: ...
    def start(self, device_id: str | None, on_chunk: Callable[[AudioChunk], None]) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class AudioPlayer(Protocol):
    def enqueue(self, chunk: AudioChunk, generation: str) -> None: ...
    def interrupt(self) -> None: ...
    def wait_until_idle(self, cancel: Event, timeout: float) -> bool: ...
    def close(self) -> None: ...


class MemoryMicrophone:
    """测试设备；只有显式 feed 才产生音频，永不后台录音。"""

    def __init__(self) -> None:
        self._callback: Callable[[AudioChunk], None] | None = None
        self.closed = False

    def devices(self) -> tuple[tuple[str, str], ...]:
        return (("memory", "内存测试麦克风"),)

    def start(self, device_id: str | None, on_chunk: Callable[[AudioChunk], None]) -> None:
        if self.closed:
            raise AudioDeviceUnavailable("麦克风已关闭")
        if device_id not in {None, "memory"}:
            raise AudioDeviceUnavailable("所选麦克风不可用")
        self._callback = on_chunk

    def feed(self, chunk: AudioChunk) -> None:
        if self._callback is not None:
            self._callback(chunk)

    def stop(self) -> None:
        self._callback = None

    def close(self) -> None:
        self.stop()
        self.closed = True


class UnavailableMicrophone:
    """设备失败时保留业务闭环，但明确拒绝录音。"""

    def devices(self) -> tuple[tuple[str, str], ...]:
        return ()

    def start(self, device_id: str | None, on_chunk: Callable[[AudioChunk], None]) -> None:
        raise AudioDeviceUnavailable("未找到可用麦克风，请检查设备或系统权限。")

    def stop(self) -> None:
        return

    def close(self) -> None:
        return


class NullAudioPlayer:
    """没有播放设备时的安全降级；不会缓存或播放任何内容。"""

    peak_size = 0

    def enqueue(self, chunk: AudioChunk, generation: str) -> None:
        raise AudioDeviceUnavailable("音频播放设备不可用")

    def interrupt(self) -> None:
        return

    def wait_until_idle(self, cancel: Event, timeout: float) -> bool:
        return True

    def close(self) -> None:
        return


class QueuedAudioPlayer:
    """后端无关的有界播放队列；新一代会话会清空旧音频。"""

    def __init__(self, sink: Callable[[AudioChunk], None], max_chunks: int) -> None:
        self._sink = sink
        self._queue: Queue[tuple[str, AudioChunk] | None] = Queue(maxsize=max_chunks)
        self._generation: str | None = None
        self._closed = Event()
        self._lock = Lock()
        self.peak_size = 0
        self._thread = Thread(target=self._run, name="oriens-audio-playback", daemon=True)
        self._thread.start()

    def enqueue(self, chunk: AudioChunk, generation: str) -> None:
        if self._closed.is_set():
            raise AudioDeviceUnavailable("音频播放器已关闭")
        with self._lock:
            if self._generation is None:
                self._generation = generation
            if self._generation != generation:
                return
        try:
            self._queue.put_nowait((generation, chunk))
        except Full:
            raise AudioError("播放队列已满，已停止本次播报") from None
        self.peak_size = max(self.peak_size, self._queue.qsize())

    def interrupt(self) -> None:
        with self._lock:
            self._generation = None
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break

    def wait_until_idle(self, cancel: Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not cancel.is_set() and time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            cancel.wait(0.01)
        return False

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.interrupt()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                self._queue.task_done()
                return
            generation, chunk = item
            with self._lock:
                active = generation == self._generation
            if active:
                try:
                    self._sink(chunk)
                finally:
                    self._queue.task_done()
            else:
                self._queue.task_done()


class QtMicrophoneInput:
    """PySide6 非阻塞麦克风；只有 start 后才创建并启动 QAudioSource。"""

    def __init__(self, audio_format: AudioFormat, chunk_duration_ms: int) -> None:
        self._format = audio_format
        self._chunk_duration_ms = chunk_duration_ms
        self._source = None
        self._device = None
        self._io = None
        self._callback: Callable[[AudioChunk], None] | None = None
        self._buffer = bytearray()
        self._sequence = 0

    def devices(self) -> tuple[tuple[str, str], ...]:
        try:
            from PySide6.QtMultimedia import QMediaDevices
        except ImportError:
            return ()
        return tuple(
            (bytes(device.id()).hex(), device.description())
            for device in QMediaDevices.audioInputs()
        )

    def start(self, device_id: str | None, on_chunk: Callable[[AudioChunk], None]) -> None:
        if self._source is not None:
            raise AudioError("麦克风已经在录音")
        try:
            from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
        except ImportError:
            raise AudioDeviceUnavailable("当前环境没有可用的音频组件") from None
        devices = QMediaDevices.audioInputs()
        selected = None
        for device in devices:
            if device_id is None or bytes(device.id()).hex() == device_id:
                selected = device
                break
        if selected is None:
            raise AudioDeviceUnavailable("未找到可用麦克风，请检查设备或系统权限。")
        qt_format = QAudioFormat()
        qt_format.setSampleRate(self._format.sample_rate)
        qt_format.setChannelCount(self._format.channels)
        qt_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not selected.isFormatSupported(qt_format):
            raise AudioDeviceUnavailable("所选麦克风不支持 16 kHz 单声道 PCM。")
        try:
            self._source = QAudioSource(selected, qt_format)
            self._callback = on_chunk
            self._buffer.clear()
            self._sequence = 0
            self._io = self._source.start()
            if self._io is None:
                raise AudioDeviceUnavailable("麦克风启动失败，请检查系统权限。")
            self._io.readyRead.connect(self._read_ready)
        except AudioDeviceUnavailable:
            self.stop()
            raise
        except Exception:
            self.stop()
            raise AudioDeviceUnavailable("麦克风初始化失败，请检查设备或系统权限。") from None

    def _read_ready(self) -> None:
        if self._io is None or self._callback is None:
            return
        self._buffer.extend(bytes(self._io.readAll()))
        size = self._format.bytes_for_ms(self._chunk_duration_ms)
        while len(self._buffer) >= size:
            data = bytes(self._buffer[:size])
            del self._buffer[:size]
            self._callback(AudioChunk(data, self._format, self._sequence))
            self._sequence += 1

    def stop(self) -> None:
        if self._io is not None:
            try:
                self._io.readyRead.disconnect(self._read_ready)
            except (RuntimeError, TypeError):
                pass
        if self._source is not None:
            try:
                self._source.stop()
            except RuntimeError:
                pass
        self._io = None
        self._source = None
        self._callback = None
        self._buffer.clear()

    def close(self) -> None:
        self.stop()


class QtAudioPlayer:
    """Qt Multimedia PCM 播放适配器；所有 QAudioSink 操作都留在 UI 线程。"""

    def __init__(self, audio_format: AudioFormat, max_chunks: int) -> None:
        try:
            from PySide6.QtCore import QObject, QTimer, Signal
            from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
        except ImportError:
            raise AudioDeviceUnavailable("当前环境没有可用的音频播放组件") from None

        class _Bridge(QObject):
            wake = Signal()
            stop = Signal()

        audio_format.validate()
        device = QMediaDevices.defaultAudioOutput()
        qt_format = QAudioFormat()
        qt_format.setSampleRate(audio_format.sample_rate)
        qt_format.setChannelCount(audio_format.channels)
        qt_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isNull() or not device.isFormatSupported(qt_format):
            raise AudioDeviceUnavailable(
                f"默认播放设备不支持 {audio_format.sample_rate} Hz 单声道 PCM。"
            )
        self._format = audio_format
        self._queue: Queue[tuple[str, AudioChunk]] = Queue(maxsize=max_chunks)
        self._generation: str | None = None
        self._pending: tuple[str, AudioChunk] | None = None
        self._lock = Lock()
        self._closed = False
        self._play_until = 0.0
        self.peak_size = 0
        self._bridge = _Bridge()
        self._sink = QAudioSink(device, qt_format)
        self._sink.setBufferSize(audio_format.bytes_for_ms(500))
        self._io = self._sink.start()
        if self._io is None:
            raise AudioDeviceUnavailable("音频播放设备启动失败。")
        self._timer = QTimer()
        self._timer.setInterval(5)
        self._timer.timeout.connect(self._drain)
        self._bridge.wake.connect(self._drain)
        self._bridge.stop.connect(self._reset_on_ui_thread)
        self._timer.start()

    def enqueue(self, chunk: AudioChunk, generation: str) -> None:
        if self._closed:
            raise AudioDeviceUnavailable("音频播放器已关闭")
        if chunk.format != self._format:
            raise AudioError("播放音频格式与设备格式不匹配")
        with self._lock:
            if self._generation is None:
                self._generation = generation
            if self._generation != generation:
                return
        try:
            self._queue.put_nowait((generation, chunk))
        except Full:
            raise AudioError("播放队列已满，已停止本次播报") from None
        self.peak_size = max(self.peak_size, self._queue.qsize())
        with self._lock:
            self._play_until = max(self._play_until, time.monotonic()) + chunk.duration_ms / 1000
        self._bridge.wake.emit()

    def interrupt(self) -> None:
        with self._lock:
            self._generation = None
            self._play_until = time.monotonic()
            pending, self._pending = self._pending, None
        if pending is not None:
            self._queue.task_done()
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break
        self._bridge.stop.emit()

    def wait_until_idle(self, cancel: Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not cancel.is_set() and time.monotonic() < deadline:
            with self._lock:
                finished = time.monotonic() >= self._play_until
            if self._queue.unfinished_tasks == 0 and finished:
                return True
            cancel.wait(0.01)
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.interrupt()
        self._timer.stop()
        self._sink.stop()

    def _drain(self) -> None:
        if self._closed or self._io is None:
            return
        while self._sink.bytesFree() > 0:
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                generation, chunk = pending
            else:
                try:
                    generation, chunk = self._queue.get_nowait()
                except Empty:
                    return
            with self._lock:
                active = generation == self._generation
            if active:
                data = chunk.data[: self._sink.bytesFree()]
                written = self._io.write(data)
                if written < len(chunk.data):
                    if written < 0:
                        written = 0
                    remainder = AudioChunk(chunk.data[written:], chunk.format, chunk.sequence)
                    with self._lock:
                        if generation == self._generation:
                            self._pending = (generation, remainder)
                            return
            self._queue.task_done()

    def _reset_on_ui_thread(self) -> None:
        if self._closed:
            return
        self._sink.reset()
        self._io = self._sink.start()
