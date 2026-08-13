"""PySide6 最小悬浮窗。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import Event
import time
from typing import Any

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

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
from .config import OriensConfig
from .knowledge import LocalItemKnowledgeBase
from .modeling import ModelCancelled
from .query import QueryEngine, QueryResponse, QueryToken
from .protocol import EventParseError, GameEvent, parse_event_line
from .state import EventOrderError, StateStore
from .tailer import LogTailer
from .voice import RealtimeASR, StreamingTTS, TerminologyCorrector, Transcript, VoiceMetrics, VoiceState
from .voice_service import VoiceCallbacks, VoiceService


ROOM_NAMES = {
    0: "无房间类型",
    1: "普通房",
    2: "商店",
    3: "错误房",
    4: "道具房",
    5: "Boss 房",
    6: "小 Boss 房",
    7: "隐藏房",
    8: "超级隐藏房",
    9: "游戏厅",
    10: "诅咒房（刺房）",
    11: "挑战房",
    12: "图书馆",
    13: "献祭房",
    14: "恶魔房",
    15: "天使房",
    16: "地牢房",
    17: "Boss Rush 房",
    18: "以撒的房间",
    19: "荒废房间",
    20: "宝箱房",
    21: "骰子房",
    22: "黑市",
    23: "贪婪模式出口",
    24: "星象房",
    25: "传送入口",
    26: "传送出口",
    27: "隐藏出口房",
    28: "蓝色房间",
    29: "究极隐藏房",
    30: "死斗房",
}


class _AdviceSignals(QObject):
    completed = Signal(object, object)
    failed = Signal(str)


class _VoiceSignals(QObject):
    state = Signal(str, object)
    transcript = Signal(str, object)
    question = Signal(str, str)
    answer = Signal(str, object, object)
    failed = Signal(str, str)
    metrics = Signal(str, object)


class OverlayWindow(QMainWindow):
    def __init__(
        self,
        *,
        config: OriensConfig,
        log_path: Path,
        knowledge: LocalItemKnowledgeBase,
        advice_engine: AdviceEngine,
        budget: BudgetTracker,
        from_start: bool = False,
        online_requested: bool = False,
        api_key_available: bool = False,
        query_engine: QueryEngine | None = None,
        asr: RealtimeASR | None = None,
        tts: StreamingTTS | None = None,
        voice_service: VoiceService | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.knowledge = knowledge
        self.advice_engine = advice_engine
        self.budget = budget
        self.tailer = LogTailer(log_path, from_start=from_start)
        self.store = StateStore()
        self._recent: deque[str] = deque(maxlen=config.app.recent_event_limit)
        self._last_event_at: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oriens-model")
        self._cancel: Event | None = None
        self._future: Future[tuple[AdviceResponse, StateToken]] | None = None
        self._signals = _AdviceSignals(self)
        self._signals.completed.connect(self._show_advice)
        self._signals.failed.connect(self._show_model_error)
        self._voice_signals = _VoiceSignals(self)
        self._voice_signals.state.connect(self._show_voice_state)
        self._voice_signals.transcript.connect(self._show_transcript)
        self._voice_signals.question.connect(self._show_question)
        self._voice_signals.answer.connect(self._show_query_answer)
        self._voice_signals.failed.connect(self._show_voice_error)
        self._voice_signals.metrics.connect(self._show_voice_metrics)
        self._drag_origin: QPoint | None = None
        self._compact_mode = False
        self._normal_size = None
        self._compact_hidden_widgets: list[QWidget] = []
        self.voice_service = voice_service

        self.setWindowTitle("Oriens：你的游戏向导")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(440)
        self.resize(500, 860)
        self._build_ui()

        if self.voice_service is None and query_engine is not None:
            try:
                microphone = QtMicrophoneInput(
                    AudioFormat(config.audio.input_sample_rate),
                    config.audio.chunk_duration_ms,
                )
                player = QtAudioPlayer(
                    AudioFormat(config.audio.playback_sample_rate),
                    config.audio.playback_queue_max_chunks,
                )
                self.voice_service = VoiceService(
                    audio_settings=config.audio,
                    voice_settings=config.voice,
                    microphone=microphone,
                    player=player,
                    asr=asr,
                    tts=tts,
                    query_engine=query_engine,
                    terminology=TerminologyCorrector.from_entities(config.rag.entities_path),
                    state_provider=lambda: self.store.state,
                    callbacks=VoiceCallbacks(
                        on_state=lambda request_id, value: self._voice_signals.state.emit(request_id, value),
                        on_transcript=lambda request_id, value: self._voice_signals.transcript.emit(request_id, value),
                        on_question=lambda request_id, value: self._voice_signals.question.emit(request_id, value),
                        on_answer=lambda request_id, value, token: self._voice_signals.answer.emit(request_id, value, token),
                        on_error=lambda request_id, value: self._voice_signals.failed.emit(request_id, value),
                        on_metrics=lambda request_id, value: self._voice_signals.metrics.emit(request_id, value),
                    ),
                )
            except AudioDeviceUnavailable as exc:
                self.voice_service = VoiceService(
                    audio_settings=config.audio,
                    voice_settings=config.voice,
                    microphone=UnavailableMicrophone(),
                    player=NullAudioPlayer(),
                    asr=None,
                    tts=None,
                    query_engine=query_engine,
                    terminology=TerminologyCorrector.from_entities(config.rag.entities_path),
                    state_provider=lambda: self.store.state,
                    callbacks=VoiceCallbacks(
                        on_state=lambda request_id, value: self._voice_signals.state.emit(request_id, value),
                        on_transcript=lambda request_id, value: self._voice_signals.transcript.emit(request_id, value),
                        on_question=lambda request_id, value: self._voice_signals.question.emit(request_id, value),
                        on_answer=lambda request_id, value, token: self._voice_signals.answer.emit(request_id, value, token),
                        on_error=lambda request_id, value: self._voice_signals.failed.emit(request_id, value),
                        on_metrics=lambda request_id, value: self._voice_signals.metrics.emit(request_id, value),
                    ),
                )
                self.voice_status_label.setText("离线不可用")
                self.voice_hint_label.setText(str(exc) + " 文字提问仍可使用。")
        self._populate_audio_devices()

        if online_requested and api_key_available:
            self.mode_label.setText("在线建议已启用")
        elif online_requested:
            self.mode_label.setText("未找到 DASHSCOPE_API_KEY，已进入离线模拟模式")
        else:
            self.mode_label.setText("离线模拟模式（启动时添加 --online 可启用百炼）")
        if self.voice_service is None or asr is None:
            self.voice_status_label.setText("离线不可用")
            self.ptt_button.setEnabled(False)
            if not online_requested:
                self.voice_hint_label.setText("语音联网默认关闭；文字提问和游戏建议仍可使用。")
            elif not api_key_available:
                self.voice_hint_label.setText("缺少百炼凭据时不会监听麦克风；文字提问和游戏建议仍可用。")
            else:
                self.voice_hint_label.setText("未找到 DASHSCOPE_WORKSPACE_ID，语音联网不可用；文字功能不受影响。")

        self.timer = QTimer(self)
        self.timer.setInterval(config.app.poll_interval_ms)
        self.timer.timeout.connect(self._poll_log)
        self.timer.start()

    def _build_ui(self) -> None:
        shell = QFrame()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(1, 1, 1, 1)

        self.main_scroll = QScrollArea()
        self.main_scroll.setObjectName("mainScroll")
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("overlayContent")
        layout = QVBoxLayout(content)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)
        self.main_scroll.setWidget(content)
        shell_layout.addWidget(self.main_scroll)

        header = QHBoxLayout()
        title = QLabel("ORIENS")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self.connection_label = QLabel("等待日志")
        self.connection_label.setObjectName("status")
        header.addWidget(self.connection_label)
        self.compact_button = QPushButton("—")
        self.compact_button.setObjectName("windowControl")
        self.compact_button.setToolTip("精简模式：只显示输入框和回答框")
        self.compact_button.clicked.connect(self._toggle_compact_mode)
        header.addWidget(self.compact_button)
        close_button = QPushButton("×")
        close_button.setObjectName("close")
        close_button.setToolTip("关闭")
        close_button.clicked.connect(self.close)
        header.addWidget(close_button)
        layout.addLayout(header)

        self.mode_label = QLabel()
        self.mode_label.setObjectName("hint")
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        state_row = QHBoxLayout()
        self.room_card, self.room_label = self._card("当前房间", "尚未进入一局游戏")
        self.resource_card, self.resource_label = self._card(
            "角色资源", "红心 — · 魂心 —\n硬币 — · 钥匙 — · 炸弹 —"
        )
        state_row.addWidget(self.room_card)
        state_row.addWidget(self.resource_card)
        layout.addLayout(state_row)

        self.voice_frame = QFrame()
        self.voice_frame.setObjectName("card")
        voice_layout = QVBoxLayout(self.voice_frame)
        voice_header = QHBoxLayout()
        self.voice_enabled = QCheckBox("启用语音")
        self.voice_enabled.setChecked(self.config.voice.enabled)
        self.voice_enabled.toggled.connect(self._voice_enabled_changed)
        self.voice_status_label = QLabel("未监听")
        self.voice_status_label.setObjectName("status")
        voice_header.addWidget(self.voice_enabled)
        voice_header.addStretch(1)
        voice_header.addWidget(self.voice_status_label)
        voice_layout.addLayout(voice_header)
        device_row = QHBoxLayout()
        self.input_device_label = QLabel("输入设备")
        device_row.addWidget(self.input_device_label)
        self.input_device_combo = QComboBox()
        self.input_device_combo.setMinimumWidth(230)
        self.input_device_combo.setMinimumHeight(38)
        device_row.addWidget(self.input_device_combo, 1)
        voice_layout.addLayout(device_row)
        ptt_row = QHBoxLayout()
        self.ptt_button = QPushButton(f"按住说话（{self.config.voice.push_to_talk_key}）")
        self.ptt_button.setObjectName("primary")
        self.ptt_button.setMinimumHeight(40)
        self.ptt_button.pressed.connect(self._voice_press)
        self.ptt_button.released.connect(self._voice_release)
        self.cancel_voice_button = QPushButton("取消")
        self.cancel_voice_button.setMinimumHeight(40)
        self.cancel_voice_button.clicked.connect(self._voice_cancel)
        ptt_row.addWidget(self.ptt_button, 1)
        ptt_row.addWidget(self.cancel_voice_button)
        voice_layout.addLayout(ptt_row)
        self.voice_hint_label = QLabel("默认不监听；只有按住说话时才打开麦克风。")
        self.voice_hint_label.setObjectName("hint")
        self.voice_hint_label.setWordWrap(True)
        voice_layout.addWidget(self.voice_hint_label)
        self.transcript_label = QLabel("当前字幕：—")
        self.transcript_label.setWordWrap(True)
        self.question_label = QLabel("最终问题：—")
        self.question_label.setWordWrap(True)
        voice_layout.addWidget(self.transcript_label)
        voice_layout.addWidget(self.question_label)
        text_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setMinimumHeight(38)
        self.question_input.setPlaceholderText("也可以键入普通游戏问题")
        self.question_input.returnPressed.connect(self._submit_text_question)
        self.ask_button = QPushButton("提问")
        self.ask_button.setMinimumHeight(38)
        self.ask_button.clicked.connect(self._submit_text_question)
        text_row.addWidget(self.question_input, 1)
        text_row.addWidget(self.ask_button)
        voice_layout.addLayout(text_row)
        self.voice_metrics_label = QLabel("ASR — · RAG — · 模型 — · TTS —")
        self.voice_metrics_label.setObjectName("metrics")
        self.voice_metrics_label.setWordWrap(True)
        voice_layout.addWidget(self.voice_metrics_label)
        layout.addWidget(self.voice_frame)

        self.item_frame = QFrame()
        self.item_frame.setObjectName("card")
        item_layout = QVBoxLayout(self.item_frame)
        self.item_title = QLabel("识别到的道具")
        self.item_title.setObjectName("caption")
        self.item_label = QLabel("等待道具房事件")
        self.item_label.setObjectName("value")
        self.item_label.setWordWrap(True)
        item_layout.addWidget(self.item_title)
        item_layout.addWidget(self.item_label)
        layout.addWidget(self.item_frame)

        self.advice_frame = QFrame()
        self.advice_frame.setObjectName("adviceCard")
        advice_layout = QVBoxLayout(self.advice_frame)
        self.advice_caption = QLabel("短建议")
        self.advice_caption.setObjectName("caption")
        self.advice_label = QLabel("进入已覆盖的道具房后，这里会显示建议。")
        self.advice_label.setObjectName("advice")
        self.advice_label.setWordWrap(True)
        self.reason_label = QLabel("")
        self.reason_label.setObjectName("reason")
        self.reason_label.setWordWrap(True)
        self.source_label = QLabel("")
        self.source_label.setObjectName("source")
        self.source_label.setTextFormat(Qt.TextFormat.RichText)
        self.source_label.setWordWrap(True)
        self.source_label.setOpenExternalLinks(True)
        self.metrics_label = QLabel("置信度 — · 状态序号 —")
        self.metrics_label.setObjectName("metrics")
        advice_layout.addWidget(self.advice_caption)
        advice_layout.addWidget(self.advice_label)
        advice_layout.addWidget(self.reason_label)
        advice_layout.addWidget(self.source_label)
        advice_layout.addWidget(self.metrics_label)
        layout.addWidget(self.advice_frame)

        self.rag_caption = QLabel("检索调试")
        self.rag_caption.setObjectName("caption")
        layout.addWidget(self.rag_caption)
        self.rag_scroll = QScrollArea()
        self.rag_scroll.setObjectName("eventScroll")
        self.rag_scroll.setWidgetResizable(True)
        self.rag_scroll.setMaximumHeight(105)
        self.rag_debug_label = QLabel("尚未执行检索")
        self.rag_debug_label.setObjectName("events")
        self.rag_debug_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rag_debug_label.setWordWrap(True)
        self.rag_scroll.setWidget(self.rag_debug_label)
        layout.addWidget(self.rag_scroll)

        self.budget_frame = QFrame()
        self.budget_frame.setObjectName("card")
        budget_layout = QVBoxLayout(self.budget_frame)
        budget_caption = QLabel("本局费用（估算）")
        budget_caption.setObjectName("caption")
        self.cost_label = QLabel("¥0.000000 · 输入 0 · 输出 0")
        self.cost_label.setObjectName("value")
        self.budget_bar = QProgressBar()
        self.budget_bar.setRange(0, 1000)
        self.budget_bar.setValue(0)
        self.budget_bar.setTextVisible(False)
        budget_layout.addWidget(budget_caption)
        budget_layout.addWidget(self.cost_label)
        budget_layout.addWidget(self.budget_bar)
        layout.addWidget(self.budget_frame)

        self.event_caption = QLabel("最近事件")
        self.event_caption.setObjectName("caption")
        layout.addWidget(self.event_caption)
        self.event_scroll = QScrollArea()
        self.event_scroll.setObjectName("eventScroll")
        self.event_scroll.setWidgetResizable(True)
        self.event_scroll.setMaximumHeight(125)
        self.events_label = QLabel("暂无事件")
        self.events_label.setObjectName("events")
        self.events_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.events_label.setWordWrap(True)
        self.event_scroll.setWidget(self.events_label)
        layout.addWidget(self.event_scroll)

        self._compact_hidden_widgets = [
            self.mode_label,
            self.room_card,
            self.resource_card,
            self.voice_enabled,
            self.voice_status_label,
            self.input_device_label,
            self.input_device_combo,
            self.ptt_button,
            self.cancel_voice_button,
            self.voice_hint_label,
            self.transcript_label,
            self.question_label,
            self.voice_metrics_label,
            self.item_frame,
            self.advice_caption,
            self.reason_label,
            self.source_label,
            self.metrics_label,
            self.rag_caption,
            self.rag_scroll,
            self.budget_frame,
            self.event_caption,
            self.event_scroll,
        ]

        self.setStyleSheet(_STYLE)

    @Slot()
    def _toggle_compact_mode(self) -> None:
        self._compact_mode = not self._compact_mode
        if self._compact_mode:
            self._normal_size = self.size()
        for widget in self._compact_hidden_widgets:
            widget.setVisible(not self._compact_mode)
        self.compact_button.setText("□" if self._compact_mode else "—")
        self.compact_button.setToolTip(
            "恢复完整界面" if self._compact_mode else "精简模式：只显示输入框和回答框"
        )
        self.main_scroll.verticalScrollBar().setValue(0)
        if self._compact_mode:
            self.resize(max(440, min(self.width(), 620)), 320)
            self.question_input.setFocus()
        elif self._normal_size is not None:
            self.resize(self._normal_size)

    def _card(self, caption: str, value: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("card")
        frame.setMinimumWidth(195)
        layout = QVBoxLayout(frame)
        title = QLabel(caption)
        title.setObjectName("caption")
        label = QLabel(value)
        label.setObjectName("value")
        label.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(label)
        return frame, label

    @Slot()
    def _poll_log(self) -> None:
        try:
            poll = self.tailer.poll()
        except OSError:
            self.connection_label.setText("日志不可读")
            return
        if poll.reopened:
            self.store.diagnostics.log_reopens += 1
        for line in poll.lines:
            self._handle_line(line)
        self._update_connection()

    def _handle_line(self, line: str) -> None:
        try:
            event = parse_event_line(line)
        except EventParseError:
            self.store.mark_invalid()
            return
        if event is None:
            self.store.mark_ignored()
            return
        previous_room = (
            self.store.state.context.get("room_index"),
            self.store.state.context.get("room_spawn_seed"),
        )
        try:
            self.store.apply(event)
        except EventOrderError:
            return
        self._last_event_at = time.monotonic()
        self.budget.set_run(self.store.state.run_id)
        self._recent.appendleft(f"#{event.seq}  {event.type}")
        self.events_label.setText("\n".join(self._recent))
        self._update_state(event)

        current_room = (
            self.store.state.context.get("room_index"),
            self.store.state.context.get("room_spawn_seed"),
        )
        if current_room != previous_room and event.type != "collectible_spawned":
            self._cancel_pending()
            if self.voice_service is not None:
                self.voice_service.room_changed()
        if event.type == "collectible_spawned":
            self._handle_collectible(event)

    def _update_state(self, event: GameEvent) -> None:
        context = self.store.state.context
        room_type = context.get("room_type")
        room_name = ROOM_NAMES.get(room_type, f"房间类型 {room_type}" if room_type else "未知房间")
        self.room_label.setText(
            f"{room_name}\n楼层 {context.get('stage', '—')} · 房间 {context.get('room_index', '—')}"
        )
        players = self.store.state.players
        if players:
            player = players[sorted(players)[0]]
            resources = player.get("resources", {})
            health = player.get("health", {})
            self.resource_label.setText(
                f"红心 {_half_hearts(health.get('red_hearts'))} · "
                f"魂心 {_half_hearts(health.get('soul_hearts'))}\n"
                f"硬币 {resources.get('coins', '—')} · 钥匙 {resources.get('keys', '—')} · "
                f"炸弹 {resources.get('bombs', '—')}"
            )
        if event.type == "run_ended":
            self.connection_label.setText("本局已结束")

    def _handle_collectible(self, event: GameEvent) -> None:
        collectible_id = event.payload.get("collectible_id")
        if type(collectible_id) is not int:
            return
        descriptor = self.advice_engine.item_descriptor(collectible_id)
        if descriptor is None:
            self.item_label.setText(f"未知道具 ID {collectible_id}\n本地资料暂未覆盖，未调用模型。")
            self.rag_debug_label.setText("资料不足：实体词典与本地索引均未命中。")
            return
        name_zh, name_en = descriptor
        self.item_label.setText(f"{name_zh} / {name_en}（ID {collectible_id}）")
        if not self.advice_engine.supports(event):
            return
        self._cancel_pending()
        if self.voice_service is not None:
            self.voice_service.cancel()
        self._cancel = Event()
        self.connection_label.setText("正在生成建议")
        self._future = self._executor.submit(self.advice_engine.generate, event, self._cancel)
        self._future.add_done_callback(self._on_advice_done)

    def _on_advice_done(self, future: Future[tuple[AdviceResponse, StateToken]]) -> None:
        try:
            response, token = future.result()
        except ModelCancelled:
            return
        except Exception:
            self._signals.failed.emit("建议生成失败，已安全忽略本次结果。")
            return
        self._signals.completed.emit(response, token)

    @Slot(object, object)
    def _show_advice(self, response: AdviceResponse, token: StateToken) -> None:
        if not token.is_current(self.store.state):
            self.connection_label.setText("过期建议已丢弃")
            return
        self.connection_label.setText("建议已就绪")
        self.advice_label.setText(response.advice)
        reason = "理由：" + response.reason
        if response.delivery_note:
            reason += "\n" + response.delivery_note
        self.reason_label.setText(reason)
        self.source_label.setText(_format_source_links(response.sources))
        if response.rag_hits:
            debug_lines = []
            for hit in response.rag_hits:
                methods = "+".join(hit.methods)
                scores = ", ".join(
                    f"{name}={value:.3f}" for name, value in sorted(hit.scores.items())
                )
                debug_lines.append(
                    f"{hit.chunk.entity_type}:{hit.chunk.entity_id} · {methods} · "
                    f"总分 {hit.score:.3f}（{scores}）· {hit.chunk.source.title}"
                )
            suffix = " · 关键词降级" if response.retrieval_degraded else " · 混合检索"
            status = (
                f"；{response.retrieval_degradation_reason}"
                if response.retrieval_degradation_reason
                else ""
            )
            debug_lines.append(
                f"语料 {response.retrieval_corpus_version} · "
                f"延迟 {response.retrieval_latency_ms:.1f} ms{suffix}{status}"
            )
            self.rag_debug_label.setText("\n".join(debug_lines))
        else:
            self.rag_debug_label.setText("阶段 1 固定资料回退；没有可显示的 RAG 命中。")
        mode = "模拟" if response.simulated else "在线"
        self.metrics_label.setText(
            f"置信度 {response.confidence:.0%} · 状态序号 {response.state_seq} · {mode}"
        )
        cost = response.cost
        self.cost_label.setText(
            f"本次 ¥{cost.estimated_cost_cny:.6f} · 本局 ¥{cost.run_total_cny:.6f} · "
            f"输入 {cost.input_tokens} · 输出 {cost.output_tokens} · {cost.model}"
        )
        progress = min(1.0, cost.run_total_cny / self.budget.run_limit_cny)
        self.budget_bar.setValue(round(progress * 1000))
        if self.voice_enabled.isChecked() and self.voice_service is not None:
            self.voice_service.speak_validated(response.advice)

    def _populate_audio_devices(self) -> None:
        self.input_device_combo.clear()
        if self.voice_service is None:
            self.input_device_combo.addItem("无可用设备", None)
            self.input_device_combo.setEnabled(False)
            self.ptt_button.setEnabled(False)
            return
        devices = self.voice_service.devices()
        if not devices:
            self.input_device_combo.addItem("未找到麦克风", None)
            self.ptt_button.setEnabled(False)
            return
        for device_id, name in devices:
            self.input_device_combo.addItem(name, device_id)
        self.ptt_button.setEnabled(self.voice_enabled.isChecked())

    @Slot(bool)
    def _voice_enabled_changed(self, enabled: bool) -> None:
        if not enabled and self.voice_service is not None:
            self.voice_service.cancel()
        self.ptt_button.setEnabled(
            enabled and self.voice_service is not None and self.input_device_combo.count() > 0
            and self.input_device_combo.currentData() is not None
        )

    @Slot()
    def _voice_press(self) -> None:
        if not self.voice_enabled.isChecked() or self.voice_service is None:
            self._show_voice_error("", "语音功能未启用或当前离线不可用。")
            return
        self._cancel_pending()
        self.voice_service.press(self.input_device_combo.currentData())

    @Slot()
    def _voice_release(self) -> None:
        if self.voice_service is not None:
            self.voice_service.release()

    @Slot()
    def _voice_cancel(self) -> None:
        if self.voice_service is not None:
            self.voice_service.cancel()

    @Slot()
    def _submit_text_question(self) -> None:
        text = self.question_input.text().strip()
        if not text:
            return
        if self.voice_service is None:
            self._show_voice_error("", "文本问答服务不可用。")
            return
        self._cancel_pending()
        self.question_input.clear()
        self.voice_service.ask_text(text, speak=self.voice_enabled.isChecked())

    @Slot(str, object)
    def _show_voice_state(self, request_id: str, state: VoiceState) -> None:
        if request_id and self.voice_service is not None and not self.voice_service.is_current(request_id) and state != VoiceState.CANCELLED:
            return
        self.voice_status_label.setText(state.value)

    @Slot(str, object)
    def _show_transcript(self, request_id: str, transcript: Transcript) -> None:
        if self.voice_service is not None and not self.voice_service.is_current(request_id):
            return
        prefix = "最终字幕" if transcript.final else "当前字幕"
        self.transcript_label.setText(f"{prefix}：{transcript.text}")

    @Slot(str, str)
    def _show_question(self, request_id: str, question: str) -> None:
        if self.voice_service is not None and not self.voice_service.is_current(request_id):
            return
        self.question_label.setText("最终问题：" + question)

    @Slot(str, object, object)
    def _show_query_answer(self, request_id: str, response: QueryResponse, token: QueryToken) -> None:
        if self.voice_service is None or not self.voice_service.is_current(request_id):
            return
        if not token.is_current(self.store.state, request_id):
            self.voice_service.cancel()
            return
        self.advice_label.setText(response.answer)
        note = response.delivery_note or "回答通过本地证据和结构校验。"
        self.reason_label.setText(note)
        self.source_label.setText(_format_source_links(response.sources))
        self.metrics_label.setText(
            f"置信度 {response.confidence:.0%} · 状态序号 {response.state_seq} · "
            f"{'离线摘要' if response.simulated else '在线模型'}"
        )
        self.rag_debug_label.setText(self._format_rag_debug(response))
        self.cost_label.setText(
            f"本次 ¥{response.cost.estimated_cost_cny:.6f} · 本局 ¥{response.cost.run_total_cny:.6f} · "
            f"输入 {response.cost.input_tokens} · 输出 {response.cost.output_tokens} · {response.cost.model}"
        )
        if response.retrieval_degraded:
            self.voice_hint_label.setText("已发生离线降级：" + (response.retrieval_degradation_reason or "关键词检索"))
        else:
            self.voice_hint_label.setText("本次问答使用本地混合检索，未联网搜索。")

    @Slot(str, str)
    def _show_voice_error(self, request_id: str, message: str) -> None:
        if request_id and self.voice_service is not None and self.voice_service.is_current(request_id):
            pass
        self.voice_hint_label.setText(message)
        if "仍可" not in message:
            self.voice_status_label.setText("已取消")

    @Slot(str, object)
    def _show_voice_metrics(self, request_id: str, metrics: VoiceMetrics) -> None:
        self.voice_metrics_label.setText(
            f"采集 {metrics.capture_start_ms:.1f} ms · ASR 中间 {metrics.asr_first_partial_ms:.1f} ms · "
            f"ASR 最终 {metrics.asr_final_ms:.1f} ms\nRAG {metrics.rag_ms:.1f} ms · "
            f"模型 {metrics.model_first_text_ms:.1f} ms · TTS 首音频 {metrics.tts_first_audio_ms:.1f} ms · "
            f"端到端 {metrics.first_audio_end_to_end_ms:.1f} ms · 打断 {metrics.interrupt_ms:.1f} ms · "
            f"队列峰值 {metrics.queue_peak}"
        )

    @staticmethod
    def _format_rag_debug(response: QueryResponse) -> str:
        lines = [
            f"{hit.chunk.entity_type}:{hit.chunk.entity_id} · {'+'.join(hit.methods)} · "
            f"总分 {hit.score:.3f} · {hit.chunk.source.title}"
            for hit in response.rag_hits
        ]
        mode = "关键词降级" if response.retrieval_degraded else "混合检索"
        lines.append(
            f"语料 {response.retrieval_corpus_version} · 延迟 {response.retrieval_latency_ms:.1f} ms · {mode}"
        )
        return "\n".join(lines)

    @Slot(str)
    def _show_model_error(self, message: str) -> None:
        self.connection_label.setText("建议不可用")
        self.advice_label.setText(message)

    def _cancel_pending(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
        if self._future is not None:
            self._future.cancel()
        self._cancel = None
        self._future = None

    def _update_connection(self) -> None:
        if self._last_event_at is None:
            self.connection_label.setText("等待游戏事件")
            return
        if time.monotonic() - self._last_event_at > 3:
            self.connection_label.setText("日志已连接，等待新事件")
        elif self.connection_label.text() not in {"正在生成建议", "建议已就绪"}:
            self.connection_label.setText("游戏已连接")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.timer.stop()
        self._cancel_pending()
        self.tailer.close()
        if self.voice_service is not None:
            self.voice_service.close()
        if self.advice_engine.rag is not None:
            self.advice_engine.rag.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if not event.isAutoRepeat() and self._matches_ptt_key(event):
            self._voice_press()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if not event.isAutoRepeat() and self._matches_ptt_key(event):
            self._voice_release()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _matches_ptt_key(self, event: QKeyEvent) -> bool:
        configured = self.config.voice.push_to_talk_key.strip().casefold()
        keys = {
            "space": Qt.Key.Key_Space,
            "f8": Qt.Key.Key_F8,
            "f9": Qt.Key.Key_F9,
            "f10": Qt.Key.Key_F10,
            "f11": Qt.Key.Key_F11,
            "f12": Qt.Key.Key_F12,
        }
        return configured in keys and event.key() == keys[configured]


def run_overlay(
    *,
    config: OriensConfig,
    log_path: Path,
    knowledge: LocalItemKnowledgeBase,
    advice_engine: AdviceEngine,
    budget: BudgetTracker,
    from_start: bool,
    online_requested: bool,
    api_key_available: bool,
    query_engine: QueryEngine | None = None,
    asr: RealtimeASR | None = None,
    tts: StreamingTTS | None = None,
) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Oriens")
    app.setApplicationDisplayName("Oriens：你的游戏向导")
    window = OverlayWindow(
        config=config,
        log_path=log_path,
        knowledge=knowledge,
        advice_engine=advice_engine,
        budget=budget,
        from_start=from_start,
        online_requested=online_requested,
        api_key_available=api_key_available,
        query_engine=query_engine,
        asr=asr,
        tts=tts,
    )
    window.show()
    return app.exec()


def _half_hearts(value: Any) -> str:
    if type(value) is not int:
        return "—"
    return f"{value / 2:g}"


def _format_source_links(sources: Any) -> str:
    links = " · ".join(
        '<a style="color:#b9e5ff;text-decoration:underline;" '
        f'href="{escape(source.url, quote=True)}">{escape(source.title)}</a>'
        for source in sources
    )
    return '<span style="color:#c9eaff;">来源：' + links + "</span>"


_STYLE = """
QFrame#shell {
    background: rgba(18, 21, 28, 242);
    border: 1px solid rgba(118, 151, 198, 110);
    border-radius: 16px;
    color: #edf3ff;
}
QLabel { color: #edf3ff; font-family: "Microsoft YaHei UI"; }
QLabel#title { font-size: 18px; font-weight: 800; letter-spacing: 3px; color: #9fc8ff; }
QLabel#status { background: #284661; color: #d8edff; padding: 5px 9px; border-radius: 9px; }
QLabel#hint { color: #91a3b8; font-size: 11px; }
QFrame#card { background: rgba(37, 43, 55, 210); border-radius: 10px; }
QFrame#adviceCard { background: rgba(26, 53, 70, 230); border: 1px solid #356d8f; border-radius: 12px; }
QLabel#caption { color: #8fa7c1; font-size: 11px; font-weight: 700; }
QLabel#value { color: #f1f5fb; font-size: 13px; }
QLabel#advice { color: #ffffff; font-size: 17px; font-weight: 750; }
QLabel#reason { color: #c4d7e7; font-size: 12px; }
QLabel#source { color: #8fc8ff; font-size: 11px; }
QScrollArea#mainScroll { background: transparent; border: 0; }
QScrollArea#mainScroll > QWidget > QWidget { background: transparent; }
QLabel#metrics { color: #83b6a4; font-size: 11px; }
QLabel#events { color: #aebccc; font-family: Consolas, "Microsoft YaHei UI"; font-size: 11px; padding: 7px; }
QPushButton#close { background: transparent; color: #aebccc; border: 0; font-size: 20px; width: 26px; }
QPushButton#close:hover { color: #ffffff; background: #7b3642; border-radius: 8px; }
QPushButton#windowControl { background: transparent; color: #c8d5e5; border: 0; font-size: 18px; width: 26px; padding: 0; }
QPushButton#windowControl:hover { color: #ffffff; background: #34465c; border-radius: 8px; }
QPushButton#primary { background: #3b789e; color: white; border: 0; border-radius: 7px; padding: 8px; font-weight: 700; }
QPushButton#primary:pressed { background: #28546f; }
QPushButton { background: #303b4b; color: #edf3ff; border: 0; border-radius: 7px; padding: 7px; }
QComboBox, QLineEdit { background: #202936; color: #edf3ff; border: 1px solid #46566c; border-radius: 6px; padding: 6px; }
QComboBox QAbstractItemView { background: #202936; color: #ffffff; border: 1px solid #5b708c; selection-background-color: #3b789e; selection-color: #ffffff; outline: 0; }
QComboBox QAbstractItemView::item { min-height: 32px; color: #ffffff; padding: 4px 8px; }
QComboBox QAbstractItemView::item:hover { background: #304d67; color: #ffffff; }
QComboBox QAbstractItemView::item:selected { background: #3b789e; color: #ffffff; }
QCheckBox { color: #edf3ff; }
QScrollArea#eventScroll { background: rgba(20, 24, 31, 180); border: 0; border-radius: 8px; }
QScrollArea#eventScroll > QWidget > QWidget { background: transparent; }
QProgressBar { background: #202936; border: 0; height: 5px; border-radius: 2px; }
QProgressBar::chunk { background: #4ea68a; border-radius: 2px; }
"""
