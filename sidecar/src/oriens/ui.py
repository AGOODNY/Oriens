"""PySide6 最小悬浮窗。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
import time
from typing import Any

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .advice import AdviceEngine, AdviceResponse, StateToken
from .budget import BudgetTracker
from .config import OriensConfig
from .knowledge import LocalItemKnowledgeBase
from .modeling import ModelCancelled
from .protocol import EventParseError, GameEvent, parse_event_line
from .state import EventOrderError, StateStore
from .tailer import LogTailer


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
        self._drag_origin: QPoint | None = None

        self.setWindowTitle("Oriens：你的游戏向导")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(440)
        self.resize(470, 680)
        self._build_ui()

        if online_requested and api_key_available:
            self.mode_label.setText("在线建议已启用")
        elif online_requested:
            self.mode_label.setText("未找到 DASHSCOPE_API_KEY，已进入离线模拟模式")
        else:
            self.mode_label.setText("离线模拟模式（启动时添加 --online 可启用百炼）")

        self.timer = QTimer(self)
        self.timer.setInterval(config.app.poll_interval_ms)
        self.timer.timeout.connect(self._poll_log)
        self.timer.start()

    def _build_ui(self) -> None:
        shell = QFrame()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("ORIENS")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self.connection_label = QLabel("等待日志")
        self.connection_label.setObjectName("status")
        header.addWidget(self.connection_label)
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
        room_card, self.room_label = self._card("当前房间", "尚未进入一局游戏")
        resource_card, self.resource_label = self._card(
            "角色资源", "红心 — · 魂心 —\n硬币 — · 钥匙 — · 炸弹 —"
        )
        state_row.addWidget(room_card)
        state_row.addWidget(resource_card)
        layout.addLayout(state_row)

        item_frame = QFrame()
        item_frame.setObjectName("card")
        item_layout = QVBoxLayout(item_frame)
        item_title = QLabel("识别到的道具")
        item_title.setObjectName("caption")
        self.item_label = QLabel("等待道具房事件")
        self.item_label.setObjectName("value")
        self.item_label.setWordWrap(True)
        item_layout.addWidget(item_title)
        item_layout.addWidget(self.item_label)
        layout.addWidget(item_frame)

        advice_frame = QFrame()
        advice_frame.setObjectName("adviceCard")
        advice_layout = QVBoxLayout(advice_frame)
        advice_caption = QLabel("短建议")
        advice_caption.setObjectName("caption")
        self.advice_label = QLabel("进入已覆盖的道具房后，这里会显示建议。")
        self.advice_label.setObjectName("advice")
        self.advice_label.setWordWrap(True)
        self.reason_label = QLabel("")
        self.reason_label.setObjectName("reason")
        self.reason_label.setWordWrap(True)
        self.source_label = QLabel("")
        self.source_label.setObjectName("source")
        self.source_label.setWordWrap(True)
        self.source_label.setOpenExternalLinks(True)
        self.metrics_label = QLabel("置信度 — · 状态序号 —")
        self.metrics_label.setObjectName("metrics")
        advice_layout.addWidget(advice_caption)
        advice_layout.addWidget(self.advice_label)
        advice_layout.addWidget(self.reason_label)
        advice_layout.addWidget(self.source_label)
        advice_layout.addWidget(self.metrics_label)
        layout.addWidget(advice_frame)

        rag_caption = QLabel("检索调试")
        rag_caption.setObjectName("caption")
        layout.addWidget(rag_caption)
        rag_scroll = QScrollArea()
        rag_scroll.setObjectName("eventScroll")
        rag_scroll.setWidgetResizable(True)
        rag_scroll.setMaximumHeight(105)
        self.rag_debug_label = QLabel("尚未执行检索")
        self.rag_debug_label.setObjectName("events")
        self.rag_debug_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rag_debug_label.setWordWrap(True)
        rag_scroll.setWidget(self.rag_debug_label)
        layout.addWidget(rag_scroll)

        budget_frame = QFrame()
        budget_frame.setObjectName("card")
        budget_layout = QVBoxLayout(budget_frame)
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
        layout.addWidget(budget_frame)

        event_caption = QLabel("最近事件")
        event_caption.setObjectName("caption")
        layout.addWidget(event_caption)
        scroll = QScrollArea()
        scroll.setObjectName("eventScroll")
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(125)
        self.events_label = QLabel("暂无事件")
        self.events_label.setObjectName("events")
        self.events_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.events_label.setWordWrap(True)
        scroll.setWidget(self.events_label)
        layout.addWidget(scroll)

        self.setStyleSheet(_STYLE)

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
        links = [
            f'<a href="{source.url}">{source.title}</a>' for source in response.sources
        ]
        self.source_label.setText("来源：" + " · ".join(links))
        if response.rag_hits:
            debug_lines = []
            for hit in response.rag_hits:
                methods = "+".join(hit.methods)
                debug_lines.append(
                    f"{hit.chunk.entity_type}:{hit.chunk.entity_id} · {methods} · "
                    f"{hit.score:.3f} · {hit.chunk.source.title}"
                )
            suffix = " · 关键词降级" if response.retrieval_degraded else " · 混合检索"
            debug_lines.append(f"延迟 {response.retrieval_latency_ms:.1f} ms{suffix}")
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
    )
    window.show()
    return app.exec()


def _half_hearts(value: Any) -> str:
    if type(value) is not int:
        return "—"
    return f"{value / 2:g}"


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
QLabel#source a { color: #8fc8ff; }
QLabel#metrics { color: #83b6a4; font-size: 11px; }
QLabel#events { color: #aebccc; font-family: Consolas, "Microsoft YaHei UI"; font-size: 11px; padding: 7px; }
QPushButton#close { background: transparent; color: #aebccc; border: 0; font-size: 20px; width: 26px; }
QPushButton#close:hover { color: #ffffff; background: #7b3642; border-radius: 8px; }
QScrollArea#eventScroll { background: rgba(20, 24, 31, 180); border: 0; border-radius: 8px; }
QScrollArea#eventScroll > QWidget > QWidget { background: transparent; }
QProgressBar { background: #202936; border: 0; height: 5px; border-radius: 2px; }
QProgressBar::chunk { background: #4ea68a; border-radius: 2px; }
"""
