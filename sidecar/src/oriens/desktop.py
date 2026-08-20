"""阶段 3.6 Windows 桌面产品外壳。"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QAbstractItemView,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .application import ListeningState, OriensApplication
from .config import ConfigError, ConfigService
from .knowledge_pack import KnowledgePackError, KnowledgePackManager
from .memory import MEMORY_KINDS
from .theme import (
    BookTitleBar,
    FramelessBookWindow,
    StatusCard,
    application_icon,
    apply_application_theme,
)
from .ui import OverlayWindow


_MEMORY_KIND_LABELS = {
    "profile": "玩家档案",
    "stable_preference": "稳定偏好",
    "guidance_preference": "提示偏好",
    "milestone": "对局里程碑",
}

_MEMORY_STATUS_LABELS = {
    "active": "已启用",
    "disabled": "已禁用",
    "pending": "待确认",
    "conflicted": "已被纠正",
    "deleted": "已删除",
}


class SettingsDialog(QDialog):
    """仅保存阶段 3.5 已允许的非敏感设置。"""

    def __init__(
        self,
        application: OriensApplication,
        parent: QWidget | None = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.application = application
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setObjectName("embeddedPanel")
        self.setWindowTitle("Oriens 设置")
        if not embedded:
            self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.voice_enabled = QCheckBox("启用语音")
        self.voice_enabled.setChecked(application.config.voice.enabled)
        self.ptt_key = QComboBox()
        self.ptt_key.addItems(("Space", "F8", "F9", "F10", "F11", "F12"))
        self.ptt_key.setCurrentText(application.config.voice.push_to_talk_key)
        self.tts_volume = QSpinBox()
        self.tts_volume.setRange(0, 100)
        self.tts_volume.setValue(application.config.voice.tts_volume)
        self.tts_rate = QDoubleSpinBox()
        self.tts_rate.setRange(0.5, 2.0)
        self.tts_rate.setSingleStep(0.1)
        self.tts_rate.setValue(application.config.voice.tts_rate)
        self.tts_voice = QComboBox()
        self.tts_voice.setEditable(False)
        self.tts_voice.addItem(application.config.voice.tts_voice)
        self.knowledge_pack = QComboBox()
        manager = KnowledgePackManager(application.paths.knowledge_dir)
        packs = manager.enumerate_installed()
        for pack in packs:
            capability = "完整" if "vector" in pack.capabilities else "轻量"
            self.knowledge_pack.addItem(
                f"{pack.manifest.display_name} · {pack.manifest.content_version} · {capability}",
                pack.manifest.pack_id,
            )
        if application.knowledge_pack is None:
            self.knowledge_pack.addItem("开发显式配置（由启动命令决定）", None)
            self.knowledge_pack.setCurrentIndex(self.knowledge_pack.count() - 1)
            self.knowledge_pack.setEnabled(False)
        else:
            index = self.knowledge_pack.findData(application.knowledge_pack.manifest.pack_id)
            if index >= 0:
                self.knowledge_pack.setCurrentIndex(index)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(0.01, 100.0)
        self.budget.setDecimals(2)
        self.budget.setValue(application.config.budget.run_limit_cny)
        self.memory_enabled = QCheckBox("启用本地长期记忆")
        self.memory_enabled.setChecked(application.config.memory.enabled)
        self.vision_enabled = QCheckBox("启用按需视觉补充")
        self.vision_enabled.setChecked(application.config.vision.enabled)
        self.vision_debug_save = QCheckBox("保存调试截图（仅用于主动排障）")
        self.vision_debug_save.setChecked(
            application.config.vision.debug_save_screenshots
        )
        self.realtime_enabled = QCheckBox("启用实时语音实验")
        self.realtime_enabled.setChecked(application.config.realtime.enabled)
        self.realtime_vad = QCheckBox("启用 semantic_vad（实验）")
        self.realtime_vad.setChecked(
            application.config.realtime.semantic_vad_enabled
        )
        self.realtime_debug_audio = QCheckBox("保存 Realtime 调试音频")
        self.realtime_debug_audio.setChecked(
            application.config.realtime.debug_save_audio
        )
        form.addRow("语音", self.voice_enabled)
        form.addRow("按键说话键", self.ptt_key)
        form.addRow("TTS 音量", self.tts_volume)
        form.addRow("TTS 语速", self.tts_rate)
        form.addRow("TTS 音色", self.tts_voice)
        form.addRow("本局预算上限（元）", self.budget)
        form.addRow("长期记忆", self.memory_enabled)
        form.addRow("视觉补充", self.vision_enabled)
        form.addRow("视觉调试", self.vision_debug_save)
        form.addRow("语音模式", self.realtime_enabled)
        form.addRow("语义打断", self.realtime_vad)
        form.addRow("音频调试", self.realtime_debug_audio)
        form.addRow("当前知识包", self.knowledge_pack)
        layout.addLayout(form)
        note = QLabel(
            "保存后将在下次启动 Oriens 时生效；开发命令的显式配置可能覆盖这些值。"
            "长期记忆开关也在下次启动时生效；关闭不会自动删除已有数据。"
            "视觉只捕获已识别的游戏窗口，不捕获整个桌面；默认不保存截图。"
            "视觉与调试截图开关均在下次启动时生效。"
            "实时语音默认关闭；启用后只发送用户主动提交的语音至百炼，默认不保存音频。"
            "semantic_vad 与调试音频也都默认关闭并在下次启动生效。"
            "API Key 和业务空间 ID 不会写入此设置。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        if embedded:
            cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
            if cancel is not None:
                cancel.hide()
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot()
    def _save(self) -> None:
        overrides = {
            "voice": {
                "enabled": self.voice_enabled.isChecked(),
                "push_to_talk_key": self.ptt_key.currentText(),
                "tts_voice": self.tts_voice.currentText(),
                "tts_rate": self.tts_rate.value(),
                "tts_volume": self.tts_volume.value(),
            },
            "budget": {"run_limit_cny": self.budget.value()},
            "memory": {"enabled": self.memory_enabled.isChecked()},
            "vision": {
                "enabled": self.vision_enabled.isChecked(),
                "debug_save_screenshots": self.vision_debug_save.isChecked(),
            },
            "realtime": {
                "enabled": self.realtime_enabled.isChecked(),
                "semantic_vad_enabled": self.realtime_vad.isChecked(),
                "debug_save_audio": self.realtime_debug_audio.isChecked(),
            },
        }
        try:
            ConfigService(self.application.paths).update_user_overrides(overrides)
            selected_pack = self.knowledge_pack.currentData()
            if selected_pack is not None:
                KnowledgePackManager(self.application.paths.knowledge_dir).select(selected_pack)
        except (ConfigError, KnowledgePackError, OSError):
            QMessageBox.warning(self, "设置未保存", "设置文件暂时无法安全保存，请稍后重试。")
            return
        QMessageBox.information(self, "设置已保存", "设置将在下次启动 Oriens 时生效。")
        if not self.embedded:
            self.accept()


class MemoryManagementDialog(QDialog):
    """通过应用命令管理本机长期记忆，不接触 SQLite 或绝对路径。"""

    def __init__(
        self,
        application: OriensApplication,
        parent: QWidget | None = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.application = application
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setObjectName("embeddedPanel")
        self.setWindowTitle("Oriens 长期记忆管理")
        if not embedded:
            self.resize(860, 560)
        else:
            self.setMinimumHeight(430)
        layout = QVBoxLayout(self)
        self.service_status = QLabel()
        self.service_status.setWordWrap(True)
        layout.addWidget(self.service_status)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("类型", "内容", "来源", "创建/更新时间", "状态"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._load_selection)
        layout.addWidget(self.table)

        editor = QFormLayout()
        self.kind = QComboBox()
        for value in sorted(MEMORY_KINDS):
            self.kind.addItem(_MEMORY_KIND_LABELS[value], value)
        self.content = QLineEdit()
        self.content.setMaxLength(240)
        self.content.setPlaceholderText("只添加明确、稳定且适合长期保存的信息")
        editor.addRow("记忆类型", self.kind)
        editor.addRow("记忆内容", self.content)
        layout.addLayout(editor)

        actions = QHBoxLayout()
        self.add_button = QPushButton("手动添加")
        self.update_button = QPushButton("保存纠正")
        self.toggle_button = QPushButton("启用/禁用选中项")
        self.delete_button = QPushButton("删除选中项")
        self.clear_button = QPushButton("清空全部长期记忆")
        self.close_button = QPushButton("关闭")
        self.close_button.setVisible(not embedded)
        for button in (
            self.add_button, self.update_button, self.toggle_button,
            self.delete_button, self.clear_button, self.close_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        note = QLabel(
            "长期记忆只保存在本机。关闭总开关只会停止写入和召回，不会自动删除已有数据；"
            "删除单条或清空全部数据后无法在 Oriens 中恢复。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.add_button.clicked.connect(self._add)
        self.update_button.clicked.connect(self._update)
        self.toggle_button.clicked.connect(self._toggle)
        self.delete_button.clicked.connect(self._delete)
        self.clear_button.clicked.connect(self._clear)
        self.close_button.clicked.connect(self.accept)
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        try:
            items = self.application.list_memories()
            available = self.application.memory.enabled
        except Exception:
            items = ()
            available = False
            self.service_status.setText("本地长期记忆暂时不可用，当前已安全切换为无记忆模式。")
        else:
            self.service_status.setText(
                "长期记忆已启用，数据仅保存在本机。"
                if available else
                "长期记忆当前已关闭。请在设置中启用并重新启动后再管理；已有数据不会被自动删除。"
            )
        self.table.setRowCount(0)
        for item in items:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (
                _MEMORY_KIND_LABELS.get(item.kind, item.kind),
                item.content,
                item.source_summary,
                _display_memory_time(item.created_at, item.updated_at),
                _MEMORY_STATUS_LABELS.get(item.status, item.status),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item.id)
                    cell.setData(Qt.ItemDataRole.UserRole + 1, item.kind)
                    cell.setData(Qt.ItemDataRole.UserRole + 2, item.status)
                self.table.setItem(row, column, cell)
        for button in (
            self.add_button, self.update_button, self.toggle_button,
            self.delete_button, self.clear_button,
        ):
            button.setEnabled(available)

    def _selected(self) -> tuple[str, str, str] | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        cell = self.table.item(row, 0)
        content = self.table.item(row, 1)
        if cell is None or content is None:
            return None
        return (
            str(cell.data(Qt.ItemDataRole.UserRole)),
            str(cell.data(Qt.ItemDataRole.UserRole + 1)),
            str(cell.data(Qt.ItemDataRole.UserRole + 2)),
        )

    @Slot()
    def _load_selection(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        _memory_id, kind, _status = selected
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        item = self.table.item(self.table.currentRow(), 1)
        if item is not None:
            self.content.setText(item.text())

    @Slot()
    def _add(self) -> None:
        try:
            self.application.add_memory(str(self.kind.currentData()), self.content.text())
        except Exception:
            self._safe_error("这条内容无法保存为长期记忆，请检查类型和内容后重试。")
            return
        self.content.clear()
        self.refresh()

    @Slot()
    def _update(self) -> None:
        selected = self._selected()
        if selected is None:
            self._safe_error("请先选择要纠正的记忆。")
            return
        try:
            self.application.update_memory(
                selected[0], str(self.kind.currentData()), self.content.text()
            )
        except Exception:
            self._safe_error("这条记忆暂时无法纠正，请稍后重试。")
            return
        self.refresh()

    @Slot()
    def _toggle(self) -> None:
        selected = self._selected()
        if selected is None:
            self._safe_error("请先选择要启用或禁用的记忆。")
            return
        try:
            self.application.set_memory_item_enabled(selected[0], selected[2] != "active")
        except Exception:
            self._safe_error("这条记忆的状态暂时无法更改，请稍后重试。")
            return
        self.refresh()

    @Slot()
    def _delete(self) -> None:
        selected = self._selected()
        if selected is None:
            self._safe_error("请先选择要删除的记忆。")
            return
        answer = QMessageBox.question(
            self, "确认删除长期记忆",
            "确定删除选中的长期记忆吗？删除后无法在 Oriens 中恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.application.delete_memory(selected[0])
        except Exception:
            self._safe_error("这条记忆暂时无法删除，请稍后重试。")
            return
        self.refresh()

    @Slot()
    def _clear(self) -> None:
        answer = QMessageBox.question(
            self, "确认清空全部长期记忆",
            "确定清空全部长期记忆吗？此操作不会关闭功能，但数据无法在 Oriens 中恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.application.clear_memories()
        except Exception:
            self._safe_error("长期记忆暂时无法清空，请稍后重试。")
            return
        self.refresh()

    def _safe_error(self, message: str) -> None:
        QMessageBox.warning(self, "长期记忆操作未完成", message)


class ControlCenterWindow(FramelessBookWindow):
    def __init__(self, controller: "DesktopController") -> None:
        super().__init__()
        self.controller = controller
        self._allow_close = False
        self.setWindowTitle("Oriens 控制中心")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)

        shell = QFrame()
        shell.setObjectName("bookShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(
            BookTitleBar(
                self,
                application_icon(controller.application.paths),
                "桌面伴侣",
            )
        )

        body = QFrame()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        shell_layout.addWidget(body, 1)

        spine = QFrame()
        spine.setObjectName("bookSpine")
        spine.setFixedWidth(210)
        spine_layout = QVBoxLayout(spine)
        spine_layout.setContentsMargins(0, 28, 0, 20)
        spine_layout.setSpacing(4)
        ornament = QLabel("◇  ·  •  ·  ◇")
        ornament.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ornament.setStyleSheet("color:#c7a052; padding:8px;")
        spine_layout.addWidget(ornament)
        self.chapter_group = QButtonGroup(self)
        self.chapter_group.setExclusive(True)
        self.chapter_buttons: list[QPushButton] = []
        for index, caption in enumerate(("总览", "对局", "语音", "知识", "记忆", "设置")):
            button = QPushButton(caption)
            button.setObjectName("chapterButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, page=index: self._set_chapter(page))
            self.chapter_group.addButton(button, index)
            self.chapter_buttons.append(button)
            spine_layout.addWidget(button)
        spine_layout.addStretch(1)
        spine_note = QLabel("•  微光守望中\n\n关闭窗口将缩到系统托盘")
        spine_note.setObjectName("spineNote")
        spine_note.setWordWrap(True)
        spine_layout.addWidget(spine_note)
        body_layout.addWidget(spine)

        self.chapter_stack = QStackedWidget()
        self.chapter_stack.setObjectName("chapterStack")
        body_layout.addWidget(self.chapter_stack, 1)

        self.status_labels: dict[str, QLabel] = {}

        overview, overview_layout = self._new_page("总览", "你的游戏向导正在守望本局")
        self.overall_banner = QLabel("一切就绪")
        self.overall_banner.setObjectName("pageStatus")
        overview_layout.addWidget(self.overall_banner)
        card_row = QHBoxLayout()
        for key, caption, glyph in (
            ("overall", "核心运行", "•"),
            ("listening", "监听状态", "◉"),
            ("cost", "费用与预算", "¥"),
        ):
            card = StatusCard(caption, glyph)
            self.status_labels[key] = card.value
            card_row.addWidget(card)
        overview_layout.addLayout(card_row)
        overview_layout.addWidget(self._section_title("快速操作"))
        quick_row = QHBoxLayout()
        for caption, callback in (
            ("暂停监听", controller.pause_listening),
            ("显示悬浮窗", controller.show_overlay),
            ("识别当前画面", controller.identify_current_view),
            ("打开设置", lambda: self._set_chapter(5)),
        ):
            button = QPushButton(caption)
            button.setObjectName("primaryAction" if caption == "显示悬浮窗" else "")
            button.clicked.connect(callback)
            quick_row.addWidget(button)
        overview_layout.addLayout(quick_row)
        summary_panel = QFrame()
        summary_panel.setObjectName("manuscriptPanel")
        summary_form = QFormLayout(summary_panel)
        self.status_labels["mode"] = self._status_value()
        summary_form.addRow("运行模式", self.status_labels["mode"])
        overview_layout.addWidget(summary_panel)
        privacy_note = QLabel("•  只捕获已识别的游戏窗口；默认不保存截图。")
        privacy_note.setObjectName("pageSubtitle")
        privacy_note.setWordWrap(True)
        overview_layout.addWidget(privacy_note)
        overview_layout.addStretch(1)

        game_page, game_layout = self._new_page("对局", "管理监听、悬浮窗与按需视觉补充")
        runtime_group = QGroupBox("本局运行")
        runtime_form = QFormLayout(runtime_group)
        self.status_labels["log"] = self._status_value()
        self.status_labels["vision"] = self._status_value()
        runtime_form.addRow("游戏日志", self.status_labels["log"])
        runtime_form.addRow("视觉补充", self.status_labels["vision"])
        game_layout.addWidget(runtime_group)
        self.start_button = QPushButton("启动监听")
        self.pause_button = QPushButton("暂停监听")
        self.show_overlay_button = QPushButton("显示悬浮窗")
        self.hide_overlay_button = QPushButton("隐藏悬浮窗")
        self.vision_button = QPushButton("识别当前游戏画面")
        self.cancel_vision_button = QPushButton("取消视觉识别")
        session_row = QHBoxLayout()
        for button in (
            self.start_button,
            self.pause_button,
            self.show_overlay_button,
            self.hide_overlay_button,
        ):
            session_row.addWidget(button)
        game_layout.addLayout(session_row)
        vision_group = QGroupBox("按需视觉补充")
        vision_row = QHBoxLayout(vision_group)
        vision_note = QLabel("只捕获已识别的游戏窗口；默认不保存截图。")
        vision_note.setWordWrap(True)
        vision_row.addWidget(vision_note, 1)
        for button in (
            self.vision_button,
            self.cancel_vision_button,
        ):
            vision_row.addWidget(button)
        game_layout.addWidget(vision_group)
        game_layout.addStretch(1)

        voice_page, voice_layout = self._new_page("语音", "链式语音为默认模式，实时实验可按需连接")
        voice_status_group = QGroupBox("语音服务")
        voice_status_form = QFormLayout(voice_status_group)
        self.status_labels["voice"] = self._status_value()
        self.status_labels["realtime"] = self._status_value()
        voice_status_form.addRow("链式语音", self.status_labels["voice"])
        voice_status_form.addRow("实时语音实验", self.status_labels["realtime"])
        voice_layout.addWidget(voice_status_group)
        realtime_group = QGroupBox("实时语音实验")
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_note = QLabel(
            "默认使用链式语音。实时模式仅发送你主动提交的语音至百炼，默认不保存音频；"
            "失败或预算耗尽会自动退回链式语音。"
        )
        realtime_note.setWordWrap(True)
        realtime_layout.addWidget(realtime_note)
        realtime_row = QHBoxLayout()
        self.realtime_connect_button = QPushButton("连接实时语音")
        self.realtime_disconnect_button = QPushButton("断开")
        self.realtime_cancel_button = QPushButton("取消与打断")
        for button in (
            self.realtime_connect_button,
            self.realtime_disconnect_button,
            self.realtime_cancel_button,
        ):
            realtime_row.addWidget(button)
        realtime_layout.addLayout(realtime_row)
        voice_layout.addWidget(realtime_group)
        voice_layout.addStretch(1)

        knowledge_page, knowledge_layout = self._new_page("知识", "本地知识包与检索服务状态")
        knowledge_group = QGroupBox("知识与检索")
        knowledge_form = QFormLayout(knowledge_group)
        for key, caption in (("knowledge", "当前知识包"), ("rag", "RAG"), ("config", "配置来源")):
            self.status_labels[key] = self._status_value()
            knowledge_form.addRow(caption, self.status_labels[key])
        knowledge_layout.addWidget(knowledge_group)
        knowledge_note = QLabel("知识与检索均优先使用本地资料；离线模式下仍可提供已覆盖内容的建议。")
        knowledge_note.setObjectName("pageSubtitle")
        knowledge_note.setWordWrap(True)
        knowledge_layout.addWidget(knowledge_note)
        knowledge_layout.addStretch(1)

        memory_page, memory_layout = self._new_page("记忆", "长期记忆只保存在本机")
        memory_group = QGroupBox("记忆服务")
        memory_form = QFormLayout(memory_group)
        self.status_labels["memory"] = self._status_value()
        memory_form.addRow("当前状态", self.status_labels["memory"])
        memory_layout.addWidget(memory_group)
        self.memory_button = QPushButton("在独立窗口打开记忆管理")
        self.memory_button.setObjectName("primaryAction")
        self.memory_panel = MemoryManagementDialog(
            controller.application,
            memory_page,
            embedded=True,
        )
        memory_layout.addWidget(self.memory_panel)
        memory_layout.addWidget(self.memory_button)
        memory_note = QLabel("关闭总开关不会自动删除已有数据；删除单条或清空全部记忆仍会再次确认。")
        memory_note.setObjectName("pageSubtitle")
        memory_note.setWordWrap(True)
        memory_layout.addWidget(memory_note)
        memory_layout.addStretch(1)

        settings_page, settings_layout = self._new_page("设置", "调整语音、预算、记忆、视觉与实验功能")
        settings_group = QGroupBox("应用设置")
        settings_group_layout = QVBoxLayout(settings_group)
        settings_copy = QLabel(
            "设置保存后在下次启动 Oriens 时生效。API Key、业务空间 ID 和完整端点不会显示或写入这里。"
        )
        settings_copy.setWordWrap(True)
        settings_group_layout.addWidget(settings_copy)
        self.settings_button = QPushButton("在独立窗口打开设置")
        self.settings_button.setObjectName("primaryAction")
        settings_group_layout.addWidget(self.settings_button)
        settings_layout.addWidget(settings_group)
        self.settings_panel = SettingsDialog(
            controller.application,
            settings_page,
            embedded=True,
        )
        settings_layout.addWidget(self.settings_panel)
        self.exit_button = QPushButton("完全退出 Oriens")
        self.exit_button.setObjectName("dangerAction")
        settings_layout.addWidget(self.exit_button)
        self.close_hint = QLabel("关闭此窗口会缩到系统托盘；只有“完全退出”才会停止后台核心。")
        self.close_hint.setWordWrap(True)
        self.close_hint.setObjectName("pageSubtitle")
        settings_layout.addWidget(self.close_hint)
        settings_layout.addStretch(1)

        self.setCentralWidget(shell)
        self.enable_resize_tracking(shell)
        self.chapter_buttons[0].setChecked(True)
        self._set_chapter(0)

        self.start_button.clicked.connect(controller.resume_listening)
        self.pause_button.clicked.connect(controller.pause_listening)
        self.show_overlay_button.clicked.connect(controller.show_overlay)
        self.hide_overlay_button.clicked.connect(controller.hide_overlay)
        self.settings_button.clicked.connect(self.open_settings)
        self.memory_button.clicked.connect(self.open_memory_management)
        self.vision_button.clicked.connect(controller.identify_current_view)
        self.cancel_vision_button.clicked.connect(controller.cancel_vision)
        self.realtime_connect_button.clicked.connect(controller.connect_realtime)
        self.realtime_disconnect_button.clicked.connect(controller.disconnect_realtime)
        self.realtime_cancel_button.clicked.connect(controller.cancel_realtime)
        self.exit_button.clicked.connect(controller.quit)

    def _new_page(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setObjectName("chapterScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QFrame()
        page.setObjectName("parchmentPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(38, 28, 38, 30)
        layout.setSpacing(14)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        layout.addWidget(title_label)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        scroll.setWidget(page)
        self.chapter_stack.addWidget(scroll)
        return page, layout

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _status_value() -> QLabel:
        label = QLabel("—")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return label

    @Slot(int)
    def _set_chapter(self, index: int) -> None:
        self.chapter_stack.setCurrentIndex(index)
        if 0 <= index < len(self.chapter_buttons):
            self.chapter_buttons[index].setChecked(True)

    @Slot()
    def open_settings(self) -> None:
        SettingsDialog(self.controller.application, self).exec()

    @Slot()
    def open_memory_management(self) -> None:
        MemoryManagementDialog(self.controller.application, self).exec()

    def refresh(self) -> None:
        snapshot = self.controller.application.runtime_snapshot()
        self.status_labels["overall"].setText(snapshot.phase.value)
        self.overall_banner.setText(
            "一切就绪" if snapshot.listening is ListeningState.LISTENING else "守望暂歇"
        )
        self.status_labels["listening"].setText(snapshot.listening.value)
        self.status_labels["mode"].setText("在线模式" if snapshot.online else "离线模式")
        self.status_labels["config"].setText(snapshot.config_source)
        self.status_labels["knowledge"].setText(
            f"{snapshot.knowledge_name} · {snapshot.knowledge_version} · {snapshot.knowledge_capability}"
        )
        self.status_labels["log"].setText(snapshot.log_connection)
        self.status_labels["rag"].setText(snapshot.rag_status)
        self.status_labels["voice"].setText(snapshot.voice_status)
        rt = snapshot.realtime
        estimate = "估算" if rt.estimated else "服务端用量"
        self.status_labels["realtime"].setText(
            f"{rt.state.value} · {rt.status}\n"
            f"{rt.session_seconds / 60:.1f} 分钟 · {rt.turns} 轮 · "
            f"¥{rt.estimated_cost_cny:.6f}（{estimate}） · 预算 {rt.budget_progress:.0%} · "
            f"semantic_vad {'开' if rt.semantic_vad else '关'}\n"
            f"Token：文本/图片入 {rt.text_image_input_tokens} · 音频入 {rt.audio_input_tokens} · "
            f"文本出 {rt.text_output_tokens} · 音频出 {rt.audio_output_tokens}"
        )
        self.status_labels["cost"].setText(
            f"¥{snapshot.run_cost_cny:.6f} / ¥{snapshot.run_budget_cny:.2f}"
        )
        self.status_labels["memory"].setText(snapshot.memory_status)
        self.status_labels["vision"].setText(snapshot.vision_status)
        self.vision_button.setEnabled(self.controller.application.vision.enabled)
        self.realtime_connect_button.setEnabled(rt.enabled and not rt.connected)
        self.realtime_disconnect_button.setEnabled(rt.connected)
        self.realtime_cancel_button.setEnabled(rt.connected)
        listening = snapshot.listening is ListeningState.LISTENING
        self.start_button.setText("恢复监听" if not listening else "监听中")
        self.start_button.setEnabled(not listening)
        self.pause_button.setEnabled(listening)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._allow_close:
            event.accept()
        elif self.controller.tray_available:
            self.hide()
            event.ignore()
        else:
            self.close_hint.setText("系统托盘不可用。请使用“完全退出 Oriens”安全退出。")
            event.ignore()


class DesktopController(QObject):
    """Qt 桌面外壳所有者；托盘和窗口只调用这里的应用命令。"""

    state_changed = Signal()

    def __init__(
        self,
        application: OriensApplication,
        qt_app: QApplication,
        *,
        tray_available: bool | None = None,
    ) -> None:
        super().__init__()
        self.application = application
        self.qt_app = qt_app
        apply_application_theme(self.qt_app, application.paths)
        self.qt_app.setQuitOnLastWindowClosed(False)
        self._exiting = False
        self.overlay = OverlayWindow(application=application, auto_poll=False)
        self.control_center = ControlCenterWindow(self)
        detected = QSystemTrayIcon.isSystemTrayAvailable()
        self.tray_available = detected if tray_available is None else tray_available
        self.tray: QSystemTrayIcon | None = None
        self._create_tray()
        self.timer = QTimer(self)
        self.timer.setInterval(application.config.app.poll_interval_ms)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.state_changed.connect(self._refresh_views)
        self._refresh_views()

    def _create_tray(self) -> None:
        if not self.tray_available:
            return
        icon = application_icon(self.application.paths)
        if icon.isNull():
            icon = self.qt_app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("Oriens 桌面伴侣")
        menu = QMenu()
        self.open_action = menu.addAction("打开控制中心")
        self.overlay_action = menu.addAction("显示悬浮窗")
        self.listening_action = menu.addAction("暂停监听")
        menu.addSeparator()
        self.status_action = menu.addAction("当前状态：已就绪")
        self.status_action.setEnabled(False)
        menu.addSeparator()
        self.exit_action = menu.addAction("完全退出 Oriens")
        self.open_action.triggered.connect(self.show_control_center)
        self.overlay_action.triggered.connect(self.toggle_overlay)
        self.listening_action.triggered.connect(self.toggle_listening)
        self.exit_action.triggered.connect(self.quit)
        tray.setContextMenu(menu)
        tray.activated.connect(self._tray_activated)
        tray.show()
        self.tray = tray

    @Slot(QSystemTrayIcon.ActivationReason)
    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_control_center()

    @Slot()
    def _tick(self) -> None:
        self.application.realtime.maintenance()
        for event in self.application.poll_events():
            self.overlay.handle_event(event)
        self.state_changed.emit()

    @Slot()
    def _refresh_views(self) -> None:
        self.control_center.refresh()
        self.overlay.refresh_realtime()
        snapshot = self.application.runtime_snapshot()
        if self.tray is not None:
            self.overlay_action.setText("隐藏悬浮窗" if self.overlay.isVisible() else "显示悬浮窗")
            self.listening_action.setText(
                "暂停监听" if snapshot.listening is ListeningState.LISTENING else "恢复监听"
            )
            self.status_action.setText(f"当前状态：{snapshot.listening.value}")

    @Slot()
    def show_control_center(self) -> None:
        self.control_center.show()
        self.control_center.raise_()
        self.control_center.activateWindow()

    @Slot()
    def show_overlay(self) -> None:
        self.overlay.show()
        self.overlay.raise_()
        self.state_changed.emit()

    @Slot()
    def hide_overlay(self) -> None:
        self.overlay.suspend_interaction()
        self.overlay.hide()
        self.state_changed.emit()

    @Slot()
    def toggle_overlay(self) -> None:
        self.hide_overlay() if self.overlay.isVisible() else self.show_overlay()

    @Slot()
    def pause_listening(self) -> None:
        self.application.pause_listening()
        self.state_changed.emit()

    @Slot()
    def resume_listening(self) -> None:
        self.application.resume_listening()
        self.state_changed.emit()

    @Slot()
    def toggle_listening(self) -> None:
        self.pause_listening() if self.application.listening else self.resume_listening()

    @Slot()
    def identify_current_view(self) -> None:
        self.show_overlay()
        self.overlay.start_vision()

    @Slot()
    def cancel_vision(self) -> None:
        self.overlay.cancel_vision()

    @Slot()
    def connect_realtime(self) -> None:
        self.application.realtime.connect()
        self.state_changed.emit()

    @Slot()
    def disconnect_realtime(self) -> None:
        self.application.realtime.disconnect()
        self.state_changed.emit()

    @Slot()
    def cancel_realtime(self) -> None:
        self.application.realtime.cancel()
        self.state_changed.emit()

    @Slot()
    def quit(self) -> None:
        if self._exiting:
            return
        self._exiting = True
        self.timer.stop()
        self.overlay.prepare_shutdown()
        self.control_center._allow_close = True
        if self.tray is not None:
            self.tray.hide()
        self.application.close()
        self.overlay.close()
        self.control_center.close()
        for window in QApplication.topLevelWidgets():
            window.close()
        self.qt_app.quit()


def run_desktop(*, application: OriensApplication) -> int:
    qt_app = QApplication.instance() or QApplication([])
    qt_app.setApplicationName("Oriens")
    qt_app.setApplicationDisplayName("Oriens：你的游戏向导")
    controller = DesktopController(application, qt_app)
    controller.show_control_center()
    controller.show_overlay()
    try:
        return qt_app.exec()
    finally:
        controller.quit()


def _display_memory_time(created_at: str, updated_at: str) -> str:
    created = created_at.replace("T", " ").replace("+00:00", " UTC")
    updated = updated_at.replace("T", " ").replace("+00:00", " UTC")
    if created == updated:
        return created
    return f"创建 {created}；更新 {updated}"
