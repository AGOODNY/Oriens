"""阶段 3.6 Windows 桌面产品外壳。"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .application import ListeningState, OriensApplication
from .config import ConfigError, ConfigService
from .knowledge_pack import KnowledgePackError, KnowledgePackManager
from .ui import OverlayWindow


class SettingsDialog(QDialog):
    """仅保存阶段 3.5 已允许的非敏感设置。"""

    def __init__(self, application: OriensApplication, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.application = application
        self.setWindowTitle("Oriens 设置")
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
        form.addRow("语音", self.voice_enabled)
        form.addRow("按键说话键", self.ptt_key)
        form.addRow("TTS 音量", self.tts_volume)
        form.addRow("TTS 语速", self.tts_rate)
        form.addRow("TTS 音色", self.tts_voice)
        form.addRow("本局预算上限（元）", self.budget)
        form.addRow("当前知识包", self.knowledge_pack)
        layout.addLayout(form)
        note = QLabel(
            "保存后将在下次启动 Oriens 时生效；开发命令的显式配置可能覆盖这些值。"
            "API Key 和业务空间 ID 不会写入此设置。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
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
        self.accept()


class ControlCenterWindow(QMainWindow):
    def __init__(self, controller: "DesktopController") -> None:
        super().__init__()
        self.controller = controller
        self._allow_close = False
        self.setWindowTitle("Oriens 控制中心")
        self.resize(620, 620)
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel("Oriens 桌面伴侣")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(title)
        self.status_labels: dict[str, QLabel] = {}
        status_group = QGroupBox("运行状态")
        status_form = QFormLayout(status_group)
        for key, caption in (
            ("overall", "整体状态"),
            ("listening", "监听状态"),
            ("mode", "运行模式"),
            ("config", "配置来源"),
            ("knowledge", "知识包"),
            ("log", "游戏日志"),
            ("rag", "RAG"),
            ("voice", "语音"),
            ("cost", "当前会话费用"),
            ("memory", "长期记忆"),
        ):
            label = QLabel("—")
            label.setWordWrap(True)
            self.status_labels[key] = label
            status_form.addRow(caption, label)
        layout.addWidget(status_group)

        row = QHBoxLayout()
        self.start_button = QPushButton("启动监听")
        self.pause_button = QPushButton("暂停监听")
        self.show_overlay_button = QPushButton("显示悬浮窗")
        self.hide_overlay_button = QPushButton("隐藏悬浮窗")
        self.settings_button = QPushButton("打开设置")
        self.exit_button = QPushButton("完全退出 Oriens")
        for button in (
            self.start_button,
            self.pause_button,
            self.show_overlay_button,
            self.hide_overlay_button,
            self.settings_button,
            self.exit_button,
        ):
            row.addWidget(button)
        layout.addLayout(row)
        self.close_hint = QLabel("关闭此窗口会缩到系统托盘；只有“完全退出”才会停止后台核心。")
        self.close_hint.setWordWrap(True)
        layout.addWidget(self.close_hint)
        self.setCentralWidget(root)

        self.start_button.clicked.connect(controller.resume_listening)
        self.pause_button.clicked.connect(controller.pause_listening)
        self.show_overlay_button.clicked.connect(controller.show_overlay)
        self.hide_overlay_button.clicked.connect(controller.hide_overlay)
        self.settings_button.clicked.connect(self.open_settings)
        self.exit_button.clicked.connect(controller.quit)

    @Slot()
    def open_settings(self) -> None:
        SettingsDialog(self.controller.application, self).exec()

    def refresh(self) -> None:
        snapshot = self.controller.application.runtime_snapshot()
        self.status_labels["overall"].setText(snapshot.phase.value)
        self.status_labels["listening"].setText(snapshot.listening.value)
        self.status_labels["mode"].setText("在线模式" if snapshot.online else "离线模式")
        self.status_labels["config"].setText(snapshot.config_source)
        self.status_labels["knowledge"].setText(
            f"{snapshot.knowledge_name} · {snapshot.knowledge_version} · {snapshot.knowledge_capability}"
        )
        self.status_labels["log"].setText(snapshot.log_connection)
        self.status_labels["rag"].setText(snapshot.rag_status)
        self.status_labels["voice"].setText(snapshot.voice_status)
        self.status_labels["cost"].setText(
            f"¥{snapshot.run_cost_cny:.6f} / ¥{snapshot.run_budget_cny:.2f}"
        )
        self.status_labels["memory"].setText("尚未实现")
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
        for event in self.application.poll_events():
            self.overlay.handle_event(event)
        self.state_changed.emit()

    @Slot()
    def _refresh_views(self) -> None:
        self.control_center.refresh()
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
