from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon
except ImportError:
    QApplication = None  # type: ignore[assignment]

from oriens.application import LaunchOptions, ListeningState, OriensApplication
from oriens.cli import build_parser
from oriens.config import ConfigService
from oriens.paths import AppPaths


@unittest.skipIf(QApplication is None, "当前解释器未安装 PySide6")
class DesktopShellTests(unittest.TestCase):
    def _build(self, directory: str):
        return OriensApplication.build(
            AppPaths.development(user_data=Path(directory) / "user"),
            LaunchOptions(
                config_path=Path("config/rag-v2.1-faiss.toml"),
                log_path=Path(directory) / "missing.log",
                online=False,
                enable_vector=False,
            ),
        )

    def _build_with_memory(self, directory: str):
        root = Path(directory)
        explicit = root / "memory-enabled.toml"
        explicit.write_text("[memory]\nenabled = true\n", encoding="utf-8")
        return OriensApplication.build(
            AppPaths.development(user_data=root / "user"),
            LaunchOptions(
                config_path=explicit,
                log_path=root / "missing.log",
                online=False,
                enable_vector=False,
            ),
        )

    def test_one_core_is_shared_by_control_center_and_overlay(self) -> None:
        assert QApplication is not None
        from oriens.desktop import DesktopController

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build(directory)
            controller = DesktopController(application, qt_app, tray_available=False)
            self.assertIs(controller.application, application)
            self.assertIs(controller.overlay.application, application)
            self.assertIs(controller.overlay.store, application.session)
            self.assertIs(controller.application.vision, controller.overlay.application.vision)
            self.assertIs(controller.overlay.budget, application.budget)
            self.assertIs(controller.overlay.advice_engine.rag, application.rag)
            self.assertIs(controller.overlay.advice_engine.router, application.router)
            self.assertIs(controller.overlay.voice_service.query_engine, application.query_engine)
            self.assertFalse(application.router.online)
            self.assertIsNone(controller.overlay.voice_service.asr)
            self.assertIsNone(controller.overlay.voice_service.tts)
            self.assertFalse(application.memory.enabled)
            voice_count = len(application._voice_services)
            controller.show_overlay()
            controller.show_overlay()
            self.assertEqual(len(application._voice_services), voice_count)
            self.assertIn("开发配置知识包", controller.control_center.status_labels["knowledge"].text())
            controller.quit()
            controller.quit()
            self.assertTrue(application.closed)

    def test_control_center_and_tray_commands_manage_windows_and_listening(self) -> None:
        assert QApplication is not None
        from oriens.desktop import DesktopController

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build(directory)
            controller = DesktopController(application, qt_app, tray_available=True)
            controller.show_control_center()
            controller.show_overlay()
            qt_app.processEvents()
            controller.control_center.close()
            qt_app.processEvents()
            self.assertFalse(controller.control_center.isVisible())
            self.assertFalse(application.closed)
            assert controller.tray is not None
            controller._tray_activated(QSystemTrayIcon.ActivationReason.Trigger)
            qt_app.processEvents()
            self.assertTrue(controller.control_center.isVisible())
            controller.toggle_overlay()
            self.assertFalse(controller.overlay.isVisible())
            controller.toggle_overlay()
            self.assertTrue(controller.overlay.isVisible())
            controller.toggle_listening()
            self.assertEqual(
                application.runtime_snapshot().listening, ListeningState.PAUSED
            )
            controller.toggle_listening()
            self.assertEqual(
                application.runtime_snapshot().listening, ListeningState.LISTENING
            )
            controller.quit()
            self.assertTrue(application.closed)
            self.assertFalse(controller.timer.isActive())
            self.assertFalse(controller.overlay.isVisible())
            self.assertFalse(controller.control_center.isVisible())

    def test_tray_unavailable_is_safe_and_does_not_close_core(self) -> None:
        assert QApplication is not None
        from oriens.desktop import DesktopController

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build(directory)
            controller = DesktopController(application, qt_app, tray_available=False)
            controller.show_control_center()
            qt_app.processEvents()
            self.assertIsNone(controller.tray)
            controller.control_center.close()
            qt_app.processEvents()
            self.assertTrue(controller.control_center.isVisible())
            self.assertFalse(application.closed)
            self.assertIn("系统托盘不可用", controller.control_center.close_hint.text())
            controller.quit()

    def test_desktop_and_legacy_ui_keep_the_same_launch_arguments(self) -> None:
        parser = build_parser()
        for command in ("desktop", "ui"):
            args = parser.parse_args(
                [command, "--config", "config/rag-v2.1-faiss.toml", "--online", "--from-start"]
            )
            self.assertEqual(args.config, Path("config/rag-v2.1-faiss.toml"))
            self.assertTrue(args.online)
            self.assertTrue(args.from_start)
            self.assertEqual(args.handler.__name__, "run_desktop_command")

    def test_settings_save_only_whitelisted_values_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(user_data=Path(directory) / "user")
            ConfigService(paths).save_user_overrides(
                {
                    "voice": {
                        "enabled": False,
                        "push_to_talk_key": "F8",
                        "tts_voice": "longanyang",
                        "tts_rate": 1.1,
                        "tts_volume": 40,
                    },
                    "budget": {"run_limit_cny": 0.3},
                }
            )
            payload = paths.user_config_file.read_text(encoding="utf-8")
            self.assertIn("enabled = false", payload)
            self.assertNotIn("api_key", payload.casefold())
            self.assertNotIn("workspace", payload.casefold())

    def test_short_offscreen_event_loop_can_exit_cleanly(self) -> None:
        assert QApplication is not None
        from oriens.desktop import DesktopController

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build(directory)
            controller = DesktopController(application, qt_app, tray_available=False)
            controller.show_control_center()
            QTimer.singleShot(20, controller.quit)
            self.assertEqual(qt_app.exec(), 0)
            self.assertTrue(application.closed)

    def test_memory_management_ui_adds_corrects_toggles_and_deletes(self) -> None:
        assert QApplication is not None
        from oriens.desktop import MemoryManagementDialog

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build_with_memory(directory)
            dialog = MemoryManagementDialog(application)
            self.assertNotIn(str(application.paths.memory_dir), dialog.service_status.text())
            dialog.kind.setCurrentIndex(dialog.kind.findData("guidance_preference"))
            dialog.content.setText("解释深度偏好：详细")
            dialog._add()
            self.assertEqual(dialog.table.rowCount(), 1)
            dialog.table.selectRow(0)
            qt_app.processEvents()
            dialog.content.setText("解释深度偏好：简短")
            dialog._update()
            self.assertEqual(dialog.table.item(0, 1).text(), "解释深度偏好：简短")
            dialog.table.selectRow(0)
            dialog._toggle()
            self.assertEqual(application.list_memories()[0].status, "disabled")
            dialog.table.selectRow(0)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
                dialog._delete()
            self.assertEqual(dialog.table.rowCount(), 1)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                dialog._delete()
            self.assertEqual(dialog.table.rowCount(), 0)
            dialog.close()
            application.close()

    def test_memory_clear_requires_confirmation_and_settings_are_restart_scoped(self) -> None:
        assert QApplication is not None
        from oriens.desktop import MemoryManagementDialog, SettingsDialog

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build_with_memory(directory)
            application.add_memory("profile", "称呼偏好：小林")
            dialog = MemoryManagementDialog(application)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
                dialog._clear()
            self.assertEqual(len(application.list_memories()), 1)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                dialog._clear()
            self.assertEqual(application.list_memories(), ())
            settings = SettingsDialog(application)
            settings.memory_enabled.setChecked(False)
            with patch.object(QMessageBox, "information"):
                settings._save()
            payload = application.paths.user_config_file.read_text(encoding="utf-8")
            self.assertIn("[memory]", payload)
            self.assertIn("enabled = false", payload)
            # 保存只影响下次装配，当前共享连接在完全退出前仍保持有效。
            self.assertTrue(application.memory.enabled)
            settings.close()
            dialog.close()
            application.close()

    def test_offscreen_memory_ui_full_exit_releases_database(self) -> None:
        assert QApplication is not None
        from oriens.desktop import DesktopController, MemoryManagementDialog

        qt_app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = self._build_with_memory(directory)
            controller = DesktopController(application, qt_app, tray_available=False)
            dialog = MemoryManagementDialog(application)
            dialog.content.setText("称呼偏好：小林")
            dialog.kind.setCurrentIndex(dialog.kind.findData("profile"))
            dialog._add()
            database_path = application.memory.database_path
            controller.show_control_center()
            dialog.show()
            QTimer.singleShot(20, controller.quit)
            self.assertEqual(qt_app.exec(), 0)
            self.assertTrue(application.closed)
            connection = sqlite3.connect(database_path, timeout=0.2)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ROLLBACK")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
