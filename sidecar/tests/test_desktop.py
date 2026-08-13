from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon
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


if __name__ == "__main__":
    unittest.main()
