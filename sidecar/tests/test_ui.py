from __future__ import annotations

import os
from pathlib import Path
import tempfile
from threading import Event
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # 阶段 0 的无依赖解释器仍可运行其余测试。
    QApplication = None  # type: ignore[assignment]

from oriens.advice import AdviceEngine
from oriens.application import LaunchOptions, OriensApplication
from oriens.budget import BudgetTracker
from oriens.config import load_config
from oriens.knowledge import LocalItemKnowledgeBase
from oriens.modeling import ModelRouter
from oriens.paths import AppPaths
from oriens.protocol import GameEvent
from oriens.voice import VoiceState


@unittest.skipIf(QApplication is None, "当前解释器未安装 PySide6")
class OverlayTests(unittest.TestCase):
    def test_room_type_names_match_game_enums(self) -> None:
        from oriens.ui import ROOM_NAMES

        self.assertEqual(set(ROOM_NAMES), set(range(31)))
        self.assertEqual(ROOM_NAMES[7], "隐藏房")
        self.assertEqual(ROOM_NAMES[8], "超级隐藏房")
        self.assertEqual(ROOM_NAMES[10], "诅咒房（刺房）")
        self.assertEqual(ROOM_NAMES[14], "恶魔房")
        self.assertEqual(ROOM_NAMES[15], "天使房")

    def test_overlay_constructs_and_closes_without_network(self) -> None:
        assert QApplication is not None
        from oriens.ui import OverlayWindow

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            application = OriensApplication.build(
                AppPaths.development(),
                LaunchOptions(
                    config_path=Path("config/rag-v2.1-faiss.toml"),
                    log_path=Path(directory) / "missing.log",
                    online=False,
                    enable_vector=False,
                ),
            )
            window = OverlayWindow(
                application=application,
            )
            window.show()
            app.processEvents()
            window.resize(500, 620)
            app.processEvents()
            self.assertIn("Oriens", window.windowTitle())
            self.assertIn("离线模拟模式", window.mode_label.text())
            self.assertEqual(window.voice_status_label.text(), "离线不可用")
            self.assertTrue(window.ask_button.isEnabled())
            self.assertFalse(window.ptt_button.isEnabled())
            self.assertGreaterEqual(window.input_device_combo.minimumHeight(), 38)
            self.assertGreaterEqual(window.ptt_button.minimumHeight(), 40)
            self.assertGreaterEqual(window.question_input.minimumHeight(), 38)
            self.assertGreaterEqual(window.question_input.height(), 38)
            self.assertGreaterEqual(window.ptt_button.height(), 40)
            self.assertTrue(window.main_scroll.widgetResizable())
            self.assertGreater(window.main_scroll.verticalScrollBar().maximum(), 0)
            window._start_thinking()
            app.processEvents()
            self.assertTrue(window.thinking_widget.isVisible())
            self.assertTrue(window.thinking_spinner.is_running())
            self.assertFalse(window.advice_label.isVisible())
            window.compact_button.click()
            app.processEvents()
            self.assertTrue(window.thinking_widget.isVisible())
            self.assertFalse(window.advice_label.isVisible())
            self.assertFalse(window.reason_label.isVisible())
            window.compact_button.click()
            app.processEvents()
            self.assertTrue(window.thinking_widget.isVisible())
            self.assertFalse(window.reason_label.isVisible())
            window._stop_thinking()
            app.processEvents()
            self.assertFalse(window.thinking_widget.isVisible())
            self.assertFalse(window.thinking_spinner.is_running())
            self.assertTrue(window.advice_label.isVisible())
            window.advice_label.setText("有效文字回答")
            window._show_voice_error("", "语音播报失败，文字回答仍可查看。")
            self.assertEqual(window.advice_label.text(), "有效文字回答")
            self.assertIn("QComboBox QAbstractItemView", window.styleSheet())
            self.assertIn("selection-color: #ffffff", window.styleSheet())
            normal_size = window.size()
            window.compact_button.click()
            app.processEvents()
            self.assertTrue(window._compact_mode)
            self.assertTrue(window.question_input.isVisible())
            self.assertTrue(window.advice_label.isVisible())
            self.assertFalse(window.input_device_combo.isVisible())
            self.assertFalse(window.item_frame.isVisible())
            self.assertFalse(window.rag_scroll.isVisible())
            self.assertLessEqual(window.height(), 320)
            window.compact_button.click()
            app.processEvents()
            self.assertFalse(window._compact_mode)
            self.assertTrue(window.input_device_combo.isVisible())
            self.assertEqual(window.size(), normal_size)
            window._voice_signals.state.emit("", VoiceState.LISTENING)
            app.processEvents()
            self.assertEqual(window.voice_status_label.text(), "正在聆听")
            event = GameEvent(
                schema_version=1,
                seq=1,
                run_id="UI TEST:0",
                type="collectible_spawned",
                game_frame=100,
                context={
                    "stage": 1,
                    "room_index": 4,
                    "room_type": 4,
                    "room_spawn_seed": 99,
                },
                payload={"collectible_id": 350, "init_seed": 123, "price": 0},
            )
            window._handle_line(event.to_json())
            deadline = time.monotonic() + 2
            while window._future is not None and not window._future.done():
                app.processEvents()
                if time.monotonic() >= deadline:
                    self.fail("模拟建议未在 2 秒内完成")
                time.sleep(0.01)
            app.processEvents()
            self.assertIn("剧毒休克", window.item_label.text())
            self.assertIn("值得拿", window.advice_label.text())
            self.assertIn("href=", window.source_label.text())
            self.assertIn("color:#b9e5ff", window.source_label.text())
            self.assertIn("¥0.000000", window.cost_label.text())
            assert window._future is not None
            old_response, old_token = window._future.result()
            moved = GameEvent(
                schema_version=1,
                seq=2,
                run_id="UI TEST:0",
                type="room_entered",
                game_frame=200,
                context={
                    "stage": 1,
                    "room_index": 5,
                    "room_type": 1,
                    "room_spawn_seed": 100,
                },
                payload={},
            )
            window._handle_line(moved.to_json())
            window.advice_label.setText("当前房间内容")
            window._show_advice(old_response, old_token)
            self.assertEqual(window.advice_label.text(), "当前房间内容")
            self.assertEqual(window.connection_label.text(), "过期建议已丢弃")
            window.close()
            app.processEvents()
            self.assertFalse(window.isVisible())

    def test_injected_voice_service_survives_hide_and_closes_on_shutdown(self) -> None:
        assert QApplication is not None
        from oriens.ui import OverlayWindow

        class FakeVoiceService:
            closed = False
            cancelled = 0
            pressed = 0
            def devices(self): return (("test", "测试麦克风"),)
            def is_current(self, request_id): return True
            def speak_validated(self, text): return "id"
            def room_changed(self): return None
            def press(self, device_id): self.pressed += 1
            def cancel(self): self.cancelled += 1
            def close(self): self.closed = True

        app = QApplication.instance() or QApplication([])
        fake = FakeVoiceService()
        with tempfile.TemporaryDirectory() as directory:
            application = OriensApplication.build(
                AppPaths.development(),
                LaunchOptions(
                    config_path=Path("config/rag-v2.1-faiss.toml"),
                    log_path=Path(directory) / "missing.log",
                    online=False,
                    enable_vector=False,
                ),
            )
            window = OverlayWindow(
                application=application,
                voice_service=fake,
            )
            self.assertEqual(window.input_device_combo.count(), 1)
            window._cancel = Event()
            window._voice_press()
            self.assertTrue(window._cancel is None)
            self.assertEqual(fake.pressed, 1)
            window.close()
            app.processEvents()
            self.assertFalse(fake.closed)
            self.assertFalse(window.isVisible())
            window.show()
            window.prepare_shutdown()
            window.close()
            app.processEvents()
            self.assertTrue(fake.closed)
            application.close()


if __name__ == "__main__":
    unittest.main()
