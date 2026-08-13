from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from oriens.config import ConfigError, ConfigService
from oriens.paths import AppPaths


class ConfigServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[2]

    def test_priority_is_default_then_user_then_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            user = Path(directory) / "user"
            paths = AppPaths.development(self.repository, user_data=user)
            service = ConfigService(paths)
            service.save_user_overrides({"voice": {"push_to_talk_key": "F8"}})
            explicit = Path(directory) / "explicit.toml"
            explicit.write_text('[voice]\npush_to_talk_key = "F9"\n', encoding="utf-8")
            self.assertEqual(service.load().voice.push_to_talk_key, "F8")
            self.assertEqual(service.load(explicit).voice.push_to_talk_key, "F9")

    def test_atomic_save_replaces_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(self.repository, user_data=Path(directory))
            service = ConfigService(paths)
            service.save_user_overrides({"voice": {"enabled": False}})
            first = paths.user_config_file.read_text(encoding="utf-8")
            service.save_user_overrides({"voice": {"enabled": True}})
            second = paths.user_config_file.read_text(encoding="utf-8")
            self.assertIn("enabled = false", first)
            self.assertIn("enabled = true", second)
            self.assertNotIn("enabled = false", second)
            self.assertEqual(list(paths.config_dir.glob("*.tmp")), [])

    def test_api_key_and_workspace_id_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(self.repository, user_data=Path(directory))
            service = ConfigService(paths)
            with self.assertRaisesRegex(ConfigError, "不允许"):
                service.save_user_overrides({"providers": {"qwen": {"api_key": "secret"}}})
            with self.assertRaisesRegex(ConfigError, "不允许"):
                service.save_user_overrides({"voice": {"workspace_id_env": "secret"}})
            self.assertFalse(paths.user_config_file.exists())

    def test_unknown_schema_field_is_rejected_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(self.repository, user_data=Path(directory) / "user")
            explicit = Path(directory) / "invalid.toml"
            explicit.write_text('[app]\nunknown_setting = true\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "未知字段"):
                ConfigService(paths).load(explicit)

    def test_settings_update_preserves_other_allowed_user_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(self.repository, user_data=Path(directory) / "user")
            service = ConfigService(paths)
            service.save_user_overrides({"audio": {"chunk_duration_ms": 80}})
            service.update_user_overrides({"voice": {"enabled": False}})
            payload = paths.user_config_file.read_text(encoding="utf-8")
            self.assertIn("chunk_duration_ms = 80", payload)
            self.assertIn("enabled = false", payload)


if __name__ == "__main__":
    unittest.main()
