from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from oriens.paths import AppPaths, RuntimeMode


class AppPathsTests(unittest.TestCase):
    def test_development_paths_keep_repository_resources_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            user = Path(directory) / "profile"
            paths = AppPaths.development(root, user_data=user)
            self.assertEqual(paths.mode, RuntimeMode.DEVELOPMENT)
            self.assertEqual(paths.resources, root.resolve())
            self.assertEqual(paths.development_data_dir, root.resolve() / "data")
            self.assertEqual(paths.user_config_file, user.resolve() / "config/settings.toml")
            self.assertEqual(paths.memory_dir, user.resolve() / "memory")
            self.assertEqual(
                paths.model_dir_for("BAAI/bge-m3"),
                user.resolve() / "models/BAAI-bge-m3",
            )
            self.assertFalse(user.exists())

    def test_installed_paths_are_deterministic_with_injected_local_app_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "Local"
            resources = Path(directory) / "Program"
            paths = AppPaths.installed(resources, local_app_data=base)
            self.assertEqual(paths.mode, RuntimeMode.INSTALLED)
            self.assertEqual(paths.user_data, base.resolve() / "Oriens")
            self.assertEqual(paths.knowledge_dir, base.resolve() / "Oriens/knowledge")
            self.assertEqual(paths.models_dir, base.resolve() / "Oriens/models")
            self.assertEqual(paths.cache_dir, base.resolve() / "Oriens/cache")
            self.assertEqual(paths.logs_dir, base.resolve() / "Oriens/logs")
            self.assertIsNone(paths.repository)


if __name__ == "__main__":
    unittest.main()
