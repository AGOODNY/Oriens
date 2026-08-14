"""测试专用配置入口，确保自动化不接触真实用户数据目录。"""

from __future__ import annotations

from pathlib import Path
import tempfile

from oriens.config import load_config
from oriens.paths import AppPaths


_TEMP_USER_ROOT = tempfile.TemporaryDirectory(prefix="oriens-config-tests-")
TEST_PATHS = AppPaths.development(user_data=Path(_TEMP_USER_ROOT.name) / "user")


def load_test_config(path: Path | None = None):
    return load_config(path, paths=TEST_PATHS)
