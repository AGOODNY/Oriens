"""Oriens 程序资源与用户数据路径边界。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path


class RuntimeMode(str, Enum):
    DEVELOPMENT = "development"
    INSTALLED = "installed"


@dataclass(frozen=True, slots=True)
class AppPaths:
    """集中描述路径，不在构造时创建或迁移任何目录。"""

    mode: RuntimeMode
    resources: Path
    user_data: Path
    repository: Path | None = None

    @classmethod
    def development(
        cls,
        repository: Path | None = None,
        *,
        user_data: Path | None = None,
    ) -> "AppPaths":
        root = (repository or Path(__file__).resolve().parents[3]).resolve()
        return cls(
            RuntimeMode.DEVELOPMENT,
            root,
            (user_data or root / ".oriens-user").resolve(),
            root,
        )

    @classmethod
    def installed(
        cls,
        resources: Path,
        *,
        local_app_data: Path | None = None,
    ) -> "AppPaths":
        base = local_app_data
        if base is None:
            value = os.environ.get("LOCALAPPDATA")
            if not value:
                raise RuntimeError("无法确定本地用户数据目录，请重新安装或联系支持。")
            base = Path(value)
        return cls(
            RuntimeMode.INSTALLED,
            resources.resolve(),
            (base / "Oriens").resolve(),
            None,
        )

    @property
    def config_dir(self) -> Path:
        return self.user_data / "config"

    @property
    def user_config_file(self) -> Path:
        return self.config_dir / "settings.toml"

    @property
    def knowledge_dir(self) -> Path:
        return self.user_data / "knowledge"

    @property
    def models_dir(self) -> Path:
        return self.user_data / "models"

    @property
    def cache_dir(self) -> Path:
        return self.user_data / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.user_data / "logs"

    @property
    def memory_dir(self) -> Path:
        """阶段 4 本地记忆位置；仅在用户启用真实记忆存储时创建。"""

        return self.user_data / "memory"

    def model_dir_for(self, model_id: str) -> Path:
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "-"
            for character in model_id.strip()
        ).strip(".-")
        if not safe_name:
            raise ValueError("模型标识无法映射到本地目录。")
        return self.models_dir / safe_name

    @property
    def default_config_file(self) -> Path:
        return self.resources / "config" / "default.toml"

    @property
    def development_data_dir(self) -> Path | None:
        return self.repository / "data" if self.repository is not None else None

    def resolve_resource_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.resources / candidate).resolve()
