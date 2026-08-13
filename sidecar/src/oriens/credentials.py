"""凭据读取边界；凭据永不进入普通配置对象或日志。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_api_key
from .paths import AppPaths, RuntimeMode


@dataclass(frozen=True, slots=True)
class CredentialService:
    paths: AppPaths

    def read(self, variable_name: str) -> str | None:
        """优先读进程环境；仅开发模式兼容仓库 `.env`。"""

        env_file: Path
        if self.paths.mode is RuntimeMode.DEVELOPMENT and self.paths.repository is not None:
            env_file = self.paths.repository / ".env"
        else:
            # 正式产品后续可在此接 Windows Credential Manager。
            env_file = self.paths.user_data / ".env-not-supported"
        return load_api_key(variable_name, env_file)

