"""阶段 1 配置与密钥加载。

模型名称、端点和单价全部来自 TOML；业务层只引用模型角色。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """配置无效或缺少必需字段。"""


@dataclass(frozen=True, slots=True)
class AppSettings:
    language: str
    poll_interval_ms: int
    recent_event_limit: int
    knowledge_path: Path


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    name: str
    region: str
    base_url: str
    api_key_env: str
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class ModelRoleSettings:
    role: str
    provider: str
    model_id: str
    display_name: str
    input_price_per_million_cny: float
    output_price_per_million_cny: float
    pricing_input_limit_tokens: int
    pricing_checked_on: str


@dataclass(frozen=True, slots=True)
class BudgetSettings:
    run_limit_cny: float


@dataclass(frozen=True, slots=True)
class RagSettings:
    source_path: Path
    chunks_path: Path
    manifest_path: Path
    index_path: Path
    eval_path: Path
    game_version: str
    vector_enabled: bool
    vector_model_id: str
    vector_model_path: Path
    vector_dimension: int
    vector_min_similarity: float
    vector_query_timeout_seconds: float
    retrieval_top_k: int


@dataclass(frozen=True, slots=True)
class OriensConfig:
    root: Path
    app: AppSettings
    providers: dict[str, ProviderSettings]
    model_roles: dict[str, ModelRoleSettings]
    budget: BudgetSettings
    rag: RagSettings

    def provider_for(self, role: str) -> tuple[ProviderSettings, ModelRoleSettings]:
        try:
            model = self.model_roles[role]
            provider = self.providers[model.provider]
        except KeyError as exc:
            raise ConfigError(f"模型角色配置不存在：{role}") from exc
        return provider, model


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: Path | None = None) -> OriensConfig:
    root = repository_root()
    config_path = (path or root / "config" / "default.toml").resolve()
    try:
        with config_path.open("rb") as source:
            raw = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"无法读取配置文件：{config_path}") from exc

    app_raw = _section(raw, "app")
    budget_raw = _section(raw, "budget")
    rag_raw = _section(raw, "rag")
    app = AppSettings(
        language=_string(app_raw, "language"),
        poll_interval_ms=_positive_int(app_raw, "poll_interval_ms"),
        recent_event_limit=_positive_int(app_raw, "recent_event_limit"),
        knowledge_path=(root / _string(app_raw, "knowledge_path")).resolve(),
    )

    providers: dict[str, ProviderSettings] = {}
    for name, value in _section(raw, "providers").items():
        if not isinstance(value, dict):
            raise ConfigError(f"providers.{name} 必须是对象")
        providers[name] = ProviderSettings(
            name=name,
            region=_string(value, "region"),
            base_url=_string(value, "base_url").rstrip("/"),
            api_key_env=_string(value, "api_key_env"),
            timeout_seconds=_positive_float(value, "timeout_seconds"),
            max_retries=_nonnegative_int(value, "max_retries"),
        )

    roles: dict[str, ModelRoleSettings] = {}
    for role, value in _section(raw, "model_roles").items():
        if not isinstance(value, dict):
            raise ConfigError(f"model_roles.{role} 必须是对象")
        roles[role] = ModelRoleSettings(
            role=role,
            provider=_string(value, "provider"),
            model_id=_string(value, "model_id"),
            display_name=_string(value, "display_name"),
            input_price_per_million_cny=_nonnegative_float(
                value, "input_price_per_million_cny"
            ),
            output_price_per_million_cny=_nonnegative_float(
                value, "output_price_per_million_cny"
            ),
            pricing_input_limit_tokens=_positive_int(
                value, "pricing_input_limit_tokens"
            ),
            pricing_checked_on=_string(value, "pricing_checked_on"),
        )

    budget = BudgetSettings(run_limit_cny=_positive_float(budget_raw, "run_limit_cny"))
    rag = RagSettings(
        source_path=(root / _string(rag_raw, "source_path")).resolve(),
        chunks_path=(root / _string(rag_raw, "chunks_path")).resolve(),
        manifest_path=(root / _string(rag_raw, "manifest_path")).resolve(),
        index_path=(root / _string(rag_raw, "index_path")).resolve(),
        eval_path=(root / _string(rag_raw, "eval_path")).resolve(),
        game_version=_string(rag_raw, "game_version"),
        vector_enabled=_boolean(rag_raw, "vector_enabled"),
        vector_model_id=_string(rag_raw, "vector_model_id"),
        vector_model_path=(root / _string(rag_raw, "vector_model_path")).resolve(),
        vector_dimension=_positive_int(rag_raw, "vector_dimension"),
        vector_min_similarity=_nonnegative_float(rag_raw, "vector_min_similarity"),
        vector_query_timeout_seconds=_positive_float(
            rag_raw, "vector_query_timeout_seconds"
        ),
        retrieval_top_k=_positive_int(rag_raw, "retrieval_top_k"),
    )
    return OriensConfig(root, app, providers, roles, budget, rag)


def load_api_key(variable_name: str, env_path: Path | None = None) -> str | None:
    """读取密钥但不记录、显示或保存在配置对象中。"""

    value = os.environ.get(variable_name)
    if value and value.strip():
        return value.strip()

    path = env_path or repository_root() / ".env"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return None
    except OSError:
        # 这里不透传可能包含用户名等本机细节的异常。
        return None

    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, raw_value = text.split("=", 1)
        if key.strip() != variable_name:
            continue
        candidate = raw_value.strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
            candidate = candidate[1:-1]
        return candidate or None
    return None


def _section(value: dict[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"缺少配置段：{name}")
    return section


def _string(value: dict[str, Any], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ConfigError(f"配置 {name} 必须是非空字符串")
    return result.strip()


def _positive_int(value: dict[str, Any], name: str) -> int:
    result = value.get(name)
    if type(result) is not int or result <= 0:
        raise ConfigError(f"配置 {name} 必须是正整数")
    return result


def _nonnegative_int(value: dict[str, Any], name: str) -> int:
    result = value.get(name)
    if type(result) is not int or result < 0:
        raise ConfigError(f"配置 {name} 必须是非负整数")
    return result


def _positive_float(value: dict[str, Any], name: str) -> float:
    result = value.get(name)
    if type(result) not in {int, float} or result <= 0:
        raise ConfigError(f"配置 {name} 必须是正数")
    return float(result)


def _nonnegative_float(value: dict[str, Any], name: str) -> float:
    result = value.get(name)
    if type(result) not in {int, float} or result < 0:
        raise ConfigError(f"配置 {name} 必须是非负数")
    return float(result)


def _boolean(value: dict[str, Any], name: str) -> bool:
    result = value.get(name)
    if type(result) is not bool:
        raise ConfigError(f"配置 {name} 必须是布尔值")
    return result
