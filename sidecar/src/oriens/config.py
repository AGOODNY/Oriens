"""阶段 1 配置与密钥加载。

模型名称、端点和单价全部来自 TOML；业务层只引用模型角色。
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any

from .paths import AppPaths


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
class AudioSettings:
    input_format: str
    input_sample_rate: int
    input_channels: int
    input_sample_width_bytes: int
    chunk_duration_ms: int
    max_recording_seconds: float
    min_recording_ms: int
    silence_rms_threshold: int
    noise_rms_threshold: int
    playback_format: str
    playback_sample_rate: int
    playback_channels: int
    playback_queue_max_chunks: int


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    enabled: bool
    push_to_talk_key: str
    asr_provider: str
    asr_model_id: str
    asr_endpoint: str
    asr_language: str
    asr_timeout_seconds: float
    asr_max_retries: int
    asr_price_per_second_cny: float
    tts_provider: str
    tts_model_id: str
    tts_endpoint: str
    tts_voice: str
    tts_rate: float
    tts_volume: int
    tts_format: str
    tts_sample_rate: int
    tts_timeout_seconds: float
    tts_max_retries: int
    tts_max_segment_chars: int
    tts_price_per_10k_chars_cny: float
    pricing_checked_on: str
    workspace_id_env: str


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
    pipeline_version: int
    content_version: str
    raw_paths: tuple[Path, ...]
    entities_path: Path
    redirects_path: Path
    dependency_audit_path: Path
    lua_facts_path: Path
    overrides_path: Path
    vector_backend: str
    vector_index_path: Path
    vector_batch_size: int
    vector_max_sequence_length: int
    vector_device: str
    vector_build_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class OriensConfig:
    root: Path
    app: AppSettings
    providers: dict[str, ProviderSettings]
    model_roles: dict[str, ModelRoleSettings]
    budget: BudgetSettings
    rag: RagSettings
    audio: AudioSettings
    voice: VoiceSettings

    def provider_for(self, role: str) -> tuple[ProviderSettings, ModelRoleSettings]:
        try:
            model = self.model_roles[role]
            provider = self.providers[model.provider]
        except KeyError as exc:
            raise ConfigError(f"模型角色配置不存在：{role}") from exc
        return provider, model


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ConfigService:
    """为 CLI 与未来设置页提供统一配置加载和原子保存。"""

    def __init__(
        self,
        paths: AppPaths,
        *,
        defaults_path: Path | None = None,
        user_path: Path | None = None,
    ) -> None:
        self.paths = paths
        self.defaults_path = (defaults_path or paths.default_config_file).resolve()
        self.user_path = (user_path or paths.user_config_file).resolve()

    def load(self, explicit_path: Path | None = None) -> OriensConfig:
        raw = _read_toml(self.defaults_path, "默认配置")
        if self.user_path.is_file():
            user = _read_toml(self.user_path, "用户配置")
            _validate_user_overrides(user)
            raw = _merge(raw, user)
        if explicit_path is not None:
            raw = _merge(raw, _read_toml(explicit_path.resolve(), "显式配置"))
        return _parse_config(raw, self.paths.resources)

    def save_user_overrides(self, overrides: dict[str, Any]) -> None:
        _validate_user_overrides(overrides)
        # 合并后走完整 schema 校验，避免未来 GUI 保存出无法启动的设置。
        combined = _merge(_read_toml(self.defaults_path, "默认配置"), overrides)
        _parse_config(combined, self.paths.resources)
        payload = _dump_toml(overrides)
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(
            prefix=f".{self.user_path.name}.", suffix=".tmp", dir=self.user_path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(name, self.user_path)
        except Exception:
            # 不自动删除失败暂存文件；它不影响正式配置，且便于诊断。
            raise

    def update_user_overrides(self, overrides: dict[str, Any]) -> None:
        """合并设置页管理的字段，同时保留其他已允许的用户设置。"""

        existing: dict[str, Any] = {}
        if self.user_path.is_file():
            existing = _read_toml(self.user_path, "用户配置")
            _validate_user_overrides(existing)
        self.save_user_overrides(_merge(existing, overrides))


def load_config(path: Path | None = None, *, paths: AppPaths | None = None) -> OriensConfig:
    selected_paths = paths or AppPaths.development(repository_root())
    return ConfigService(selected_paths).load(path)


def _read_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"无法读取{label}。") from exc


def _parse_config(raw: dict[str, Any], root: Path) -> OriensConfig:
    _reject_unknown(raw, {
        "app", "providers", "model_roles", "budget", "rag", "audio", "voice"
    }, "配置")

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
        _reject_unknown(value, {
            "region", "base_url", "api_key_env", "timeout_seconds", "max_retries"
        }, f"providers.{name}")
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
        _reject_unknown(value, {
            "provider", "model_id", "display_name", "input_price_per_million_cny",
            "output_price_per_million_cny", "pricing_input_limit_tokens",
            "pricing_checked_on",
        }, f"model_roles.{role}")
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
    missing_providers = sorted({role.provider for role in roles.values()} - set(providers))
    if missing_providers:
        raise ConfigError(f"模型角色引用了不存在的提供商：{'、'.join(missing_providers)}")

    budget = BudgetSettings(run_limit_cny=_positive_float(budget_raw, "run_limit_cny"))
    audio_raw = _section(raw, "audio")
    voice_raw = _section(raw, "voice")
    _reject_unknown(app_raw, {"language", "poll_interval_ms", "recent_event_limit", "knowledge_path"}, "app")
    _reject_unknown(budget_raw, {"run_limit_cny"}, "budget")
    _reject_unknown(audio_raw, {
        "input_format", "input_sample_rate", "input_channels", "input_sample_width_bytes",
        "chunk_duration_ms", "max_recording_seconds", "min_recording_ms",
        "silence_rms_threshold", "noise_rms_threshold", "playback_format",
        "playback_sample_rate", "playback_channels", "playback_queue_max_chunks",
    }, "audio")
    _reject_unknown(voice_raw, {
        "enabled", "push_to_talk_key", "asr_provider", "asr_model_id", "asr_endpoint",
        "asr_language", "asr_timeout_seconds", "asr_max_retries", "asr_price_per_second_cny",
        "tts_provider", "tts_model_id", "tts_endpoint", "tts_voice", "tts_rate", "tts_volume",
        "tts_format", "tts_sample_rate", "tts_timeout_seconds", "tts_max_retries",
        "tts_max_segment_chars", "tts_price_per_10k_chars_cny", "pricing_checked_on",
        "workspace_id_env",
    }, "voice")
    _reject_unknown(rag_raw, {
        "source_path", "chunks_path", "manifest_path", "index_path", "eval_path",
        "game_version", "vector_enabled", "vector_model_id", "vector_model_path",
        "vector_dimension", "vector_min_similarity", "vector_query_timeout_seconds",
        "retrieval_top_k", "pipeline_version", "content_version", "raw_paths", "entities_path",
        "redirects_path", "dependency_audit_path", "lua_facts_path", "overrides_path",
        "vector_backend", "vector_index_path", "vector_batch_size",
        "vector_max_sequence_length", "vector_device", "vector_build_timeout_seconds",
    }, "rag")
    audio = AudioSettings(
        input_format=_optional_choice(audio_raw, "input_format", "pcm_s16le", {"pcm_s16le"}),
        input_sample_rate=_positive_int(audio_raw, "input_sample_rate"),
        input_channels=_optional_positive_int(audio_raw, "input_channels", 1),
        input_sample_width_bytes=_optional_positive_int(audio_raw, "input_sample_width_bytes", 2),
        chunk_duration_ms=_optional_positive_int(audio_raw, "chunk_duration_ms", 100),
        max_recording_seconds=_optional_positive_float(audio_raw, "max_recording_seconds", 30.0),
        min_recording_ms=_optional_positive_int(audio_raw, "min_recording_ms", 250),
        silence_rms_threshold=_optional_nonnegative_int(audio_raw, "silence_rms_threshold", 90),
        noise_rms_threshold=_optional_nonnegative_int(audio_raw, "noise_rms_threshold", 180),
        playback_format=_optional_choice(audio_raw, "playback_format", "pcm_s16le", {"pcm_s16le"}),
        playback_sample_rate=_positive_int(audio_raw, "playback_sample_rate"),
        playback_channels=_optional_positive_int(audio_raw, "playback_channels", 1),
        playback_queue_max_chunks=_optional_positive_int(audio_raw, "playback_queue_max_chunks", 64),
    )
    voice = VoiceSettings(
        enabled=_optional_boolean(voice_raw, "enabled", True),
        push_to_talk_key=_optional_string(voice_raw, "push_to_talk_key", "Space"),
        asr_provider=_string(voice_raw, "asr_provider"),
        asr_model_id=_string(voice_raw, "asr_model_id"),
        asr_endpoint=_string(voice_raw, "asr_endpoint"),
        asr_language=_optional_string(voice_raw, "asr_language", "zh"),
        asr_timeout_seconds=_positive_float(voice_raw, "asr_timeout_seconds"),
        asr_max_retries=_nonnegative_int(voice_raw, "asr_max_retries"),
        asr_price_per_second_cny=_nonnegative_float(voice_raw, "asr_price_per_second_cny"),
        tts_provider=_string(voice_raw, "tts_provider"),
        tts_model_id=_string(voice_raw, "tts_model_id"),
        tts_endpoint=_string(voice_raw, "tts_endpoint"),
        tts_voice=_string(voice_raw, "tts_voice"),
        tts_rate=_optional_range_float(voice_raw, "tts_rate", 1.0, 0.5, 2.0),
        tts_volume=_optional_range_int(voice_raw, "tts_volume", 50, 0, 100),
        tts_format=_optional_choice(voice_raw, "tts_format", "pcm", {"pcm", "wav", "mp3", "opus"}),
        tts_sample_rate=_choice_int(voice_raw, "tts_sample_rate", {8000, 16000, 22050, 24000, 44100, 48000}),
        tts_timeout_seconds=_positive_float(voice_raw, "tts_timeout_seconds"),
        tts_max_retries=_nonnegative_int(voice_raw, "tts_max_retries"),
        tts_max_segment_chars=_optional_positive_int(voice_raw, "tts_max_segment_chars", 80),
        tts_price_per_10k_chars_cny=_nonnegative_float(voice_raw, "tts_price_per_10k_chars_cny"),
        pricing_checked_on=_string(voice_raw, "pricing_checked_on"),
        workspace_id_env=_optional_string(voice_raw, "workspace_id_env", "DASHSCOPE_WORKSPACE_ID"),
    )
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
        pipeline_version=_optional_positive_int(rag_raw, "pipeline_version", 1),
        content_version=_optional_string(rag_raw, "content_version", "rag-v1"),
        raw_paths=tuple(
            (root / item).resolve()
            for item in _optional_string_list(rag_raw, "raw_paths")
        ),
        entities_path=(
            root
            / _optional_string(
                rag_raw, "entities_path", "data/knowledge/rag-v1/entities.jsonl"
            )
        ).resolve(),
        redirects_path=(
            root
            / _optional_string(
                rag_raw, "redirects_path", "data/knowledge/rag-v1/redirects.jsonl"
            )
        ).resolve(),
        dependency_audit_path=(
            root
            / _optional_string(
                rag_raw,
                "dependency_audit_path",
                "data/knowledge/rag-v1/dependency-audit.json",
            )
        ).resolve(),
        lua_facts_path=(
            root
            / _optional_string(
                rag_raw, "lua_facts_path", "data/knowledge/rag-v1/lua-facts.jsonl"
            )
        ).resolve(),
        overrides_path=(
            root
            / _optional_string(
                rag_raw,
                "overrides_path",
                "data/dictionaries/rag-v2-overrides.json",
            )
        ).resolve(),
        vector_backend=_optional_string(rag_raw, "vector_backend", "sqlite-vec"),
        vector_index_path=(
            root
            / _optional_string(
                rag_raw, "vector_index_path", _string(rag_raw, "index_path")
            )
        ).resolve(),
        vector_batch_size=_optional_positive_int(rag_raw, "vector_batch_size", 32),
        vector_max_sequence_length=_optional_positive_int(
            rag_raw, "vector_max_sequence_length", 8192
        ),
        vector_device=_optional_choice(
            rag_raw, "vector_device", "cpu", {"cpu", "cuda"}
        ),
        vector_build_timeout_seconds=_optional_positive_float(
            rag_raw, "vector_build_timeout_seconds", 7200.0
        ),
    )
    if audio.input_channels != 1 or audio.input_sample_width_bytes != 2:
        raise ConfigError("阶段 3 麦克风输入必须是单声道 16-bit PCM")
    if audio.input_sample_rate not in {8000, 16000}:
        raise ConfigError("Qwen 实时 ASR 输入采样率必须是 8000 或 16000 Hz")
    if voice.tts_format != "pcm":
        raise ConfigError("阶段 3 本地播放器仅支持 TTS 输出 PCM；请将 voice.tts_format 设为 pcm")
    if voice.tts_sample_rate != audio.playback_sample_rate:
        raise ConfigError("voice.tts_sample_rate 必须与 audio.playback_sample_rate 一致")
    return OriensConfig(root, app, providers, roles, budget, rag, audio, voice)


_USER_CONFIG_ALLOWED: dict[str, frozenset[str]] = {
    "app": frozenset({"language", "poll_interval_ms", "recent_event_limit"}),
    "budget": frozenset({"run_limit_cny"}),
    "audio": frozenset({
        "input_format", "input_sample_rate", "input_channels", "input_sample_width_bytes",
        "chunk_duration_ms", "max_recording_seconds", "min_recording_ms",
        "silence_rms_threshold", "noise_rms_threshold", "playback_format",
        "playback_sample_rate", "playback_channels", "playback_queue_max_chunks",
    }),
    "voice": frozenset({"enabled", "push_to_talk_key", "tts_voice", "tts_rate", "tts_volume"}),
}


def _validate_user_overrides(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ConfigError("用户配置必须是对象。")
    for section, fields in value.items():
        if section not in _USER_CONFIG_ALLOWED or not isinstance(fields, dict):
            raise ConfigError(f"用户配置不允许修改：{section}")
        for field in fields:
            lowered = field.casefold()
            secret_field = (
                lowered in {"api_key", "secret", "token", "workspace_id"}
                or lowered.endswith(("_api_key", "_secret", "_token", "_workspace_id"))
            )
            if field not in _USER_CONFIG_ALLOWED[section] or secret_field:
                raise ConfigError(f"用户配置不允许保存敏感或受保护字段：{section}.{field}")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _dump_toml(value: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in sorted(value):
        fields = value[section]
        lines.append(f"[{section}]")
        for key in sorted(fields):
            lines.append(f"{key} = {_toml_scalar(fields[key])}")
        lines.append("")
    return "\n".join(lines)


def _toml_scalar(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in {int, float}:
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise ConfigError("用户配置包含暂不支持的值类型。")


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


def _reject_unknown(value: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"配置段 {section} 包含未知字段：{'、'.join(unknown)}")


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


def _optional_string(value: dict[str, Any], name: str, default: str) -> str:
    if name not in value:
        return default
    return _string(value, name)


def _optional_positive_int(value: dict[str, Any], name: str, default: int) -> int:
    if name not in value:
        return default
    return _positive_int(value, name)


def _optional_positive_float(
    value: dict[str, Any], name: str, default: float
) -> float:
    if name not in value:
        return default
    return _positive_float(value, name)


def _optional_nonnegative_int(value: dict[str, Any], name: str, default: int) -> int:
    if name not in value:
        return default
    return _nonnegative_int(value, name)


def _optional_nonnegative_float(value: dict[str, Any], name: str, default: float) -> float:
    if name not in value:
        return default
    return _nonnegative_float(value, name)


def _optional_boolean(value: dict[str, Any], name: str, default: bool) -> bool:
    if name not in value:
        return default
    return _boolean(value, name)


def _optional_range_int(
    value: dict[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    result = value.get(name, default)
    if type(result) is not int or not minimum <= result <= maximum:
        raise ConfigError(f"配置 {name} 必须在 {minimum} 到 {maximum} 之间")
    return result


def _optional_range_float(
    value: dict[str, Any], name: str, default: float, minimum: float, maximum: float
) -> float:
    result = value.get(name, default)
    if type(result) not in {int, float} or not minimum <= float(result) <= maximum:
        raise ConfigError(f"配置 {name} 必须在 {minimum} 到 {maximum} 之间")
    return float(result)


def _optional_choice_int(
    value: dict[str, Any], name: str, default: int, choices: set[int]
) -> int:
    result = value.get(name, default)
    if type(result) is not int or result not in choices:
        allowed = "、".join(str(item) for item in sorted(choices))
        raise ConfigError(f"配置 {name} 必须是：{allowed}")
    return result


def _choice_int(value: dict[str, Any], name: str, choices: set[int]) -> int:
    result = value.get(name)
    if type(result) is not int or result not in choices:
        allowed = "、".join(str(item) for item in sorted(choices))
        raise ConfigError(f"配置 {name} 必须是：{allowed}")
    return result


def _optional_choice(
    value: dict[str, Any], name: str, default: str, choices: set[str]
) -> str:
    result = _optional_string(value, name, default)
    if result not in choices:
        allowed = "、".join(sorted(choices))
        raise ConfigError(f"配置 {name} 必须是：{allowed}")
    return result


def _optional_string_list(value: dict[str, Any], name: str) -> tuple[str, ...]:
    result = value.get(name, [])
    if not isinstance(result, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise ConfigError(f"配置 {name} 必须是字符串数组")
    return tuple(item.strip() for item in result)
