"""统一模型适配器与模型角色路由。"""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
from threading import Event
import time
from typing import Any, Protocol
from urllib import error, request

from .config import ModelRoleSettings, OriensConfig, ProviderSettings


class ModelError(RuntimeError):
    """可安全展示的模型错误；消息不得包含凭据或原始请求。"""


class ModelCancelled(ModelError):
    """模型任务已取消。"""


class ModelTimeout(ModelError):
    """模型请求超时。"""


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: tuple["ModelImage", ...] = ()


@dataclass(frozen=True, slots=True)
class ModelImage:
    mime_type: str
    data: bytes

    def data_url(self) -> str:
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ModelError("视觉图片格式不受支持")
        return (
            f"data:{self.mime_type};base64,"
            + base64.b64encode(self.data).decode("ascii")
        )


@dataclass(frozen=True, slots=True)
class AdapterResponse:
    content: str
    usage: ModelUsage


@dataclass(frozen=True, slots=True)
class RoutedResponse:
    content: str
    usage: ModelUsage
    model: ModelRoleSettings | None
    display_name: str
    simulated: bool


class ModelAdapter(Protocol):
    def complete(
        self,
        model: ModelRoleSettings,
        model_request: ModelRequest,
        cancel: Event,
    ) -> AdapterResponse: ...


class QwenOpenAIAdapter:
    """百炼 OpenAI 兼容 Chat Completions 适配器。"""

    def __init__(self, provider: ProviderSettings, api_key: str) -> None:
        self._provider = provider
        self._api_key = api_key

    def complete(
        self,
        model: ModelRoleSettings,
        model_request: ModelRequest,
        cancel: Event,
    ) -> AdapterResponse:
        if cancel.is_set():
            raise ModelCancelled("模型任务已取消")
        user_content: str | list[dict[str, Any]] = model_request.user_prompt
        if model_request.images:
            user_content = [{"type": "text", "text": model_request.user_prompt}]
            user_content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image.data_url()},
                }
                for image in model_request.images
            )
        body = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": model_request.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "enable_thinking": False,
        }
        # 配置的视觉角色可能不支持服务端结构化输出；结果由业务层严格校验。
        if not model_request.images:
            body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        endpoint = self._provider.base_url + "/chat/completions"
        outgoing = request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(  # noqa: S310 - configured HTTPS endpoint
                outgoing, timeout=self._provider.timeout_seconds
            ) as response:
                raw = response.read()
        except TimeoutError as exc:
            raise ModelTimeout("模型请求超时") from exc
        except error.HTTPError as exc:
            raise ModelError(f"模型服务请求失败（HTTP {exc.code}）") from None
        except error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ModelTimeout("模型请求超时") from None
            raise ModelError("无法连接模型服务，已保留离线能力") from None
        except OSError:
            raise ModelError("模型网络请求失败，已保留离线能力") from None

        if cancel.is_set():
            raise ModelCancelled("模型任务已取消")
        try:
            value = json.loads(raw)
            content = value["choices"][0]["message"]["content"]
            usage = value.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            raise ModelError("模型服务返回格式无效") from None
        if not isinstance(content, str):
            raise ModelError("模型服务未返回文本内容")
        if type(input_tokens) is not int or input_tokens < 0:
            input_tokens = 0
        if type(output_tokens) is not int or output_tokens < 0:
            output_tokens = 0
        return AdapterResponse(content, ModelUsage(input_tokens, output_tokens))


class LocalSimulationAdapter:
    """零费用的确定性模拟模型，用于默认离线运行和自动化测试。"""

    def complete(
        self,
        model: ModelRoleSettings,
        model_request: ModelRequest,
        cancel: Event,
    ) -> AdapterResponse:
        if cancel.is_set():
            raise ModelCancelled("模型任务已取消")
        metadata = model_request.metadata
        content = json.dumps(
            {
                "advice": metadata["fallback_advice"],
                "reason": metadata["fallback_reason"],
                "confidence": 0.82,
                "sources": metadata["allowed_source_ids"],
                "state_seq": metadata["state_seq"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        input_tokens = max(1, (len(model_request.system_prompt) + len(model_request.user_prompt)) // 3)
        output_tokens = max(1, len(content) // 3)
        return AdapterResponse(content, ModelUsage(input_tokens, output_tokens))


class ModelRouter:
    """按业务角色选模型；上层永远不接触具体模型 ID。"""

    def __init__(
        self,
        config: OriensConfig,
        *,
        online: bool,
        api_key: str | None,
        adapters: dict[str, ModelAdapter] | None = None,
    ) -> None:
        self._config = config
        self._online = online and bool(api_key)
        self._api_key = api_key
        self._adapters = adapters or {}
        self._simulation = LocalSimulationAdapter()

    @property
    def online(self) -> bool:
        return self._online

    def available_for(self, role: str) -> bool:
        """在线凭据可用，或测试/离线闭环显式注入了该角色适配器。"""

        return self._online or role in self._adapters

    def complete(
        self,
        role: str,
        model_request: ModelRequest,
        cancel: Event | None = None,
    ) -> RoutedResponse:
        cancel_event = cancel or Event()
        provider, model = self._config.provider_for(role)
        adapter = self._adapters.get(role)
        simulated = not self._online
        if adapter is None:
            if self._online:
                assert self._api_key is not None
                adapter = QwenOpenAIAdapter(provider, self._api_key)
            else:
                adapter = self._simulation
        elif not isinstance(adapter, LocalSimulationAdapter):
            simulated = False

        last_error: ModelError | None = None
        for attempt in range(provider.max_retries + 1):
            if cancel_event.is_set():
                raise ModelCancelled("模型任务已取消")
            try:
                response = adapter.complete(model, model_request, cancel_event)
                return RoutedResponse(
                    response.content,
                    response.usage,
                    None if simulated else model,
                    "本地模拟模型" if simulated else model.display_name,
                    simulated,
                )
            except ModelCancelled:
                raise
            except ModelError as exc:
                last_error = exc
                if attempt >= provider.max_retries:
                    break
                time.sleep(min(0.15 * (attempt + 1), 0.5))
        assert last_error is not None
        raise last_error

    def complete_offline(
        self,
        role: str,
        model_request: ModelRequest,
        cancel: Event | None = None,
    ) -> RoutedResponse:
        cancel_event = cancel or Event()
        _provider, model = self._config.provider_for(role)
        response = self._simulation.complete(model, model_request, cancel_event)
        return RoutedResponse(
            response.content,
            response.usage,
            None,
            "本地模拟模型",
            True,
        )
