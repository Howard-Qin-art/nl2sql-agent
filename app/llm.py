"""LLM 双后端抽象。

为什么不用 LangChain 的 ChatOpenAI？
    - 本机 Ollama 的 OpenAI 兼容端点无法传 `num_gpu` / `num_ctx` 这类原生参数，
      而本机 GPU 与 Ollama 的 CUDA 内核不兼容，必须能强制 CPU 推理并做自动降级；
    - Ollama 原生 /api/chat 直接返回 token 计数与耗时，做 trace/成本统计更准。
于是：Ollama 走原生 API，云端走 OpenAI 兼容 API，两者统一成同一个 `LLMClient.chat()` 接口。

面试可讲点：
    1. 后端可插拔 + 抽象边界（消息、结构化输出、用量统计、重试）；
    2. 结构化输出用 `format=json` / `response_format=json_object`，再叠加 pydantic 校验 +
       一次「带错误信息的自修复重试」，而不是拿正则从自然语言里抠 JSON；
    3. GPU 崩溃自动降级 CPU 并记忆，避免整条链路挂在一个环境问题上。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import requests
from pydantic import BaseModel, ValidationError

from app.config import SETTINGS, Settings

Message = dict[str, str]


# ---------------------------------------------------------------- 结果与工具
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    backend: str = ""
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
        }


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里稳当地取出 JSON 对象：先剥 ``` 围栏，再取第一个平衡的 {...}。"""
    if not text:
        raise ValueError("模型返回为空")
    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 退一步：扫描出第一个括号平衡的 JSON 片段
    start = candidate.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)
    raise ValueError("无法从模型输出中解析 JSON: " + text[:200])


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- 后端
class BaseBackend:
    name = "base"

    def __init__(self, settings: Settings):
        self.s = settings

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        raise NotImplementedError


class OllamaBackend(BaseBackend):
    """Ollama 原生 /api/chat：能控制 num_gpu / num_ctx，返回真实 token 计数。"""

    name = "ollama"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.num_gpu = settings.ollama_num_gpu
        self._degraded = False

    def _payload(
        self,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        num_gpu: int,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.s.ollama_num_ctx,
            "num_gpu": num_gpu,
        }
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": self.s.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
            "keep_alive": self.s.ollama_keep_alive,
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        temperature = self.s.temperature if temperature is None else temperature
        max_tokens = self.s.max_tokens if max_tokens is None else max_tokens
        url = f"{self.s.ollama_base}/api/chat"
        last_err: Exception | None = None

        for attempt in range(1, self.s.max_retries + 2):
            body = self._payload(messages, json_mode, temperature, max_tokens, seed, self.num_gpu)
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=body, timeout=self.s.request_timeout)
                if resp.status_code >= 400:
                    text = resp.text
                    # GPU 内核不兼容 -> 自动降级 CPU 再试一次（真实踩过的坑）
                    if self.num_gpu != 0 and _looks_like_gpu_failure(text):
                        self.num_gpu = 0
                        self._degraded = True
                        continue
                    raise LLMError(f"Ollama HTTP {resp.status_code}: {text[:300]}")
                data = resp.json()
                content = (data.get("message") or {}).get("content", "")
                return LLMResult(
                    text=content,
                    prompt_tokens=int(data.get("prompt_eval_count") or 0),
                    completion_tokens=int(data.get("eval_count") or 0),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    model=self.s.model,
                    backend=self.name,
                    attempts=attempt,
                    raw=data,
                )
            except Exception as exc:  # 网络/解析/HTTP 错误统一重试
                last_err = exc
                if attempt <= self.s.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                break
        raise LLMError(f"Ollama 调用失败（已降级 num_gpu={self.num_gpu}）: {last_err}")


def _looks_like_gpu_failure(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ("cuda error", "device kernel image is invalid", "llama-server process has terminated"))


class OpenAIBackend(BaseBackend):
    """OpenAI 兼容后端：OpenAI / DeepSeek / 阿里云百炼（DashScope compatible-mode）等，改 base_url 即可。

    实测踩坑（阿里云百炼）：
        1. 百炼官方兼容参数清单里**没有 `response_format`**，而它在 JSON 模式下会 400；
           所以这里做了自动回退：一旦发现是不支持 response_format 导致的失败，
           就永久去掉这个参数，改用「纯提示词要求 JSON + 本地解析」。
        2. 百炼文档建议 temperature **不要正好取 0**（范围 [0,2)），所以云端给一个下限。
        3. API Key 与地域绑定：北京 key 调新加坡 endpoint 会返回 401 invalid_api_key，
           报错信息里会带上这个提示，省得排查半天。
    """

    name = "openai"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from openai import OpenAI  # 延迟导入：没装也不影响用 Ollama

        if not settings.openai_key:
            raise LLMError("使用 openai 后端需要在 .env 里配置 NL2SQL_OPENAI_KEY / OPENAI_API_KEY")
        self.client = OpenAI(base_url=settings.openai_base, api_key=settings.openai_key, timeout=settings.request_timeout)
        self._json_mode_supported = True   # 探到不支持就置 False，后续不再带 response_format
        self.last_json_mode_error: str | None = None

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        temperature = self.s.temperature if temperature is None else temperature
        # 云端下限：百炼明确"不建议取 0"
        temperature = max(temperature, self.s.min_temperature)
        max_tokens = self.s.max_tokens if max_tokens is None else max_tokens
        kwargs: dict[str, Any] = {
            "model": self.s.openai_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = seed

        last_err: Exception | None = None
        attempts = 0
        while attempts <= self.s.max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=resp.choices[0].message.content or "",
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    model=self.s.openai_model,
                    backend=self.name,
                    attempts=attempts,
                )
            except Exception as exc:
                last_err = exc
                text = str(exc)
                # 兼容回退 1：该服务不支持 response_format -> 去掉它再试（不算重试次数）
                if "response_format" in kwargs and _looks_like_unsupported_param(text, "response_format"):
                    self._json_mode_supported = False
                    self.last_json_mode_error = text[:200]
                    kwargs.pop("response_format", None)
                    attempts -= 1
                    continue
                # 兼容回退 2：不支持 seed 就去掉
                if "seed" in kwargs and _looks_like_unsupported_param(text, "seed"):
                    kwargs.pop("seed", None)
                    attempts -= 1
                    continue
                if attempts <= self.s.max_retries:
                    time.sleep(min(2 ** (attempts - 1), 4))
                    continue
                break
        raise LLMError(f"OpenAI 兼容后端调用失败：{_cloud_error_hint(str(last_err))}")


def _looks_like_unsupported_param(error_text: str, param: str) -> bool:
    lowered = error_text.lower()
    return param.lower() in lowered and any(k in lowered for k in
                                            ("unsupported", "not support", "unknown", "invalid", "unexpected", "does not support"))


def _cloud_error_hint(error_text: str) -> str:
    """把云端的报错翻译成能直接行动的中文提示。"""
    lowered = error_text.lower()
    if "invalid_api_key" in lowered or "invalid api-key" in lowered or "401" in lowered:
        return (f"{error_text[:300]}\n  提示：百炼的 API Key **按地域绑定** —— "
                "北京 key 调新加坡 endpoint 就会报 invalid_api_key，请确认 key 与 base_url 同地域。")
    if "model" in lowered and ("not exist" in lowered or "not found" in lowered or "unsupported" in lowered):
        return f"{error_text[:300]}\n  提示：模型名可能没开通或写错，去百炼控制台确认已开通的模型名。"
    if "429" in lowered:
        return f"{error_text[:300]}\n  提示：触发限流或额度不足，稍后重试或检查配额。"
    if "503" in lowered or "overloaded" in lowered:
        return f"{error_text[:300]}\n  提示：服务端过载，稍后重试。"
    return error_text[:400]


class FakeBackend(BaseBackend):
    """离线后端：给单元测试和 CI 用（不联网也能端到端跑通 Agent）。

    传一个 callable(messages) -> str，或者一组按顺序消费的响应文本。
    """

    name = "fake"

    def __init__(self, settings: Settings, responder: Callable[[Sequence[Message]], str] | None = None,
                 responses: Iterable[str] | None = None):
        super().__init__(settings)
        self.responder = responder
        self.queue = list(responses or [])
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if self.responder is not None:
            text = self.responder(messages)
        elif self.queue:
            text = self.queue.pop(0)
        else:
            raise LLMError("FakeBackend 没有可用的响应了")
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text) // 4,
            latency_ms=1.0,
            model="fake",
            backend=self.name,
        )


# ---------------------------------------------------------------- 门面
class LLMClient:
    """统一入口：`chat()` 文本，`chat_json()` 结构化 + pydantic 校验 + 自修复重试。"""

    def __init__(self, settings: Settings | None = None, backend: BaseBackend | None = None, tracer=None):
        self.s = settings or SETTINGS
        self.tracer = tracer
        if backend is not None:
            self.backend = backend
        elif self.s.llm_backend == "openai":
            self.backend = OpenAIBackend(self.s)
        elif self.s.llm_backend == "fake":
            self.backend = FakeBackend(self.s)
        else:
            self.backend = OllamaBackend(self.s)

    # 便捷构造：测试用
    @classmethod
    def fake(cls, responses: Iterable[str] | None = None,
             responder: Callable[[Sequence[Message]], str] | None = None,
             settings: Settings | None = None) -> "LLMClient":
        s = settings or SETTINGS
        return cls(settings=s, backend=FakeBackend(s, responder=responder, responses=responses))

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResult:
        result = self.backend.chat(messages, **kwargs)
        if self.tracer is not None:
            self.tracer.record_llm(result.prompt_tokens, result.completion_tokens)
        return result

    def chat_json(
        self,
        messages: Sequence[Message],
        *,
        model_cls: type[BaseModel] | None = None,
        repair_attempts: int = 1,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], LLMResult]:
        convo: list[Message] = list(messages)
        last_error: str = ""
        result: LLMResult | None = None

        for attempt in range(repair_attempts + 1):
            result = self.chat(convo, json_mode=True, **kwargs)
            try:
                data = extract_json(result.text)
                if model_cls is not None:
                    data = model_cls.model_validate(data).model_dump()
                return data, result
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:400]
                if attempt >= repair_attempts:
                    break
                # 自修复：把上一次的输出和校验错误一起回注，让模型只修 JSON
                convo = list(messages) + [
                    {"role": "assistant", "content": result.text[:1500]},
                    {
                        "role": "user",
                        "content": (
                            "上面的输出不是合法/合规的 JSON。错误信息：\n"
                            f"{last_error}\n\n"
                            "请只输出修复后的 JSON 对象本身，不要解释、不要 markdown 代码块。"
                        ),
                    },
                ]
        raise LLMError(f"结构化输出解析失败: {last_error}")


def build_llm(settings: Settings | None = None, tracer=None) -> LLMClient:
    return LLMClient(settings=settings, tracer=tracer)
