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
    # 工具调用（function calling）：统一成 [{"id","name","arguments"}]
    # 不同后端格式不一样（Ollama 原生没有 id、OpenAI 协议 id 必填），在这里归一化
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

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
        tools: Sequence[dict[str, Any]] | None = None,
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
        tools: Sequence[dict[str, Any]] | None = None,
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
            "messages": _normalize_messages_for_ollama(messages),
            "stream": False,
            "options": options,
            "keep_alive": self.s.ollama_keep_alive,
        }
        if json_mode:
            payload["format"] = "json"
        if tools:
            payload["tools"] = list(tools)
        return payload

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResult:
        temperature = self.s.temperature if temperature is None else temperature
        max_tokens = self.s.max_tokens if max_tokens is None else max_tokens
        url = f"{self.s.ollama_base}/api/chat"
        last_err: Exception | None = None

        for attempt in range(1, self.s.max_retries + 2):
            body = self._payload(messages, json_mode, temperature, max_tokens, seed, self.num_gpu, tools)
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
                    # 工具调用参数被模型写成残缺 JSON：Ollama 会返回 400 且报
                    # "Value looks like object, but can't find closing '}' symbol"。
                    # 这是小模型输出质量问题，重试一次通常就好了，不该直接放弃。
                    if tools and _looks_like_tool_json_error(text):
                        last_err = LLMError(f"模型输出的工具参数不是合法 JSON：{text[:200]}")
                        if attempt <= self.s.max_retries:
                            time.sleep(0.5)
                            continue
                        raise last_err
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
                    tool_calls=_normalize_tool_calls((data.get("message") or {}).get("tool_calls")),
                )
            except Exception as exc:  # 网络/解析/HTTP 错误统一重试
                last_err = exc
                if attempt <= self.s.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                break
        raise LLMError(f"Ollama 调用失败（已降级 num_gpu={self.num_gpu}）: {last_err}")


def _normalize_messages_for_ollama(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """把 OpenAI 风格的历史消息转成 Ollama 原生能吃的格式。

    关键差异（真实踩坑）：多轮工具调用时，历史里的 assistant 消息要带回 tool_calls。
        OpenAI 协议：function.arguments 是 **JSON 字符串**
        Ollama 原生：function.arguments 必须是 **对象**（Go 侧强类型 map）
    发字符串给 Ollama 会 400：`Value looks like object, but can't find closing '}' symbol`。
    所以在发出前只做这一层转换，上层（工具循环）保持协议中立。
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = dict(msg)
        calls = item.get("tool_calls")
        if calls:
            converted = []
            for call in calls:
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                new_call = {k: v for k, v in call.items() if k != "function"}
                new_call["function"] = fn
                converted.append(new_call)
            item["tool_calls"] = converted
        # Ollama 的 tool 消息只认 role/content；tool_call_id 与 name 去掉更稳
        if item.get("role") == "tool":
            item = {"role": "tool", "content": item.get("content", "")}
        out.append(item)
    return out


def _looks_like_gpu_failure(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ("cuda error", "device kernel image is invalid", "llama-server process has terminated"))


def _looks_like_tool_json_error(text: str) -> bool:
    """Ollama 解析模型生成的工具参数失败时的报错特征（可重试）。"""
    lowered = text.lower()
    return ("can't find closing" in lowered
            or "value looks like object" in lowered
            or ("tool" in lowered and "json" in lowered and "error" in lowered))


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
        tools: Sequence[dict[str, Any]] | None = None,
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
        if tools:
            kwargs["tools"] = list(tools)

        last_err: Exception | None = None
        attempts = 0
        while attempts <= self.s.max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                msg = resp.choices[0].message
                return LLMResult(
                    text=msg.content or "",
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    model=self.s.openai_model,
                    backend=self.name,
                    attempts=attempts,
                    tool_calls=_normalize_tool_calls(getattr(msg, "tool_calls", None)),
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
                # 兼容回退 3：不支持 tools 就报清楚（不静默降级，否则工具调用会假成功）
                if "tools" in kwargs and _looks_like_unsupported_param(text, "tools"):
                    raise LLMError(f"该模型不支持工具调用（tools）：{text[:200]}")
                if attempts <= self.s.max_retries:
                    time.sleep(min(2 ** (attempts - 1), 4))
                    continue
                break
        raise LLMError(f"OpenAI 兼容后端调用失败：{_cloud_error_hint(str(last_err))}")


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """把不同后端的工具调用归一化成 [{"id","name","arguments"}]。

    - OpenAI 协议：tool_calls[i] = {id, type:"function", function:{name, arguments:"json字符串"}}
    - Ollama 原生：tool_calls[i] = {function:{name, arguments:{dict}}}（通常没有 id）
    统一补 id、把 arguments 解析成 dict，失败则给 {}（让工具层报参数错，而不是崩）。
    """
    calls: list[dict[str, Any]] = []
    if not raw:
        return calls
    if isinstance(raw, dict):
        raw = [raw]
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or item
        name = fn.get("name") or item.get("name")
        if not name:
            continue
        args = fn.get("arguments", item.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"id": item.get("id") or f"call_{i}", "name": str(name), "arguments": args})
    return calls


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
                 responses: Iterable[str] | None = None,
                 tool_script: Iterable[dict[str, Any]] | None = None):
        super().__init__(settings)
        self.responder = responder
        self.queue = list(responses or [])
        # 工具调用脚本：每项是一轮模型响应，形如
        #   {"tool_calls": [{"name": "get_schema", "arguments": {"tables": ["orders"]}}]}
        #   {"text": "最终回答……"}
        # 用于离线测「模型自主选工具」的循环，不需要真模型。
        self.tool_script = list(tool_script or [])
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        tool_calls: list[dict[str, Any]] = []
        text = ""
        if self.tool_script:
            step = self.tool_script.pop(0)
            if isinstance(step, str):
                text = step
            else:
                text = step.get("text", "") or ""
                tool_calls = _normalize_tool_calls(step.get("tool_calls"))
        elif self.responder is not None:
            text = self.responder(messages)
        elif self.queue:
            text = self.queue.pop(0)
        else:
            raise LLMError("FakeBackend 没有可用的响应了")
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text) // 4,
            latency_ms=1.0,
            model="fake",
            backend=self.name,
            tool_calls=tool_calls,
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
            exchange = None
            # 只有显式打开 NL2SQL_TRACE_LLM 才记录原始请求/响应（否则 trace 文件会明显变大）
            if getattr(self.s, "trace_llm", False) and getattr(self.tracer, "enabled", False):
                exchange = {
                    "model": result.model,
                    "messages": list(messages),
                    "response": result.text,
                    "tool_calls": result.tool_calls,
                }
            self.tracer.record_llm(result.prompt_tokens, result.completion_tokens, exchange=exchange)
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
