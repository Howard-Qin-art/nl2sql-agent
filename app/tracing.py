"""全链路 Trace：记录每个节点的输入/输出、耗时、token 与成本。

这是面试里很吃分的一块 —— 「你线上怎么定位一次查询为什么慢/为什么答错」，
靠的就是这种结构化 trace。输出为 JSONL，一行一个 span，可直接喂给日志系统。

用法：
    tracer = Tracer(run_id="...")
    with tracer.span("schema_linking", question=q) as sp:
        ...
        sp.outputs["tables"] = [...]
    print(tracer.summary())
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    started_at: float
    duration_ms: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    # 记录本轮 span 内发生的 LLM 调用次数（节点内可能多次调模型）
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
        }


class Tracer:
    """一个 run 一条 trace；span 支持嵌套（父子关系只影响展示顺序，不参与统计）。"""

    def __init__(self, run_id: str | None = None, enabled: bool = True):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.spans: list[Span] = []
        self._lock = threading.Lock()
        self._stack: list[Span] = []

    # ---------- span ----------
    @contextmanager
    def span(self, name: str, **inputs: Any):
        sp = Span(name=name, started_at=time.perf_counter(), inputs=inputs)
        if self.enabled:
            with self._lock:
                self.spans.append(sp)
                self._stack.append(sp)
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as exc:  # 异常也要留证，方便定位
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t0) * 1000
            if self.enabled:
                with self._lock:
                    if self._stack and self._stack[-1] is sp:
                        self._stack.pop()

    def record_llm(self, prompt_tokens: int = 0, completion_tokens: int = 0, calls: int = 1) -> None:
        """把一次模型调用的用量记到当前（最内层）span 上；没有 span 就丢弃。"""
        with self._lock:
            target = self._stack[-1] if self._stack else None
        if target is None:
            return
        target.prompt_tokens += prompt_tokens
        target.completion_tokens += completion_tokens
        target.llm_calls += calls

    # ---------- 汇总 ----------
    def reset(self) -> None:
        """清空 span，用于「一题一次统计」：评测里必须逐题归零，
        否则 token/调用次数会跨题累积，指标全废。"""
        with self._lock:
            self.spans.clear()
            self._stack.clear()
        self.started_at = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.spans)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.spans)

    def summary(self, unit_prompt_price: float = 0.0, unit_completion_price: float = 0.0) -> dict[str, Any]:
        cost = (
            self.total_prompt_tokens / 1000 * unit_prompt_price
            + self.total_completion_tokens / 1000 * unit_completion_price
        )
        return {
            "run_id": self.run_id,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "llm_calls": sum(s.llm_calls for s in self.spans),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cost": round(cost, 6),
            "nodes": {s.name: round(s.duration_ms, 1) for s in self.spans},
            "errors": [{"node": s.name, "error": s.error} for s in self.spans if s.error],
        }

    def pretty(self) -> str:
        lines = [f"trace {self.run_id}  {self.elapsed_ms:.0f} ms"]
        for s in self.spans:
            flag = " !" if s.error else ""
            lines.append(
                "  %-22s %8.1f ms  llm=%-2d tok=%s/%s%s"
                % (s.name, s.duration_ms, s.llm_calls, s.prompt_tokens, s.completion_tokens, flag)
            )
        return "\n".join(lines)

    def to_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for s in self.spans:
                row = {"run_id": self.run_id, **s.to_dict()}
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path
