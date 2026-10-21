# -*- coding: utf-8 -*-
"""工具调用循环（Plan B）：让模型自主决定"调哪个工具、调几次"，而不是走固定节点。

和现有 LangGraph 流水线的区别：
    流水线（app/agent/core.py）：开发者规定顺序（先召回→再生成→再校验…），模型只负责写 SQL。
    工具循环（本文件）：只把工具和规则交给模型，由它自己决定先看库表、再查数、还是直接回答；
                        循环直到模型不再调用工具为止。

这一层不绑定具体工具来源：
    - `local_tools()`：直接用本项目的 list_tables / get_schema / run_sql（不依赖子进程）
    - `app/agent/mcp_client.py`：通过 MCP 协议连外部 server，动态发现工具
所以同一个循环既能跑本地工具，也能跑任何 MCP server 的工具。

安全边界不变：run_sql 内部仍然走 sqlglot 白名单 + 只读连接 + 补 LIMIT。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from app.config import SETTINGS, Settings
from app.db.executor import SQLExecutor
from app.db.schema import introspect, schema_prompt
from app.guardrails import validate_sql
from app.llm import LLMClient
from app.tracing import Tracer

SYSTEM_PROMPT = """你是一个数据查询助手，可以调用工具去查数据库，最后用中文回答用户的问题。

【工具使用原则】
1. 不确定表结构时，先调 get_schema 看列名、类型、业务注释；不要凭空猜列名。
2. 查明细或聚合都可以用 run_sql，但只允许 SELECT；系统会自动补 LIMIT、拒绝写操作。
3. 一个问题可能需要多轮：先看结构 -> 再查数 -> 需要再查就再调一次。
4. 已经能回答时，直接给出结论（别再调工具），结论里要带关键数字。
5. 如果问题有歧义（没说什么算"最近"、指标不明），先用一句话向用户确认，不要瞎猜口径。

【业务口径】
- 成交额/GMV：orders.status = 'completed' 的 total_amount 之和；
- 客单价：SUM(total_amount) / COUNT(*)（仅 completed）；
- 活跃客户：customer.is_active = 1（只有问题问"活跃"时才加）；
- 在售商品：product.status = 'on_sale'（只有问"在售"时才加）。
"""


@dataclass
class ToolSpec:
    """一个工具：名字 + 说明 + JSON Schema 参数 + 同步可调用对象。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    output: str
    latency_ms: float
    ok: bool = True

    def brief(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.name,
            "arguments": self.arguments,
            "latency_ms": round(self.latency_ms, 1),
            "output_preview": self.output[:160],
            "ok": self.ok,
        }


@dataclass
class ToolRunResult:
    question: str
    ok: bool = False
    answer: str = ""
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sql: str = ""
    rows: list[list[Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None
    latency_ms: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "answer": self.answer,
            "steps": self.steps,
            "tool_calls": [c.brief() for c in self.tool_calls],
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "trace": self.trace,
        }


# ---------------------------------------------------------------- 本地工具
def local_tools(settings: Settings | None = None) -> list[ToolSpec]:
    """本项目自带的三个只读工具（复用 db/schema、db/executor、guardrails）。"""
    s = settings or SETTINGS
    docs = introspect(s.db_path)
    allowed = {d.name for d in docs}
    executor = SQLExecutor(s.db_path, timeout_s=s.query_timeout_s, max_rows=s.max_rows)
    by_name = {d.name: d for d in docs}

    def list_tables() -> str:
        return json.dumps(
            [{"name": d.name, "rows": d.row_count, "note": d.note} for d in docs], ensure_ascii=False)

    def get_schema(tables: list[str] | None = None) -> str:
        wanted = [t for t in (tables or []) if t in by_name]
        if tables and not wanted:
            return json.dumps({"error": f"这些表不存在：{tables}", "available": sorted(by_name)}, ensure_ascii=False)
        return schema_prompt([by_name[t] for t in wanted] if wanted else docs)

    def run_sql(sql: str) -> str:
        guard = validate_sql(sql, default_limit=s.default_limit, allowed_tables=allowed)
        if not guard.ok:
            return json.dumps({"error": "SQL 被安全护栏拒绝", "reasons": guard.errors}, ensure_ascii=False)
        res = executor.run(guard.sql)
        if not res.ok:
            return json.dumps({"error": res.error, "sql": guard.sql}, ensure_ascii=False)
        return json.dumps({
            "sql": guard.sql,
            "columns": res.columns,
            "rows": [list(r) for r in res.rows[:50]],
            "row_count": res.row_count,
            "truncated": res.truncated,
        }, ensure_ascii=False, default=str)

    return [
        ToolSpec(
            name="list_tables",
            description="列出数据库里的表和行数，用于先了解库里有什么",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=list_tables,
        ),
        ToolSpec(
            name="get_schema",
            description="查看表结构：列名、类型、外键、抽样值和业务注释。不传 tables 则返回全部表",
            parameters={
                "type": "object",
                "properties": {
                    "tables": {"type": "array", "items": {"type": "string"},
                               "description": "要查看的表名列表，例如 [\"orders\"]"},
                },
                "additionalProperties": False,
            },
            func=get_schema,
        ),
        ToolSpec(
            name="run_sql",
            description="执行一条只读 SELECT 查询（自动补 LIMIT，禁止写操作）；返回列名和行数据",
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "SQLite 语法的 SELECT 语句"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
            func=run_sql,
        ),
    ]


# ---------------------------------------------------------------- 循环
def run_tool_agent(
    question: str,
    *,
    llm: LLMClient,
    tools: Sequence[ToolSpec],
    settings: Settings | None = None,
    max_steps: int = 6,
    temperature: float = 0.1,
    tracer: Tracer | None = None,
) -> ToolRunResult:
    """让模型自己调工具，直到它给出最终回答 / 步数用完 / 超预算。"""
    s = settings or SETTINGS
    tracer = tracer or Tracer(enabled=False)
    result = ToolRunResult(question=question)
    t0 = time.perf_counter()
    deadline = t0 + s.request_budget_s

    by_name = {t.name: t for t in tools}
    schemas = [t.to_openai() for t in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    with tracer.span("tool_agent", question=question[:200], tools=[t.name for t in tools]) as sp:
        for step in range(1, max_steps + 1):
            if time.perf_counter() > deadline:
                result.error = f"超过时间预算 {s.request_budget_s:.0f}s"
                break
            result.steps = step
            res = llm.chat(messages, tools=schemas, temperature=temperature)
            if not res.tool_calls:
                result.answer = (res.text or "").strip()
                result.ok = bool(result.answer)
                break

            # 模型这轮选择调工具：先把 assistant 消息（含 tool_calls）写回历史
            messages.append({
                "role": "assistant",
                "content": res.text or "",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in res.tool_calls
                ],
            })
            for call in res.tool_calls:
                name, args = call["name"], call["arguments"]
                spec = by_name.get(name)
                call_t0 = time.perf_counter()
                if spec is None:
                    output = json.dumps({"error": f"没有这个工具：{name}", "available": sorted(by_name)}, ensure_ascii=False)
                    ok = False
                else:
                    try:
                        # 只传工具声明里存在的参数，防止模型多给字段导致 TypeError
                        kwargs = {k: v for k, v in args.items()
                                  if k in (spec.parameters.get("properties") or {})}
                        output = str(spec.func(**kwargs))
                        ok = True
                    except Exception as exc:  # 工具内部报错要回传给模型，让它自己改
                        output = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
                        ok = False
                record = ToolCallRecord(step=step, name=name, arguments=args, output=output,
                                        latency_ms=(time.perf_counter() - call_t0) * 1000, ok=ok)
                result.tool_calls.append(record)
                if name == "run_sql" and ok:
                    try:
                        payload = json.loads(output)
                        result.sql = payload.get("sql", result.sql)
                        if payload.get("columns"):
                            result.columns = payload["columns"]
                            result.rows = payload.get("rows") or []
                            result.row_count = payload.get("row_count", 0)
                    except json.JSONDecodeError:
                        pass
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": output})
        else:
            result.error = f"达到最大步数 {max_steps}，仍未给出结论"

        sp.outputs["steps"] = result.steps
        sp.outputs["tools_called"] = [c.name for c in result.tool_calls]
        sp.outputs["ok"] = result.ok

    if not result.ok and not result.error:
        result.error = "模型没有给出最终回答"
    result.latency_ms = (time.perf_counter() - t0) * 1000
    result.trace = tracer.summary()
    return result
