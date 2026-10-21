# -*- coding: utf-8 -*-
"""MCP Server：把 NL2SQL 的能力暴露成标准 MCP 工具。

传输：stdio（由 MCP 客户端启动本进程，比如 Claude Desktop / Cursor / 自研客户端）

四个工具：
    list_tables  列出库里的表、行数、业务说明
    get_schema   返回指定表的结构（列/类型/外键/抽样值/业务注释），不传表名就返回全部
    run_sql      只读执行 SQL —— 复用现有护栏：sqlglot 白名单 + 自动补 LIMIT + 超时中断
    ask          用中文提问，走完整 Agent（召回→生成候选→投票→自愈→结论）

约定：
    - 所有工具返回 JSON 字符串（模型和人都能读）；
    - 只读是硬约束：run_sql 走 guardrails，连接层再以 mode=ro 打开；
    - 任何异常都转成 {"ok": false, "error": "..."}，不让进程崩掉；
    - 绝不往 stdout 打印（那是 MCP 协议通道），调试信息走 stderr。

启动（手动调试）：
    python mcp_server.py
客户端配置示例（Cursor / Claude Desktop 的 mcp.json / claude_desktop_config.json）：
    {
      "mcpServers": {
        "nl2sql": {
          "command": "C:\\\\Users\\\\22855\\\\miniconda3\\\\envs\\\\llm\\\\python.exe",
          "args": ["E:\\\\dddddd\\\\nl2sql-agent\\\\mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 以本文件所在目录为项目根，保证被任何客户端以任意工作目录启动都能 import 到 app 包
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.config import SETTINGS  # noqa: E402
from app.db.executor import SQLExecutor  # noqa: E402
from app.db.schema import introspect, schema_prompt  # noqa: E402
from app.guardrails import validate_sql  # noqa: E402

server = MCPServer(
    name="nl2sql-agent",
    title="NL2SQL 智能查询 Agent",
    description="把中文业务问题翻译成 SQL，并以只读方式查询电商订单库",
    instructions=(
        "这是一个只读的电商订单数据查询服务。"
        "建议流程：先用 list_tables / get_schema 了解表结构，"
        "再用 ask 提中文问题（会自动生成 SQL 并给结论），或直接用 run_sql 执行只读查询。"
        "所有写操作、多语句、系统表访问都会被拒绝。"
    ),
    version="0.2.0",
)

_docs = None
_allowed_tables: set[str] = set()


def _load_docs():
    """惰性加载表画像：被客户端调用时才读库，避免启动就慢。"""
    global _docs, _allowed_tables
    if _docs is None:
        _docs = introspect(SETTINGS.db_path)
        _allowed_tables = {d.name for d in _docs}
    return _docs


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


@server.tool(name="list_tables", description="列出数据库中的所有表、行数和一句话业务说明", structured_output=False)
def list_tables() -> str:
    docs = _load_docs()
    return _json({
        "ok": True,
        "tables": [{"name": d.name, "rows": d.row_count, "note": d.note} for d in docs],
    })


@server.tool(
    name="get_schema",
    description="返回指定表的结构（列名/类型/外键/空值率/抽样值/业务注释）；tables 不传或为空则返回全部表",
    structured_output=False,
)
def get_schema(tables: list[str] | None = None) -> str:
    docs = _load_docs()
    by_name = {d.name: d for d in docs}
    wanted = [t.strip() for t in (tables or []) if t and t.strip()]
    if wanted:
        missing = [t for t in wanted if t not in by_name]
        if missing:
            return _json({"ok": False, "error": f"表不存在：{', '.join(missing)}",
                          "available": sorted(by_name)})
        chosen = [by_name[t] for t in wanted]
    else:
        chosen = docs
    return _json({"ok": True, "schema_text": schema_prompt(chosen)})


@server.tool(
    name="run_sql",
    description="执行一条只读 SELECT 查询（会自动补 LIMIT；禁止写操作/多语句/系统表）",
    structured_output=False,
)
def run_sql(sql: str, limit: int = 100) -> str:
    _load_docs()
    guard = validate_sql(sql, default_limit=limit, allowed_tables=_allowed_tables)
    if not guard.ok:
        return _json({"ok": False, "error": "SQL 被安全护栏拒绝", "reasons": guard.errors})
    executor = SQLExecutor(SETTINGS.db_path, timeout_s=SETTINGS.query_timeout_s, max_rows=SETTINGS.max_rows)
    res = executor.run(guard.sql)
    if not res.ok:
        return _json({"ok": False, "error": res.error, "sql": guard.sql})
    return _json({
        "ok": True,
        "sql": guard.sql,
        "columns": res.columns,
        "rows": [list(r) for r in res.rows],
        "row_count": res.row_count,
        "truncated": res.truncated,
        "latency_ms": round(res.latency_ms, 1),
        "notes": guard.warnings,
    })


@server.tool(
    name="ask",
    description="用中文提问，由 NL2SQL Agent 自动生成 SQL、查询并给出结论（需要本地 Ollama 或云端模型可用）",
    structured_output=False,
)
def ask(question: str) -> str:
    _load_docs()
    try:
        # 延迟导入：只有真正调用 ask 时才初始化 Agent（避免客户端只做 list/schema 时付出初始化成本）
        from app.agent.core import NL2SQLAgent
        from app.llm import LLMClient
        from app.tracing import Tracer

        tracer = Tracer(enabled=False)
        agent = NL2SQLAgent(settings=SETTINGS, llm=LLMClient(settings=SETTINGS, tracer=tracer),
                            docs=_docs, tracer=tracer, use_summary=False)
        res = agent.run(question)
        return _json({
            "ok": res.ok,
            "question": question,
            "sql": res.sql,
            "linked_tables": res.linked_tables,
            "columns": res.columns,
            "rows": [list(r) for r in res.rows[:50]],
            "row_count": res.row_count,
            "answer": res.answer,
            "repair_rounds": res.repair_rounds,
            "error": res.error,
            "trace": res.trace,
        })
    except Exception as exc:  # 模型不可用（比如 Ollama 没启动）也要给客户端一个可读的答复
        return _json({"ok": False, "question": question,
                      "error": f"{type(exc).__name__}: {exc}",
                      "hint": "如果 backend=ollama，请确认本地 Ollama 已启动；也可以在 .env 里切到云端后端。"})


def main() -> None:
    # stdio 传输：stdin/stdout 就是协议通道，不要 print 任何东西
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
