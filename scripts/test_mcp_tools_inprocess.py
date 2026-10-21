# -*- coding: utf-8 -*-
"""进程内测试：直接调用 MCP Server 的工具处理器，验证工具逻辑（不需要子进程/管道）。

这样即使沙箱不允许创建命名管道，也能验证四个工具的入参、出参和护栏行为。

用法：python scripts/test_mcp_tools_inprocess.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mcp_server  # noqa: E402

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  -> {detail}" if detail and not ok else ""))


def text_of(result) -> str:
    return "\n".join(getattr(b, "text", "") or "" for b in (result.content or []))


async def call(name: str, args: dict) -> dict:
    raw = text_of(await mcp_server.server.call_tool(name, args))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


async def main() -> int:
    tools = await mcp_server.server.list_tools()
    names = [t.name for t in tools]
    print(f"工具列表：{names}")
    check("注册了 4 个工具", len(tools) == 4, str(names))
    for expected in ("list_tables", "get_schema", "run_sql", "ask"):
        check(f"存在 {expected}", expected in names, str(names))

    data = await call("list_tables", {})
    tables = data.get("tables", [])
    print(f"\n[list_tables] ok={data.get('ok')} 表数={len(tables)} 示例={[t['name'] for t in tables[:4]]}")
    check("list_tables 返回 9 张表", data.get("ok") and len(tables) == 9, str(data)[:160])
    check("list_tables 带行数与说明", bool(tables) and tables[0].get("rows", 0) > 0 and bool(tables[0].get("note")))

    data = await call("get_schema", {"tables": ["orders", "refund"]})
    schema = data.get("schema_text", "")
    print(f"\n[get_schema] 长度={len(schema)} 含 orders={'TABLE orders' in schema} 含 refund={'TABLE refund' in schema}")
    check("get_schema 只返回指定表", data.get("ok") and "TABLE orders" in schema and "TABLE refund" in schema
          and "TABLE customer" not in schema)
    bad = await call("get_schema", {"tables": ["not_a_table"]})
    check("get_schema 对不存在的表报错", bad.get("ok") is False and "not_a_table" in str(bad.get("error")), str(bad)[:120])

    data = await call("run_sql", {"sql": "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status"})
    print(f"\n[run_sql] ok={data.get('ok')} 行数={data.get('row_count')} 列={data.get('columns')}")
    print(f"         sql={str(data.get('sql'))[:80]}")
    check("run_sql 正常执行并返回行", data.get("ok") and data.get("row_count", 0) > 0)
    check("run_sql 自动补了 LIMIT", "LIMIT" in str(data.get("sql", "")).upper())

    data = await call("run_sql", {"sql": "DROP TABLE orders"})
    print(f"\n[run_sql 危险] ok={data.get('ok')} 理由={data.get('reasons')}")
    check("拦下 DROP", data.get("ok") is False)

    data = await call("run_sql", {"sql": "SELECT 1; DELETE FROM refund;"})
    check("拦下多语句注入", data.get("ok") is False, str(data)[:120])

    data = await call("run_sql", {"sql": "SELECT * FROM sqlite_master"})
    check("拦下系统表查询", data.get("ok") is False, str(data)[:120])

    data = await call("run_sql", {"sql": "SELECT * FROM not_a_table"})
    check("拦下不存在的表", data.get("ok") is False, str(data)[:120])

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 70)
    print(f"进程内工具测试：{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
