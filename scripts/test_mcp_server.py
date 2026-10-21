# -*- coding: utf-8 -*-
"""协议级测试：像真实 MCP 客户端那样拉起 mcp_server.py，走 握手 → tools/list → tools/call。

用法：
    python scripts/test_mcp_server.py              # 跑全部工具
    python scripts/test_mcp_server.py --skip-ask   # 跳过需要 LLM 的 ask 工具

它验证的是「协议层」是否通：initialize 握手、工具列表、参数 schema、逐个工具调用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "mcp_server.py"

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  -> {detail}" if detail and not ok else ""))


def text_of(result) -> str:
    """从 CallToolResult 里取出纯文本（v2 的 content 是 TextContent 列表）。"""
    parts = []
    for block in (result.content or []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def main(skip_ask: bool) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        cwd=str(ROOT),
    )
    print(f"启动服务器：{sys.executable} {SERVER}")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # ---- 1. 握手 ----
            try:
                init = await session.initialize()
            except TypeError:  # 某些版本要求显式传 client_info
                from mcp import types as mtypes
                init = await session.initialize(
                    client_info=mtypes.Implementation(name="nl2sql-test-client", version="0.1"))
            info = init.server_info
            print(f"\n[握手] 协议版本={init.protocol_version} 服务器={getattr(info, 'name', '?')} "
                  f"版本={getattr(info, 'version', '?')}")
            check("initialize 握手", bool(init.protocol_version), str(init.protocol_version))
            check("服务器名字正确", getattr(info, "name", "") == "nl2sql-agent", str(info))

            # ---- 2. 工具列表 ----
            tools = (await session.list_tools()).tools
            names = [t.name for t in tools]
            print(f"\n[工具列表] {names}")
            check("工具数量 = 4", len(tools) == 4, str(names))
            for expected in ("list_tables", "get_schema", "run_sql", "ask"):
                check(f"存在工具 {expected}", expected in names, str(names))
            run_sql_tool = next((t for t in tools if t.name == "run_sql"), None)
            check("run_sql 的参数 schema 含 sql", bool(run_sql_tool and "sql" in (run_sql_tool.input_schema or {}).get("properties", {})),
                  json.dumps(getattr(run_sql_tool, "input_schema", {}), ensure_ascii=False)[:200])

            # ---- 3. list_tables ----
            r = await session.call_tool("list_tables", {})
            data = json.loads(text_of(r))
            print(f"\n[list_tables] ok={data.get('ok')} 表数={len(data.get('tables', []))} "
                  f"示例={[t['name'] for t in data.get('tables', [])][:4]}")
            check("list_tables 返回 9 张表", data.get("ok") and len(data.get("tables", [])) == 9,
                  str(data)[:200])

            # ---- 4. get_schema ----
            r = await session.call_tool("get_schema", {"tables": ["orders", "refund"]})
            data = json.loads(text_of(r))
            schema_text = data.get("schema_text", "")
            print(f"\n[get_schema] 返回 {len(schema_text)} 字符，含 orders={'TABLE orders' in schema_text} "
                  f"含 refund={'TABLE refund' in schema_text}")
            check("get_schema 命中指定表", data.get("ok") and "TABLE orders" in schema_text and "TABLE refund" in schema_text)
            check("get_schema 不夹带其它表", "TABLE customer" not in schema_text)

            # ---- 5. run_sql：正常查询 ----
            r = await session.call_tool("run_sql", {"sql": "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status"})
            data = json.loads(text_of(r))
            print(f"\n[run_sql 正常] ok={data.get('ok')} 行数={data.get('row_count')} "
                  f"列={data.get('columns')} 自动补 LIMIT={'LIMIT' in str(data.get('sql', '')).upper()}")
            check("run_sql 正常执行", data.get("ok") and data.get("row_count", 0) > 0)
            check("run_sql 自动补 LIMIT", "LIMIT" in str(data.get("sql", "")).upper())

            # ---- 6. run_sql：危险语句必须被拦 ----
            r = await session.call_tool("run_sql", {"sql": "DROP TABLE orders"})
            data = json.loads(text_of(r))
            print(f"\n[run_sql 危险语句] ok={data.get('ok')} 原因={data.get('reasons')}")
            check("run_sql 拦下 DROP", data.get("ok") is False)

            # ---- 7. run_sql：多语句注入必须被拦 ----
            r = await session.call_tool("run_sql", {"sql": "SELECT 1; DELETE FROM refund;"})
            data = json.loads(text_of(r))
            check("run_sql 拦下多语句注入", data.get("ok") is False, str(data)[:160])

            # ---- 8. ask（需要模型可用） ----
            if skip_ask:
                print("\n[ask] 已跳过（--skip-ask）")
            else:
                r = await session.call_tool("ask", {"question": "一共有多少个客户？"})
                data = json.loads(text_of(r))
                print(f"\n[ask] ok={data.get('ok')} sql={str(data.get('sql'))[:70]}")
                print(f"      结论={str(data.get('answer'))[:80]}")
                print(f"      错误={data.get('error')}")
                # ask 依赖模型服务，可用与否都算协议通；这里只记录不判负
                print("      （ask 依赖 LLM，若本地 Ollama 未启动会返回 ok=false + hint，属正常）")

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 70)
    print(f"MCP 协议测试：{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ask", action="store_true", help="跳过需要 LLM 的 ask 工具")
    args = parser.parse_args()
    sys.exit(anyio.run(main, args.skip_ask))
