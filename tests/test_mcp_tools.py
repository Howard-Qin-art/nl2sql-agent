# -*- coding: utf-8 -*-
"""MCP 工具测试（进程内调用，不需要子进程/管道，CI 可跑）。

注意：真正的 stdio 传输测试在 `scripts/test_mcp_server.py`，
它需要给子进程创建管道 —— 某些沙箱环境会禁止（WinError 5），所以不放进 pytest。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp", reason="需要 mcp SDK 才能测 MCP 工具")

import mcp_server  # noqa: E402


def call(name: str, args: dict) -> dict:
    async def _run():
        result = await mcp_server.server.call_tool(name, args)
        text = "\n".join(getattr(b, "text", "") or "" for b in (result.content or []))
        return json.loads(text)
    return asyncio.run(_run())


def test_four_tools_registered():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = sorted(t.name for t in tools)
    assert names == ["ask", "get_schema", "list_tables", "run_sql"]


def test_run_sql_tool_has_sql_param():
    tools = asyncio.run(mcp_server.server.list_tools())
    tool = next(t for t in tools if t.name == "run_sql")
    props = (tool.input_schema or {}).get("properties", {})
    assert "sql" in props
    assert "sql" in (tool.input_schema or {}).get("required", [])


def test_list_tables():
    data = call("list_tables", {})
    assert data["ok"] is True
    assert len(data["tables"]) == 9
    names = {t["name"] for t in data["tables"]}
    assert {"orders", "order_item", "refund"} <= names
    assert all(t["rows"] > 0 for t in data["tables"])


def test_get_schema_filtered():
    data = call("get_schema", {"tables": ["orders", "refund"]})
    assert data["ok"] is True
    text = data["schema_text"]
    assert "TABLE orders" in text and "TABLE refund" in text
    assert "TABLE customer" not in text          # 不能夹带没要的表


def test_get_schema_unknown_table():
    data = call("get_schema", {"tables": ["not_a_table"]})
    assert data["ok"] is False
    assert "not_a_table" in data["error"]
    assert "available" in data                   # 要告诉客户端有哪些表可选


def test_run_sql_ok_and_auto_limit():
    data = call("run_sql", {"sql": "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status"})
    assert data["ok"] is True
    assert data["row_count"] > 0
    assert "LIMIT" in data["sql"].upper()         # 自动补 LIMIT
    assert data["columns"] == ["status", "cnt"]


def test_run_sql_blocks_drop():
    data = call("run_sql", {"sql": "DROP TABLE orders"})
    assert data["ok"] is False
    assert data["reasons"]                        # 必须给出拒绝原因


def test_run_sql_blocks_multi_statement():
    data = call("run_sql", {"sql": "SELECT 1; DELETE FROM refund;"})
    assert data["ok"] is False


def test_run_sql_blocks_system_table_and_unknown_table():
    assert call("run_sql", {"sql": "SELECT * FROM sqlite_master"})["ok"] is False
    assert call("run_sql", {"sql": "SELECT * FROM not_a_table"})["ok"] is False
