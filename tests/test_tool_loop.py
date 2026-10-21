# -*- coding: utf-8 -*-
"""工具调用循环（Plan B）的离线测试：用 FakeBackend 脚本化「模型决定调哪个工具」。

覆盖：
    1. 两轮工具调用 + 最终结论（正常路径）
    2. 不调工具直接回答
    3. 模型调了不存在的工具 -> 错误回传后自我纠正
    4. run_sql 被护栏拒绝 -> 错误回传
    5. 步数用尽 -> 明确报错
    6. 结果里能拿到最后执行的 SQL / 行数据
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tool_loop import local_tools, run_tool_agent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import FakeBackend, LLMClient  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed")


@pytest.fixture(scope="module")
def seed():
    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    return SETTINGS


def make_llm(tool_script, settings=None) -> LLMClient:
    s = settings or dataclasses.replace(SETTINGS, trace_enabled=False)
    return LLMClient(settings=s, backend=FakeBackend(s, tool_script=tool_script))


def make_tools(settings=None):
    return local_tools(settings or SETTINGS)


def test_two_rounds_then_answer(seed):
    """模型先看结构、再查数、最后给结论（典型的真实执行轨迹）。"""
    script = [
        {"tool_calls": [{"name": "get_schema", "arguments": {"tables": ["orders"]}}]},
        {"tool_calls": [{"name": "run_sql",
                         "arguments": {"sql": "SELECT COUNT(*) AS cnt FROM orders WHERE status = 'completed'"}}]},
        {"text": "已完成订单共 2400 单。"},
    ]
    res = run_tool_agent("已完成订单有多少单？", llm=make_llm(script), tools=make_tools(seed))
    assert res.ok, res.error
    assert res.steps == 3
    assert [c.name for c in res.tool_calls] == ["get_schema", "run_sql"]
    assert "已完成订单" in res.answer
    # run_sql 的结果被提取到了结构化字段
    assert res.sql.upper().startswith("SELECT")
    assert res.row_count == 1
    assert res.columns == ["cnt"]
    assert res.rows[0][0] > 0


def test_answer_without_tools(seed):
    res = run_tool_agent("你好", llm=make_llm([{"text": "你好，我可以帮你查订单数据。"}]), tools=make_tools(seed))
    assert res.ok
    assert res.steps == 1
    assert res.tool_calls == []
    assert res.answer.startswith("你好")


def test_unknown_tool_feeds_error_back_and_recovers(seed):
    script = [
        {"tool_calls": [{"name": "drop_database", "arguments": {}}]},   # 模型瞎调了一个不存在的工具
        {"tool_calls": [{"name": "list_tables", "arguments": {}}]},     # 纠正后正常调
        {"text": "库里有 9 张表。"},
    ]
    res = run_tool_agent("库里有什么？", llm=make_llm(script), tools=make_tools(seed))
    assert res.ok
    assert res.tool_calls[0].name == "drop_database"
    assert res.tool_calls[0].ok is False
    assert "没有这个工具" in res.tool_calls[0].output
    assert res.tool_calls[1].name == "list_tables" and res.tool_calls[1].ok


def test_guardrail_rejection_is_fed_back(seed):
    script = [
        {"tool_calls": [{"name": "run_sql", "arguments": {"sql": "DROP TABLE orders"}}]},
        {"tool_calls": [{"name": "run_sql", "arguments": {"sql": "SELECT COUNT(*) AS c FROM orders"}}]},
        {"text": "订单表共 3000 行。"},
    ]
    res = run_tool_agent("订单有多少？", llm=make_llm(script), tools=make_tools(seed))
    assert res.ok
    first, second = res.tool_calls
    assert first.ok and "护栏拒绝" in first.output      # 工具执行成功但返回了拒绝原因
    assert second.ok and second.output.count("row_count") == 1


def test_max_steps_guard(seed):
    """模型一直调工具不收敛 -> 到上限要明确报错，不能死循环。"""
    script = [{"tool_calls": [{"name": "list_tables", "arguments": {}}]} for _ in range(10)]
    res = run_tool_agent("随便问问", llm=make_llm(script), tools=make_tools(seed), max_steps=3)
    assert res.ok is False
    assert res.steps == 3
    assert "最大步数" in (res.error or "")


def test_extra_arguments_are_dropped(seed):
    """模型多给了参数（常见毛病）不能让工具抛 TypeError。"""
    script = [
        {"tool_calls": [{"name": "list_tables", "arguments": {"bogus": 1, "another": "x"}}]},
        {"text": "共 9 张表。"},
    ]
    res = run_tool_agent("库里有什么？", llm=make_llm(script), tools=make_tools(seed))
    assert res.ok
    assert res.tool_calls[0].ok is True
