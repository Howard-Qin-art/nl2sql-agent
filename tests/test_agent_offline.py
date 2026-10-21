# -*- coding: utf-8 -*-
"""Agent 离线端到端测试（FakeLLM，不联网、秒级）。

覆盖：
    1. 顺序引擎 happy path
    2. 执行自愈（坏 SQL -> 报错回注 -> 修好）
    3. 护栏拒绝 -> 自愈（模型写了 DELETE）
    4. 歧义澄清（core 的 clarify_handler）
    5. LangGraph 引擎 happy path
    6. LangGraph interrupt 挂起 + Command(resume) 恢复
    7. 两条编排路径结果一致
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.agent.graph import GraphAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import LLMClient  # noqa: E402

GOOD_SQL = (
    "SELECT r.region_name, COUNT(*) AS cnt FROM orders o "
    "JOIN region r ON o.region_id = r.region_id "
    "WHERE o.status = 'completed' GROUP BY r.region_name ORDER BY cnt DESC LIMIT 5"
)
BAD_COLUMN_SQL = "SELECT non_exist_col FROM orders LIMIT 5"
DELETE_SQL = "DELETE FROM orders"


def plan_json(sql: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "sql": sql,
            "tables_used": ["orders", "region"],
            "explanation": "按大区统计完成订单数",
            "confidence": confidence,
            "needs_clarification": False,
            "clarify_question": "",
        },
        ensure_ascii=False,
    )


class FakeResponder:
    """按 prompt 内容分发假响应；可配置首轮给坏 SQL、是否判定歧义。"""

    def __init__(self, *, ambiguous: bool = False, bad_sql: str | None = None, bad_rounds: int = 1):
        self.ambiguous = ambiguous
        self.bad_sql = bad_sql
        self.bad_rounds = bad_rounds
        self.gen_calls = 0
        self.repair_calls = 0
        self.rerank_calls = 0
        self.clarify_calls = 0
        self.seen_questions: list[str] = []

    def __call__(self, messages) -> str:
        text = "\n".join(m.get("content", "") for m in messages)
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "【候选表】" in text:
            self.rerank_calls += 1
            return json.dumps({"tables": ["orders", "region"], "reason": "统计订单需要订单表和区域表"})
        if "判断这个问题是否" in text:
            self.clarify_calls += 1
            return json.dumps(
                {"ambiguous": self.ambiguous, "question": "请问要统计哪个时间范围？", "confidence": 0.4 if self.ambiguous else 0.95},
                ensure_ascii=False,
            )
        if "上一次生成的 SQL" in text:
            self.repair_calls += 1
            return plan_json(GOOD_SQL, confidence=0.85)
        if "【结果集】" in text:
            return "华东大区完成订单数最多。"
        # 生成阶段
        self.gen_calls += 1
        self.seen_questions.append(user_text[:200])
        if self.bad_sql and self.gen_calls <= self.bad_rounds:
            return plan_json(self.bad_sql, confidence=0.6)
        return plan_json(GOOD_SQL)


@pytest.fixture(scope="module")
def fake_settings():
    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    return settings_with()


def settings_with(**overrides):
    base = dict(llm_backend="fake", n_candidates=2, max_repair_rounds=2, trace_enabled=False)
    base.update(overrides)
    return dataclasses.replace(SETTINGS, **base)


def make_agent(settings, responder) -> NL2SQLAgent:
    llm = LLMClient(settings=settings, backend=None)  # 先建默认后端
    from app.llm import FakeBackend

    llm.backend = FakeBackend(settings, responder=responder)
    return NL2SQLAgent(settings=settings, llm=llm)


# ---------------------------------------------------------------- 顺序引擎
def test_core_pipeline_ok(fake_settings):
    responder = FakeResponder()
    agent = make_agent(fake_settings, responder)
    res = agent.run("哪个大区的完成订单最多？")
    assert res.ok, res.error
    assert res.row_count > 0
    assert "GROUP BY" in res.sql.upper()
    # LLM 重排选中的表必须在最终 schema 里（还会额外补入值命中的维表）
    assert "orders" in res.linked_tables and "region" in res.linked_tables
    assert res.answer
    assert responder.gen_calls == fake_settings.n_candidates


def test_self_repair_on_bad_column(fake_settings):
    # 只给 1 个候选，让它必然失败，才能验证自愈路径
    settings = settings_with(n_candidates=1)
    responder = FakeResponder(bad_sql=BAD_COLUMN_SQL, bad_rounds=1)
    agent = make_agent(settings, responder)
    res = agent.run("哪个大区的完成订单最多？")
    assert res.ok, res.error
    assert res.repair_rounds >= 1
    assert res.repaired
    # 护栏会用 sqlglot 重渲染 SQL（补 AS 别名等），所以比对语义而不是字符串
    assert "NON_EXIST_COL" not in res.sql.upper()
    assert "COUNT(*)" in res.sql.upper()
    assert any(r.get("reason") and "non_exist_col" in str(r.get("reason")) for r in res.rejected)


def test_repair_after_guardrail_rejection(fake_settings):
    settings = settings_with(n_candidates=1)
    responder = FakeResponder(bad_sql=DELETE_SQL, bad_rounds=1)
    agent = make_agent(settings, responder)
    res = agent.run("把订单表清空")
    # 护栏拦下 DELETE -> 进入自愈 -> 改成合法 SELECT
    assert res.ok, res.error
    assert responder.repair_calls >= 1
    assert "DELETE" not in res.sql.upper()
    assert res.sql.upper().startswith("SELECT")


def test_clarification_flow_core(fake_settings):
    responder = FakeResponder(ambiguous=True)
    agent = make_agent(fake_settings, responder)
    asked: list[str] = []

    def handler(question: str) -> str:
        asked.append(question)
        return "只看 2024 年"

    res = agent.run("订单情况怎么样？", clarify_handler=handler)
    assert asked and "时间范围" in asked[0]
    assert res.ok
    assert "2024" in res.state["question"], "澄清内容应拼回问题"
    assert res.state["clarifications"][0]["answer"] == "只看 2024 年"


# ---------------------------------------------------------------- LangGraph 引擎
def test_graph_pipeline_ok(fake_settings):
    responder = FakeResponder()
    ga = GraphAgent(agent=make_agent(fake_settings, responder))
    out = ga.run("哪个大区的完成订单最多？")
    assert not out.get("interrupted")
    assert out["ok"], out.get("error")
    assert out["engine"] == "langgraph"
    assert out["row_count"] > 0


def test_graph_interrupt_and_resume(fake_settings):
    responder = FakeResponder(ambiguous=True)
    ga = GraphAgent(agent=make_agent(fake_settings, responder))

    first = ga.run("订单情况怎么样？", thread_id="t-clarify")
    assert first.get("interrupted") is True, first
    assert "时间范围" in first["clarify"]["question"]

    second = ga.resume("只看 2024 年", thread_id="t-clarify")
    assert not second.get("interrupted")
    assert second["ok"], second.get("error")
    state = ga.state("t-clarify")
    assert state.get("clarified") is True
    assert any("2024" in c.get("answer", "") for c in state.get("clarifications", []))


def test_two_engines_agree(fake_settings):
    core_res = make_agent(fake_settings, FakeResponder()).run("哪个大区的完成订单最多？")
    graph_res = GraphAgent(agent=make_agent(fake_settings, FakeResponder())).run("哪个大区的完成订单最多？")
    assert core_res.sql == graph_res["sql"]
    assert core_res.row_count == graph_res["row_count"]


# ---------------------------------------------------------------- 体验：快速通道与可打断
def test_fast_path_for_chitchat_calls_no_llm(fake_settings):
    """「你好」不该走完整链路（6 次模型调用，CPU 上要好几分钟）。"""
    responder = FakeResponder()
    agent = make_agent(fake_settings, responder)
    res = agent.run("你好")
    assert not res.ok                        # 没有结果集
    assert "数据查询助手" in res.answer       # 但要有引导语
    assert (responder.gen_calls, responder.rerank_calls, responder.clarify_calls) == (0, 0, 0)
    assert res.trace["llm_calls"] == 0


def test_budget_aborts_before_any_llm_call(fake_settings):
    """总时间预算用尽时必须立刻返回并说明原因，而不是继续傻等。"""
    responder = FakeResponder()
    agent = make_agent(fake_settings, responder)
    res = agent.run("华东大区2024年每个月的成交额是多少？", budget_s=0.0001)
    assert res.error and "预算" in res.error
    assert "没有跑完" in res.answer
    assert responder.gen_calls == 0
    assert any("提前结束" in n for n in res.state.get("notes", []))


def test_cancel_event_aborts(fake_settings):
    """界面点「停止」时置位 cancel_event，阶段之间立刻退出。"""
    import threading

    event = threading.Event()
    event.set()
    agent = make_agent(fake_settings, FakeResponder())
    res = agent.run("华东大区2024年每个月的成交额是多少？", cancel_event=event)
    assert res.error and "停止" in res.error
