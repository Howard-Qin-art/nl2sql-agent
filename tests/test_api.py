"""FastAPI 接口测试：健康检查、schema、/query（用 FakeLLM 注入，不依赖真实模型）。

注意：TestClient 会触发 lifespan（构建真实 Agent 对象，但 Ollama 后端在初始化时不发请求），
所以这里在请求前把 STATE 里的 agent/graph 换成 FakeLLM 驱动的实例。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import api as api_module  # noqa: E402
from app.agent.core import NL2SQLAgent  # noqa: E402
from app.agent.graph import GraphAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import FakeBackend, LLMClient  # noqa: E402

PLAN = json.dumps({
    "sql": "SELECT COUNT(*) AS cnt FROM orders WHERE status = 'completed'",
    "tables_used": ["orders"], "explanation": "统计完成订单数",
    "confidence": 0.9, "needs_clarification": False, "clarify_question": "",
}, ensure_ascii=False)


def responder(messages):
    text = "\n".join(m.get("content", "") for m in messages)
    if "【候选表】" in text:
        return json.dumps({"tables": ["orders"], "reason": "统计订单"})
    if "判断这个问题是否" in text:
        return json.dumps({"ambiguous": False, "question": "", "confidence": 0.95})
    if "【结果集】" in text:
        return "已完成订单共若干单。"
    return PLAN


@pytest.fixture(scope="module")
def client():
    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    with TestClient(api_module.app) as c:
        settings = dataclasses.replace(SETTINGS, llm_backend="fake", n_candidates=2, trace_enabled=True)
        tracer = api_module.Tracer(enabled=True)
        llm = LLMClient(settings=settings, backend=FakeBackend(settings, responder=responder), tracer=tracer)
        agent = NL2SQLAgent(settings=settings, llm=llm, tracer=tracer, use_summary=True)
        api_module.STATE["agent"] = agent
        api_module.STATE["graph"] = GraphAgent(agent=agent)
        api_module.STATE["docs"] = agent.docs
        yield c


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["tables"] >= 9
    assert data["n_candidates"] == 2


def test_schema(client):
    data = client.get("/schema").json()
    names = [t["name"] for t in data["tables"]]
    assert "orders" in names and "refund" in names
    orders = next(t for t in data["tables"] if t["name"] == "orders")
    assert orders["row_count"] > 0
    assert any(c["name"] == "status" for c in orders["columns"])


def test_query_graph_engine(client):
    resp = client.post("/query", json={"question": "已完成订单有多少单？", "engine": "graph"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] and data["row_count"] == 1
    assert "COUNT" in data["sql"].upper()
    assert data["trace"]["llm_calls"] >= 3


def test_query_core_engine(client):
    resp = client.post("/query", json={"question": "已完成订单有多少单？", "engine": "core"})
    assert resp.status_code == 200
    assert resp.json()["ok"]


def test_query_stream_sse(client):
    with client.stream("POST", "/query/stream", json={"question": "已完成订单有多少单？"}) as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    stages = [json.loads(line[6:])["stage"] for line in body.splitlines() if line.startswith("data: ")]
    assert stages[0] == "start" and stages[-1] == "done"
    assert "link_schema" in stages and "generate" in stages and "execute" in stages


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
