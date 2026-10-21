# -*- coding: utf-8 -*-
"""校验 trace 用量是否真的挂到了 Agent 的 span 上（用 FakeLLM，秒级）。"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.llm import FakeBackend, LLMClient  # noqa: E402
from app.tracing import Tracer  # noqa: E402

GOOD = "SELECT COUNT(*) AS cnt FROM customer"
PLAN = json.dumps({"sql": GOOD, "tables_used": ["customer"], "explanation": "计数",
                   "confidence": 0.9, "needs_clarification": False, "clarify_question": ""}, ensure_ascii=False)


def responder(messages):
    text = "\n".join(m.get("content", "") for m in messages)
    if "【候选表】" in text:
        return json.dumps({"tables": ["customer"], "reason": "数客户"})
    if "判断这个问题是否" in text:
        return json.dumps({"ambiguous": False, "question": "", "confidence": 0.95})
    return PLAN


settings = dataclasses.replace(SETTINGS, llm_backend="fake", n_candidates=2, trace_enabled=True)
tracer = Tracer(enabled=True)
llm = LLMClient(settings=settings, backend=FakeBackend(settings, responder=responder), tracer=tracer)
agent = NL2SQLAgent(settings=settings, llm=llm, tracer=tracer, use_summary=False)
res = agent.run("一共有多少个客户？")

print("EX 通道检查：ok =", res.ok, " row_count =", res.row_count)
print("trace:", json.dumps(res.trace, ensure_ascii=False))
print(agent.tracer.pretty())
assert res.ok
assert res.trace["llm_calls"] >= 3, f"LLM 调用次数没记上：{res.trace}"
assert res.trace["prompt_tokens"] > 0, f"prompt token 没记上：{res.trace}"
print("PASS：trace 的 llm_calls / prompt_tokens 已正确挂到 span 上")

# ---- 逐题归零：第二次运行不能把第一次的用量累加进来 ----
first = res.trace["llm_calls"]
agent.tracer.reset()
res2 = agent.run("在售商品有多少件？")
print("第二次（reset 后）llm_calls =", res2.trace["llm_calls"], "| 第一次 =", first)
assert res2.trace["llm_calls"] == first, f"trace 没有逐题归零：{res2.trace['llm_calls']} != {first}"
print("PASS：trace 已逐题归零（评测指标不会被跨题累积污染）")
