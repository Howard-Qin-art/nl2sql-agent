# -*- coding: utf-8 -*-
"""长短期记忆 / 评测 manifest / trace 原始请求 的单元测试。"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def work_dir():
    """自建临时目录（不用 pytest 的 work_dir：某些受限环境下它没法扫目录）。"""
    d = ROOT / ".pytest-work" / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

from app.config import SETTINGS  # noqa: E402
from app.llm import FakeBackend, LLMClient  # noqa: E402
from app.memory import LongTermMemory, SessionStore, build_memory_block  # noqa: E402
from app.prompts import build_generation_messages  # noqa: E402
from app.tracing import Tracer  # noqa: E402
from eval.manifest import build_manifest, describe  # noqa: E402


# ---------------------------------------------------------------- 短期记忆
def test_session_append_and_history(work_dir):
    s = SessionStore("t1", root=work_dir, max_turns=4)
    assert s.history() == []
    s.append("一共有多少个客户", answer="200 个客户", sql="SELECT COUNT(*) FROM customer", tables=["customer"])
    s.append("那上个月呢", answer="上月新增 12 个", sql="SELECT ... WHERE ...")
    hist = s.history()
    assert len(hist) == 4                       # 两轮 → 4 条消息
    assert hist[0]["role"] == "user"
    assert hist[1]["role"] == "assistant"
    assert hist[2]["content"].startswith("那上个月呢")
    # 落盘了：新实例能读到同一份记忆
    assert SessionStore("t1", root=work_dir).history() == hist


def test_session_truncates_to_max_turns(work_dir):
    s = SessionStore("t2", root=work_dir, max_turns=2)
    for i in range(5):
        s.append(f"问题{i}", answer=f"答案{i}")
    assert len(s.turns()) == 2
    assert s.turns()[-1].question == "问题4"


def test_session_clear_and_corrupt_file(work_dir):
    s = SessionStore("t3", root=work_dir)
    s.append("q", answer="a")
    s.clear()
    assert s.history() == []
    # 文件被写坏时不能让主流程崩
    s.path.write_text("{not json", encoding="utf-8")
    assert s.history() == []


# ---------------------------------------------------------------- 长期记忆
def test_long_term_remember_list_forget(work_dir):
    mem = LongTermMemory(work_dir / "mem.db")
    fid = mem.remember("成交额只算已完成订单", kind="metric")
    rows = mem.list()
    assert len(rows) == 1 and rows[0]["text"] == "成交额只算已完成订单"
    assert mem.forget(fid) is True
    assert mem.list() == []
    assert mem.forget("不存在的id") is False


def test_long_term_recall_ranks_relevant_first(work_dir):
    mem = LongTermMemory(work_dir / "mem.db")
    mem.remember("成交额只算已完成订单")
    mem.remember("金额单位统一用万元")
    mem.remember("华东大区的数据要排除测试账号")
    hits = mem.recall("华东大区成交额多少")
    assert hits, "应该召回至少一条"
    assert any("成交额" in h["text"] for h in hits)
    assert hits[0]["hits"] >= 1                  # 命中计数被更新


def test_long_term_recall_filters_irrelevant(work_dir):
    mem = LongTermMemory(work_dir / "mem.db")
    mem.remember("成交额只算已完成订单")
    assert mem.recall("今天天气怎么样") == []       # 不相关就别塞进上下文


# ---------------------------------------------------------------- Prompt 注入
def test_memory_block_contains_both_parts(work_dir):
    session = SessionStore("t4", root=work_dir)
    session.append("华东有多少客户", answer="120 个")
    mem = LongTermMemory(work_dir / "mem.db")
    mem.remember("成交额只算已完成订单")
    block = build_memory_block("华东成交额", long_term=mem, session=session)
    assert "【历史对话" in block and "华东有多少客户" in block
    assert "【已知偏好" in block and "成交额只算已完成订单" in block


def test_memory_block_empty_when_nothing(work_dir):
    assert build_memory_block("随便问问") == ""


def test_generation_prompt_injects_memory():
    msgs = build_generation_messages("几个客户", "TABLE customer", memory_text="【已知偏好】默认看最近 30 天")
    prompt = msgs[-1]["content"]
    assert "默认看最近 30 天" in prompt
    # 不传时不能凭空多出段落
    assert "【已知偏好】" not in build_generation_messages("几个客户", "TABLE customer")[-1]["content"]


# ---------------------------------------------------------------- trace 原始请求
def test_trace_records_llm_exchange_when_enabled(work_dir):
    import dataclasses
    s = dataclasses.replace(SETTINGS, trace_llm=True)
    tracer = Tracer(enabled=True)
    llm = LLMClient(settings=s, backend=FakeBackend(s, responses=["ok"]), tracer=tracer)
    with tracer.span("generate", n=1):
        llm.chat([{"role": "user", "content": "你好"}], max_tokens=8)
    span = tracer.spans[0]
    assert span.llm_calls == 1
    assert len(span.llm_exchanges) == 1
    ex = span.llm_exchanges[0]
    assert ex["messages"][0]["content"] == "你好"
    # 落盘里能看到
    out = tracer.to_jsonl(work_dir / "trace.jsonl")
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["llm_exchanges"][0]["response"] == "ok"


def test_trace_skips_exchange_when_disabled():
    import dataclasses
    s = dataclasses.replace(SETTINGS, trace_llm=False)
    tracer = Tracer(enabled=True)
    llm = LLMClient(settings=s, backend=FakeBackend(s, responses=["ok"]), tracer=tracer)
    with tracer.span("generate"):
        llm.chat([{"role": "user", "content": "你好"}], max_tokens=8)
    assert tracer.spans[0].llm_exchanges == []
    assert "llm_exchanges" not in tracer.spans[0].to_dict()      # 不写空字段，文件更小


# ---------------------------------------------------------------- 评测 manifest
def test_manifest_has_reproducible_fields():
    qp = ROOT / "eval" / "questions.jsonl"
    m = build_manifest(SETTINGS, qp)
    assert set(m) >= {"created_at", "git", "settings", "fingerprints", "env"}
    assert m["fingerprints"]["questions"], "题库指纹不能为空"
    assert m["fingerprints"]["prompts.py"], "提示词指纹不能为空（改没改过靠它判断）"
    assert m["settings"]["llm_backend"]
    text = describe(m)
    assert "prompts.py#" in text and "题库#" in text


def test_manifest_fingerprint_stable_and_sensitive(work_dir):
    f = work_dir / "a.txt"
    f.write_text("hello", encoding="utf-8")
    q1 = build_manifest(SETTINGS, f)["fingerprints"]["questions"]
    q2 = build_manifest(SETTINGS, f)["fingerprints"]["questions"]
    assert q1 == q2                               # 同一文件指纹稳定
    f.write_text("hello!", encoding="utf-8")
    assert build_manifest(SETTINGS, f)["fingerprints"]["questions"] != q1   # 改了就能发现
