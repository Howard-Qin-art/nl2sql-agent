# -*- coding: utf-8 -*-
"""阶段二验收：Schema Linking（BM25+值命中+RRF+LLM重排）与候选生成/自一致性投票。

会用真实模型（默认本地 Ollama qwen2.5:7b，CPU），耗时约 1~4 分钟。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.candidates import SQLPlan, run_candidates  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.executor import SQLExecutor  # noqa: E402
from app.db.schema import introspect, schema_prompt  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.prompts import build_generation_messages  # noqa: E402
from app.schema_linker import SchemaLinker  # noqa: E402
from app.tracing import Tracer  # noqa: E402

if not SETTINGS.db_path.exists():
    seed_database(SETTINGS.db_path)

docs = introspect(SETTINGS.db_path)
executor = SQLExecutor(SETTINGS.db_path, timeout_s=SETTINGS.query_timeout_s, max_rows=SETTINGS.max_rows)
tracer = Tracer()
llm = LLMClient(tracer=tracer)
linker = SchemaLinker(docs, llm=llm, top_k=SETTINGS.top_k_tables,
                      embed_model=None, ollama_base=SETTINGS.ollama_base)
ALLOWED = {d.name for d in docs}

QUESTIONS = [
    "华东大区2024年每个月的成交额是多少？",
    "各渠道客单价从高到低排前5",
    "每个大类里销量最高的商品是哪个？",
    "退款原因占比分别是多少？",
    "苹果手机的销量怎么样？",  # 库里没有这个商品：考察召回是否会退化 + 模型该不该说不知道
]

print("=" * 78)
print("1) Schema Linking（含 LLM 重排）")
link_results = {}
for q in QUESTIONS:
    t0 = time.time()
    res = linker.link(q)
    link_results[q] = res
    print(f"\n问：{q}   ({time.time() - t0:.1f}s)")
    print("   选中表：", res.table_names)
    print("   融合分数：", {k: v for k, v in list(res.scores.items())[:6]})
    if res.value_hits:
        print("   值命中：", res.value_hits[:4])
    print("   RRF 池：", res.debug["pool"][:8])
    print("   LLM 重排：", str(res.debug["llm_rerank"])[:150])
    print("   向量召回：", res.debug["embedding"])

print("=" * 78)
print("2) 候选生成 + 自一致性投票（N=%d）" % SETTINGS.n_candidates)
for q in QUESTIONS[:3]:
    linked = link_results[q]
    schema_text = schema_prompt(linked.tables)
    messages = build_generation_messages(q, schema_text)
    t0 = time.time()
    plans: list[tuple[int, SQLPlan, int | None]] = []
    with tracer.span("generate_candidates", question=q, n=SETTINGS.n_candidates) as sp:
        for i in range(SETTINGS.n_candidates):
            seed = 1000 + i * 17
            try:
                data, _r = llm.chat_json(messages, model_cls=SQLPlan, temperature=SETTINGS.sample_temperature,
                                         max_tokens=SETTINGS.max_tokens, seed=seed)
                plans.append((i, SQLPlan(**data), seed))
            except Exception as exc:
                print(f"   候选#{i} 生成失败：{type(exc).__name__}: {str(exc)[:120]}")
        sp.outputs["plans"] = [p[1].sql[:120] for p in plans]
    cset = run_candidates(plans, executor, allowed_tables=ALLOWED, default_limit=SETTINGS.default_limit)
    print(f"\n问：{q}   总耗时 {time.time() - t0:.1f}s")
    for c in cset.candidates:
        print("   ", c.brief()[:150])
    print("   投票分布:", cset.votes, " 一致性:", round(cset.agreement, 2))
    if cset.winner:
        print("   >>> 胜出 SQL:", cset.winner.sql)
        print("   >>> 结果预览:", cset.winner.exec.preview(3))
        print("   >>> 模型解释:", cset.winner.plan.explanation[:100])
    else:
        print("   >>> 全部候选失败，需要自愈")

print("=" * 78)
print("3) Trace 汇总")
print(tracer.pretty())
print(tracer.summary())
trace_path = tracer.to_jsonl(SETTINGS.trace_dir / "verify_phase2.jsonl")
print("trace 已写入:", trace_path)
