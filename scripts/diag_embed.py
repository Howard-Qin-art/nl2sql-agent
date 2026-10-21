# -*- coding: utf-8 -*-
"""临时诊断：打印向量召回的真实堆栈（定位 bge-m3 接线问题）。"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.schema_linker import SchemaLinker, _embed_batch  # noqa: E402

print("--- 1) 直接批量 embedding ---")
vecs = _embed_batch(SETTINGS.ollama_base, "bge-m3", ["测试一", "测试二"], num_gpu=SETTINGS.ollama_num_gpu)
print("返回:", type(vecs).__name__, "元素:", type(vecs[0]).__name__, "维度:", len(vecs[0]))

print("\n--- 2) SchemaLinker._embedding_rankings ---")
docs = introspect(SETTINGS.db_path)
linker = SchemaLinker(docs, llm=None, use_llm_rerank=False, top_k=SETTINGS.top_k_tables,
                      embed_model="bge-m3", ollama_base=SETTINGS.ollama_base,
                      embed_cache_dir=SETTINGS.embed_cache_dir, num_gpu=SETTINGS.ollama_num_gpu)
try:
    tables, cols = linker._embedding_rankings("哪个地区的销售额最高？")
    print("OK 表级:", tables[:5])
    print("OK 列级归并:", cols[:5])
    print("stats:", linker.embed_stats)
except Exception:
    traceback.print_exc()
