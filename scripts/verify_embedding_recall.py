# -*- coding: utf-8 -*-
"""向量召回 A/B 对比：同义词/业务黑话问题下，BM25 召回 vs BM25+bge-m3 召回。

为什么需要这个实验：
    BM25 靠字面词重叠，用户说「地区」「退货」「顾客」「货品」时，
    库里写的是「大区」「退款」「客户」「商品」—— 字面不重叠就召不回。
    向量召回（bge-m3 是中文强项的多语 embedding）正是补这个缺口。
    为了只看召回路，两边都关掉 LLM 重排（那是精排阶段的增强）。

用法：
    python scripts/verify_embedding_recall.py                    # 默认 bge-m3
    python scripts/verify_embedding_recall.py --model bge-m3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.schema_linker import SchemaLinker  # noqa: E402

# (问题, 回答它必须用到的表)  —— 故意用同义词/口语，避开库里的字面词
CASES: list[tuple[str, set[str]]] = [
    ("哪个地区的销售额最高？", {"orders", "region"}),
    ("退货都是什么原因？", {"refund"}),
    ("顾客里有多少活跃用户？", {"customer"}),
    ("货品卖得最好的是哪个？", {"order_item", "product", "orders"}),
    ("供货商的评分怎么样？", {"supplier"}),
    ("账单收款情况如何？", {"payment"}),
    ("买家下单渠道分布是怎样的？", {"orders"}),
    ("售后赔付金额合计多少？", {"refund"}),
    ("哪类东西最贵？", {"product", "category"}),
    ("积分等级高的客户有哪些？", {"customer"}),
]


def evaluate(linker: SchemaLinker, cases) -> tuple[int, list[tuple[str, bool, list[str]]]]:
    hits = 0
    detail = []
    for question, gold in cases:
        res = linker.link(question)
        linked = set(res.table_names)
        ok = gold <= linked
        hits += ok
        detail.append((question, ok, res.table_names))
    return hits, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bge-m3")
    args = parser.parse_args()

    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    docs = introspect(SETTINGS.db_path)

    base = SchemaLinker(docs, llm=None, use_llm_rerank=False, top_k=SETTINGS.top_k_tables,
                        embed_model=None, ollama_base=SETTINGS.ollama_base)
    vec = SchemaLinker(docs, llm=None, use_llm_rerank=False, top_k=SETTINGS.top_k_tables,
                       embed_model=args.model, ollama_base=SETTINGS.ollama_base,
                       embed_cache_dir=SETTINGS.embed_cache_dir, num_gpu=SETTINGS.ollama_num_gpu)

    t0 = time.perf_counter()
    hits_base, detail_base = evaluate(base, CASES)
    t_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    hits_vec, detail_vec = evaluate(vec, CASES)
    t_vec = time.perf_counter() - t0

    print(f"问题数 {len(CASES)}｜top_k={SETTINGS.top_k_tables}｜向量模型={args.model}")
    print(f"{'问题':<22} {'BM25':<6} {'BM25+bge-m3':<12} 命中表（向量版）")
    print("-" * 96)
    for (q, gold), (_, ok_b, _), (_, ok_v, tables_v) in zip(CASES, detail_base, detail_vec):
        mark = lambda ok: "✅" if ok else "❌"
        print(f"{q:<22} {mark(ok_b):<6} {mark(ok_v):<12} {', '.join(tables_v)}")
    print("-" * 96)
    print(f"表召回命中（回答所需表全在 top_k 内）：BM25 {hits_base}/{len(CASES)}"
          f"  →  BM25+{args.model} {hits_vec}/{len(CASES)}")
    print(f"耗时：BM25 {t_base:.1f}s（{len(CASES)} 题）｜向量版 {t_vec:.1f}s（首次含模型加载与建缓存）")
    print(f"向量统计：{vec.embed_stats or ('跳过：' + str(vec.embed_skip_reason))}")
    if vec.embed_stats.get("used"):
        print(f"缓存文件：{vec._cache_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
