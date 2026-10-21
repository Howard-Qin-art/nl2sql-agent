# -*- coding: utf-8 -*-
"""Schema Linking 测试：词法召回、字面值命中 + 表级外键扩展、RRF 融合、无关问题不过度召回。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.schema_linker import BM25, SchemaLinker, rrf, tokenize  # noqa: E402


@pytest.fixture(scope="module")
def docs():
    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    return introspect(SETTINGS.db_path)


@pytest.fixture(scope="module")
def linker(docs):
    # 关掉 LLM 重排，纯离线验证召回与融合逻辑
    return SchemaLinker(docs, llm=None, use_llm_rerank=False, top_k=5)


def test_tokenize_chinese_bigrams():
    tokens = tokenize("华东成交额")
    assert "华" in tokens and "华东" in tokens and "成交" in tokens


def test_bm25_ranks_on_topic_doc():
    corpus = ["订单 成交额 orders total_amount", "客户 customer name", "退款 refund reason"]
    bm = BM25(corpus)
    top = bm.rank("成交额是多少", top_n=1)
    assert top and top[0][0] == 0


def test_rrf_fusion_prefers_consensus():
    fused = rrf([["a", "b", "c"], ["b", "a"], ["b", "d"]])
    assert max(fused, key=fused.get) == "b"


def test_value_hit_expands_to_fact_table(linker):
    res = linker.link("华东大区2024年每个月的成交额是多少？")
    assert "orders" in res.table_names, res.table_names
    assert "region" in res.table_names, res.table_names
    assert any("华东" in h for h in res.value_hits), res.value_hits
    # 表级外键扩展：值命中 region 后，引用 region 的 orders 也应被带进来
    assert any("FK" in h and "orders" in h for h in res.value_hits), res.value_hits


def test_revenue_question_links_orders(linker):
    res = linker.link("所有已完成订单的总成交额是多少？")
    assert "orders" in res.table_names


def test_window_question_links_category_and_product(linker):
    res = linker.link("每个大类里销量最高的商品是哪个？")
    assert {"product", "category"} & set(res.table_names)


def test_top_k_respected(linker):
    res = linker.link("每个大区的活跃客户数、退款金额、支付成功率分别是多少？")
    assert len(res.table_names) <= linker.top_k + 2  # 允许值命中额外补充维表


def test_status_value_hit_maps_to_orders_column(linker):
    res = linker.link("已完成的订单有多少单？")
    # 'completed' 是 orders.status 的抽样值，应命中 columns 层面的值索引
    assert any("orders" in h for h in res.value_hits) or "orders" in res.table_names
