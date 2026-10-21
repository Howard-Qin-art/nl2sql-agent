# -*- coding: utf-8 -*-
"""评测指标测试：结果集比对（行序/列序/浮点/空值/列数）与统计聚合。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import markdown_report, percentile, results_equal, sql_normalized, summarize  # noqa: E402


def test_row_order_irrelevant():
    gold = [("华东", 293), ("华中", 309)]
    pred = [("华中", 309), ("华东", 293)]
    assert results_equal(gold, pred)


def test_float_rounding_tolerated():
    assert results_equal([(1, 123.456789)], [(1, 123.4568)])


def test_numeric_tolerance_accepts_rounding():
    """gold 写 ROUND(x,2)、模型返回全精度：同一个数，应当算对。"""
    gold = [("华东", 10343933.45)]
    pred = [("华东", 10343933.451234)]
    assert results_equal(gold, pred)
    # 但「最严口径」下它是 False，两个口径都要能看到
    assert not results_equal(gold, pred, tolerant=False)


def test_numeric_tolerance_rejects_real_error():
    """真正的算法错误不能被容差放过。"""
    assert not results_equal([("华东", 12345.67)], [("华东", 12350.12)])
    assert not results_equal([(100,)], [(101,)])


def test_tolerance_compares_strings_exactly():
    assert not results_equal([("在售",)], [("在售商品",)])
    assert results_equal([("在售",)], [("在售",)])


def test_null_handling():
    assert results_equal([(None, 1)], [(None, 1)])
    assert not results_equal([(None, 1)], [("", 1)])


def test_column_count_must_match():
    assert not results_equal([(1, 2)], [(1,)])
    assert not results_equal([(1,)], [(1, 2)])


def test_relaxed_column_order():
    strict = results_equal([("华东", 293)], [(293, "华东")])
    relaxed = results_equal([("华东", 293)], [(293, "华东")], relaxed=True)
    assert strict is False and relaxed is True


def test_empty_results_equal():
    assert results_equal([], [])
    assert not results_equal([], [(1,)])


def test_sql_normalization():
    a = sql_normalized("SELECT COUNT(*) FROM orders WHERE status = 'completed'")
    b = sql_normalized("select count(*) from orders where status='completed'")
    assert a == b
    assert sql_normalized("SELECT 1 FROM orders") != a


def test_percentile():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 0.5) == 5.5
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 10
    assert percentile([], 0.5) == 0.0


def test_summarize_and_report():
    records = [
        {"gradable": True, "correct": True, "correct_relaxed": True, "em": False, "valid_sql": True,
         "latency_ms": 100, "tags": ["simple"], "llm_calls": 4, "prompt_tokens": 100, "completion_tokens": 10},
        {"gradable": True, "correct": False, "correct_relaxed": True, "em": False, "valid_sql": True,
         "latency_ms": 300, "tags": ["join"], "llm_calls": 6, "prompt_tokens": 200, "completion_tokens": 20,
         "needed_repair": True},
        {"expect": "clarify", "clarify_ok": True, "tags": ["ambiguous"], "latency_ms": 150},
        {"expect": "safe", "safe_ok": True, "tags": ["hostile"], "latency_ms": 120},
    ]
    s = summarize(records)
    assert s["n_questions"] == 4 and s["n_gradable"] == 2
    assert s["ex"] == 0.5 and s["ex_relaxed"] == 1.0
    assert s["clarify_accuracy"] == 1.0 and s["guardrail_block_rate"] == 1.0
    assert s["repair_success_rate"] == 0.0
    assert s["latency_ms"]["mean"] > 0
    assert set(s["by_tag"]) == {"simple", "join", "ambiguous", "hostile"}
    report = markdown_report(s, model="qwen2.5:7b", candidates=3, engine="core")
    assert "执行准确率" in report and "50.0%" in report
