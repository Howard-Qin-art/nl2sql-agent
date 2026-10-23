# -*- coding: utf-8 -*-
"""澄清规则白名单的测试：命中该问的，且**不能**误伤能直接答的。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.clarify_rules import clarify_rule  # noqa: E402


# ---------------------------------------------------------------- 应该命中
@pytest.mark.parametrize("question", [
    "最近的销售情况怎么样？",          # 相对时间 + 指标不明（全量评测里漏掉了）
    "客户增长情况如何？",              # 指标不明（漏掉了）
    "我们的大客户都有哪些？",          # 没有量化标准（漏掉了）
    "退款情况正常吗？",                # "正常"没有标准
    "帮我看看数据",                    # 毫无指标
    "哪个产品卖得好？",                # 销量还是金额未定
    "最近表现怎么样",
    "近期有什么异常吗",
])
def test_rule_forces_clarification(question):
    hit = clarify_rule(question)
    assert hit, f"应该触发规则：{question}"
    assert len(hit) > 8


# ---------------------------------------------------------------- 不能误伤
@pytest.mark.parametrize("question", [
    "一共有多少个客户？",
    "上海的客户有多少人？",
    "会员等级一共有几种？",
    "各状态订单的数量分布是怎样的？",
    "最近 30 天的成交额是多少？",       # 有明确时间范围
    "2024 年每个月的成交额是多少？",
    "华东大区已完成订单的总成交额是多少？",
    "活跃客户有多少人？",              # "活跃客户"在口径里有定义，属于明确指标
    "每个大区各有多少活跃客户？",
    "消费金额最高的前 10 个客户是谁？",
    "退款总金额是多少？",
    "苹果手机的销量是多少？",           # 由模型/护栏处理，不是"口径不明"
])
def test_rule_does_not_fire_on_clear_questions(question):
    assert clarify_rule(question) is None, f"不该触发规则：{question}"


def test_rule_handles_empty_and_whitespace():
    assert clarify_rule("") is None
    assert clarify_rule("   ") is None
