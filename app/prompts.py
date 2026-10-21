"""Prompt 模板集中管理（便于做 A/B 与消融实验，而不是散落在各处拼字符串）。

设计要点：
    - 结构化输出：只给 JSON 契约，配 `format=json` / `response_format=json_object`，
      再用 pydantic 校验（见 app/llm.chat_json 的自修复重试），不靠正则抠 JSON；
    - Few-shot 覆盖 Join / GROUP BY+HAVING / 窗口函数 / 日期口径 / 占比，把「怎么算」写死；
    - 口径写进 prompt：GMV 只算 completed、销售额=quantity*unit_price*(1-discount)，
      避免模型自己发明口径（这是 NL2SQL 评测里最常见的错误来源）；
    - 自愈 prompt 只给「原 SQL + 数据库报错原文 + 相关列」，要求最小改动，
      防止模型借机把整条 SQL 重写跑偏。
"""
from __future__ import annotations

import os
from typing import Any, Sequence

# 严格模式（默认开）：由第一轮全量评测的失败样本倒逼出来的两条硬约束 ——
#   1) 只输出问题要求的指标列（问「多少笔」就给一列 COUNT，不要顺手带上金额）；
#   2) 能单表回答就不要 JOIN（模型会为了"看起来完整"而多连表、多加过滤）。
# 关掉（NL2SQL_STRICT=0）即可复现基线 prompt 的评测结果，用于做消融对比。
STRICT = os.getenv("NL2SQL_STRICT", "1").strip().lower() not in {"0", "false", "no", "off"}

# 严格模式补充约束（由第一轮全量评测的失败样本倒逼）：模型倾向于"顺手给出更多信息"——
# 问「用信用卡成功支付有多少笔」它会自己 JOIN 订单表、再额外加一列 GMV，
# 结果列数和语义都偏离问题。这类系统性偏差靠投票救不了（三条候选都犯同样的错），
# 只能通过 prompt 硬约束 + 评测回归来治。
STRICT_BULLETS = """
- **只输出问题要的指标列**：问「有多少笔 / 多少个 / 哪几种」就只给一列 COUNT / DISTINCT 结果，
  不要顺手把金额、GMV、平均值等没问的指标也 SELECT 出来；
- **能单表回答就不要 JOIN**：只有当指标或筛选条件确实依赖另一张表时才连接，且连接必须有明确条件；
- 列数自检：输出列数应当等于「问题要求的指标数」（分组问题 = 维度数 + 指标数）。""" if STRICT else ""

JSON_CONTRACT = """{
  "sql": "一条 SQLite 语法的 SELECT 语句",
  "tables_used": ["用到的表名"],
  "explanation": "一句话中文说明这条 SQL 怎么算的",
  "confidence": 0.0,
  "needs_clarification": false,
  "clarify_question": "如果需要反问用户，把问题写在这里，否则空字符串"
}"""

SYSTEM_SQL = f"""你是一位严谨的数据分析工程师，负责把中文业务问题翻译成 SQLite 查询。

【硬性规则】
1. 只输出一条 SELECT（允许 WITH/子查询/UNION），禁止任何写操作；
2. 只能用下面 schema 里出现的表和列，不要臆造列名；
3. 口径必须按【业务口径】执行，不要自己发明；
4. 需要列明细时加 LIMIT；聚合查询不需要；
5. 日期列是 'YYYY-MM-DD' 文本，用 strftime('%Y-%m', order_date) 取月份，比较用字符串即可；
6. 字符串常量用单引号；
7. **严格按字面照做：问题里没要求的过滤条件一律不要加**。
   例如问「一共有多少个客户」就是全部客户，不要加 is_active = 1；
   问「有多少个客户」就只数 customer 表，不要因为存在订单表就额外 JOIN；
   反之，问题里明确要求的过滤条件（华东、已完成、2024 年）一个都不能漏；
{STRICT_BULLETS}
8. 只输出 JSON，不要 markdown 代码块、不要多余解释。

【输出 JSON 契约】
{JSON_CONTRACT}

字段含义：
- confidence：你对这条 SQL 正确性的把握（0~1）。口径不明、问题含糊、缺少必要信息时给低分；
- needs_clarification：当问题有歧义（例如没说什么算「最近」、指代不明、缺少时间范围）时置 true，
  并在 clarify_question 里写出你要问用户的那一句话。宁可反问，也不要用错误口径硬答。"""

FEW_SHOT: list[tuple[str, str]] = [
    (
        "华东大区已完成的订单总金额是多少？",
        "SELECT ROUND(SUM(o.total_amount), 2) AS gmv FROM orders o "
        "JOIN region r ON o.region_id = r.region_id "
        "WHERE o.status = 'completed' AND r.region_name = '华东'",
    ),
    (
        "2024 年每个月的订单数和成交额分别是多少？",
        "SELECT strftime('%Y-%m', order_date) AS ym, COUNT(*) AS orders_cnt, "
        "ROUND(SUM(total_amount), 2) AS gmv FROM orders "
        "WHERE status = 'completed' AND order_date >= '2024-01-01' AND order_date < '2025-01-01' "
        "GROUP BY ym ORDER BY ym",
    ),
    (
        "各渠道的客单价从高到低排一下，取前 5",
        "SELECT channel, ROUND(SUM(total_amount) * 1.0 / COUNT(*), 2) AS avg_order_value "
        "FROM orders WHERE status = 'completed' GROUP BY channel "
        "ORDER BY avg_order_value DESC LIMIT 5",
    ),
    (
        "每个大类里销量最高的商品是哪个？",
        "WITH sales AS ("
        "  SELECT c.parent_id, p.product_id, p.product_name, SUM(oi.quantity) AS qty "
        "  FROM order_item oi JOIN orders o ON oi.order_id = o.order_id "
        "  JOIN product p ON oi.product_id = p.product_id "
        "  JOIN category c ON p.category_id = c.category_id "
        "  WHERE o.status = 'completed' "
        "  GROUP BY c.parent_id, p.product_id, p.product_name) "
        "SELECT parent_id, product_name, qty FROM ("
        "  SELECT *, ROW_NUMBER() OVER (PARTITION BY parent_id ORDER BY qty DESC) AS rn FROM sales) "
        "WHERE rn = 1",
    ),
    (
        "退款原因都是什么，各占多少比例？",
        "SELECT reason, COUNT(*) AS cnt, "
        "ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM refund), 1) AS pct "
        "FROM refund GROUP BY reason ORDER BY cnt DESC",
    ),
]

BUSINESS_RULES = """【业务口径】（**只有当问题里明确提到对应概念时才使用，不要主动套用**）
- 成交额/GMV：orders.status = 'completed' 的 total_amount 之和（不要包含 refunded/pending/cancelled）；
- 商品销售额：SUM(order_item.quantity * order_item.unit_price * (1 - order_item.discount))；
- 客单价：SUM(total_amount) / COUNT(*)（仅 completed）；
- 活跃客户：customer.is_active = 1（只有问题问「活跃客户」时才加这个条件；问「客户」就是全部客户）；
- 在售商品：product.status = 'on_sale'（只有问「在售商品」时才加；问「商品」就是全部商品）；
- 类目是两级：category.parent_id IS NULL 为大类，parent_id 指向大类的是小类；
- 支付成功：payment.status = 'success'（金额看 payment.amount）；
- 退款只在 refund 表里，对应的订单 status='refunded'。"""


def _format_history(history: Sequence[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines = ["【历史对话（用于理解指代，如「那上个月呢」）】"]
    for turn in history[-6:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}: {str(turn.get('content', ''))[:400]}")
    return "\n".join(lines) + "\n"


def build_generation_messages(question: str, schema_text: str, *, history: Sequence[dict[str, str]] | None = None,
                              retry_hint: str | None = None) -> list[dict[str, str]]:
    parts = [f"【相关表结构】\n{schema_text}", BUSINESS_RULES, _format_history(history)]
    if retry_hint:
        parts.append(f"【上一次的错误，必须修正】\n{retry_hint}")
    parts.append("【Few-shot 示例】")
    for q, sql in FEW_SHOT:
        parts.append(f"问：{q}\nSQL：{sql}")
    parts.append(f"【现在的问题】\n{question}\n\n请按 JSON 契约输出。")
    return [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": "\n\n".join(p for p in parts if p)},
    ]


def build_repair_messages(question: str, schema_text: str, bad_sql: str, error: str,
                          *, history: Sequence[dict[str, str]] | None = None) -> list[dict[str, str]]:
    user = (
        f"【相关表结构】\n{schema_text}\n\n"
        f"【业务口径】\n{BUSINESS_RULES}\n\n"
        f"【原始问题】\n{question}\n\n"
        f"【上一次生成的 SQL（执行失败）】\n{bad_sql}\n\n"
        f"【数据库返回的原始错误】\n{error}\n\n"
        "请分析错误原因（列名拼错？表名不对？函数用错？聚合层级不对？），做**最小必要修改**后重新给出 SQL。"
        "不要引入新的表或列，保持原来的业务口径。按 JSON 契约输出。"
    )
    return [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": user},
    ]


def build_rerank_messages(question: str, candidates_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "你是数据库专家，负责做表级 Schema Linking 重排。"},
        {
            "role": "user",
            "content": (
                f"【用户问题】\n{question}\n\n"
                f"【候选表】\n{candidates_text}\n\n"
                "请选出回答该问题**必须**用到的表（最多 6 张，宁可少选精确，不要多选）。"
                '只输出 JSON：{"tables": ["表名"], "reason": "一句话理由"}'
            ),
        },
    ]


def build_clarify_messages(question: str, schema_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_SQL},
        {
            "role": "user",
            "content": (
                f"【相关表结构】\n{schema_text}\n\n{BUSINESS_RULES}\n\n"
                f"【用户问题】\n{question}\n\n"
                "【任务】判断这个问题是否**必须先问用户**才能给出唯一正确答案。\n\n"
                "判据（务必从严；能先给答案就不要乱问）：\n"
                "1. 只有「缺少必要信息 -> 无法给出唯一正确答案」时才澄清；\n"
                "2. 能用业务默认值兜住的一律**不要**澄清：时间范围默认全部时间、排序默认降序、\n"
                "   条数默认取合理上限（Top 10）、是否含退款按业务口径默认；\n"
                "3. 问「有多少 / 几个 / 哪几种 / 最高 / 最低 / 排序 / 占比」这类**指标明确**的问题，\n"
                "   一律不澄清；\n"
                "4. 只有以下情况才澄清：\n"
                "   - 同一个词在库里有多种合理解释且结果会明显不同（如「销售额」指订单金额还是明细金额）；\n"
                "   - 主语指代不明（如「那个产品」而前文没有指代对象）；\n"
                "   - 问的对象在库里根本没有建模（如库里没有手机品牌维度，却问「苹果手机的销量」）；\n"
                "   - 需要用户提供库里没有的外部条件。\n\n"
                "反例（**不要**澄清）：「上海的客户有多少人？」「会员等级一共有几种？」"
                "「各状态订单的数量分布是怎样的？」\n"
                "正例（**应该**澄清）：「哪个产品卖得好？」（销量还是金额未定）、「帮我看看数据」（毫无信息）。\n\n"
                '只输出 JSON：{"ambiguous": true/false, "question": "要反问用户的那一句话", "confidence": 0.0}'
            ),
        },
    ]


def build_answer_messages(question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence[Any]],
                          *, row_count: int) -> list[dict[str, str]]:
    preview = "\n".join(
        " | ".join(str(v) for v in row) for row in list(rows)[:10]
    )
    return [
        {"role": "system", "content": "你是数据分析助手，用简洁中文回答业务问题。"},
        {
            "role": "user",
            "content": (
                f"【用户问题】{question}\n\n【执行的 SQL】\n{sql}\n\n"
                f"【结果集】共 {row_count} 行，列：{', '.join(columns)}\n{preview}\n\n"
                "请用 1~3 句中文给出结论（带关键数字），不要再解释 SQL 语法。"
            ),
        },
    ]
