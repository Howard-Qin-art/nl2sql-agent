"""SQL 安全护栏：基于 sqlglot AST 的只读白名单校验。

为什么必须是 AST 而不是正则/关键词黑名单：
    - 字符串里出现 "drop table" 会误杀，注释里藏 `/**/DELETE` 会漏杀；
    - AST 能稳定回答「这是什么语句」「碰了哪些表」「有没有 LIMIT」。
护栏做什么：
    1. 单语句（挡 `SELECT 1; DROP TABLE orders`）
    2. 只允许查询类语句（SELECT / UNION / 子查询 / CTE），其余全拒
    3. 禁止系统表（sqlite_master）与危险函数（load_extension 等）
    4. 强制补 LIMIT（防「SELECT * FROM 大表」把结果集打爆）
    5. 输出口语化错误信息 —— 这些信息会回注给模型做自愈，所以要能看懂
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

DIALECT = "sqlite"

# 允许的最外层语句类型
ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)
# 出现即拒绝的节点类型
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Pragma, exp.Attach, exp.Detach, exp.Command, exp.Transaction,
    exp.Commit, exp.Rollback, exp.Grant, exp.Copy, exp.Merge, exp.TruncateTable,
)
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
SYSTEM_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master"}
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.I),
    re.compile(r"(你现在是|假装你是|进入开发者模式|忽略(之前|上面|以上)|无视(之前|上面|以上))", re.I),
    re.compile(r"drop\s+table", re.I),
    re.compile(r"delete\s+from", re.I),
    # ---- 中文写意图：之前只拦英文 SQL 模式，导致"把订单表删掉"这类说法漏过去 ----
    # 只匹配"动词 + 数据对象"的组合，避免把「已取消的订单有多少」这类查询误判
    re.compile(r"(删|清空|清除|截断|抹掉|干掉)\s*(掉|除)?\s*(表|库|数据|记录|订单|商品|客户|明细)"),
    re.compile(r"把.{0,12}(删掉|删除|清空|改成|改一下|更新成|插入|写入|覆盖)"),
    re.compile(r"(改成|改为|修改|更新为|置为)\s*(已完成|已取消|已支付|已退款|成功|无效)"),
    re.compile(r"(插入|新增|写入|导入)\s*(一条|几条|一批|数据|记录)"),
]


@dataclass
class GuardResult:
    ok: bool
    sql: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    forced_limit: bool = False
    statement_type: str = ""

    def feedback(self) -> str:
        """给模型看的错误反馈。"""
        return "; ".join(self.errors) if self.errors else "OK"


def _cte_names(tree: exp.Expression) -> set[str]:
    """CTE 的名字不是真实表，必须从表名集合里排除，否则 `WITH t AS (...) SELECT * FROM t` 会被误杀。"""
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}


def _collect_tables(tree: exp.Expression) -> list[str]:
    ctes = _cte_names(tree)
    names: list[str] = []
    for t in tree.find_all(exp.Table):
        name = t.name
        if name and name.lower() not in ctes and name not in names:
            names.append(name)
    return names


def _function_names(func: exp.Func) -> set[str]:
    """sqlglot 对未知函数用 Anonymous 表示，此时真正的函数名在 .name 上。"""
    names: set[str] = set()
    try:
        sql_name = func.sql_name()
        if sql_name:
            names.add(sql_name.lower())
    except Exception:
        pass
    raw_name = getattr(func, "name", None)
    if isinstance(raw_name, str) and raw_name:
        names.add(raw_name.lower())
    return names


def _has_limit(tree: exp.Expression) -> bool:
    if tree.args.get("limit") is not None:
        return True
    # 子查询里有限制不算外层有限制
    for node in tree.find_all(exp.Limit):
        if node.parent is tree or (node.parent and node.parent.parent is tree):
            return True
    return False


def _force_limit(tree: exp.Expression, limit: int) -> tuple[exp.Expression, bool]:
    if _has_limit(tree):
        return tree, False
    try:
        return tree.limit(limit, copy=False), True  # type: ignore[attr-defined]
    except Exception:
        wrapped = sqlglot.parse_one(f"SELECT * FROM ({tree.sql(dialect=DIALECT)}) LIMIT {limit}", read=DIALECT)
        return wrapped, True


def validate_sql(
    sql: str,
    *,
    default_limit: int = 100,
    allowed_tables: set[str] | None = None,
    max_sql_length: int = 4000,
) -> GuardResult:
    """校验一条 SQL；返回可能被改写过（补 LIMIT）的 SQL。"""
    if not sql or not sql.strip():
        return GuardResult(ok=False, errors=["SQL 为空"])
    raw = sql.strip()
    if len(raw) > max_sql_length:
        return GuardResult(ok=False, errors=[f"SQL 过长（>{max_sql_length} 字符），拒绝执行"])

    # 预处理：去掉 markdown 围栏
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()

    try:
        statements = [s for s in sqlglot.parse(raw, read=DIALECT) if s is not None]
    except Exception as exc:  # sqlglot 解析失败 = 语法错误
        return GuardResult(ok=False, errors=[f"SQL 解析失败（语法错误）：{str(exc).splitlines()[0][:200]}"])

    if not statements:
        return GuardResult(ok=False, errors=["没有解析出任何语句"])
    if len(statements) > 1:
        return GuardResult(
            ok=False,
            errors=[f"只允许单条语句，检测到 {len(statements)} 条（多条语句通常是注入或模型跑偏）"],
        )

    tree = statements[0]
    stmt_type = type(tree).__name__
    errors: list[str] = []
    warnings: list[str] = []

    # 1) 语句类型白名单
    if not isinstance(tree, ALLOWED_ROOTS):
        errors.append(f"只允许 SELECT 查询，检测到 {stmt_type}（本系统为只读查询 Agent）")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES) and not isinstance(node, ALLOWED_ROOTS):
            errors.append(f"语句中包含被禁止的操作：{type(node).__name__}")
            break

    # 2) 系统表 / 危险函数
    tables = _collect_tables(tree)
    for name in tables:
        if name.lower() in SYSTEM_TABLES:
            errors.append(f"禁止访问系统表 {name}")
    if allowed_tables is not None:
        unknown = [t for t in tables if t not in allowed_tables]
        if unknown:
            errors.append("引用了不存在的表：" + "、".join(unknown) + "（请只用 schema 里给出的表）")
    for func in tree.find_all(exp.Func):
        bad = _function_names(func) & FORBIDDEN_FUNCTIONS
        if bad:
            errors.append(f"禁止调用危险函数 {sorted(bad)[0]}()")

    # 3) 质量类警告（不拦截，但会写进 trace，方便复盘）
    if any(isinstance(t, exp.Star) for t in tree.find_all(exp.Star)):
        warnings.append("使用了 SELECT *，建议显式列出需要的列")
    for join in tree.find_all(exp.Join):
        if not join.args.get("on") and not join.args.get("using"):
            warnings.append("存在没有连接条件的 JOIN（笛卡尔积风险）")

    if errors:
        return GuardResult(ok=False, sql=raw, errors=errors, warnings=warnings, tables=tables,
                           statement_type=stmt_type)

    # 4) 强制 LIMIT
    tree2, forced = _force_limit(tree, default_limit)
    final_sql = tree2.sql(dialect=DIALECT)
    if forced:
        warnings.append(f"原 SQL 没有 LIMIT，已自动补 LIMIT {default_limit}")
    return GuardResult(ok=True, sql=final_sql, errors=[], warnings=warnings, tables=_collect_tables(tree2),
                       forced_limit=forced, statement_type=stmt_type)


def check_question(question: str) -> list[str]:
    """对用户自然语言输入做浅层检查（提示注入/危险意图），只提示不拦截。

    注意只做"提示"：真正的安全靠 SQL 层的 AST 白名单 + 只读连接。
    这里是尽早识别意图，好让上层给出"我只支持只读查询"的回复，而不是傻乎乎去澄清或生成 SQL。
    """
    hits: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(question):
            hits.append(f"输入命中可疑模式：{pattern.pattern}")
    return hits


def normalize_result_rows(rows: list[tuple] | list[list]) -> list[tuple]:
    """结果集规范化：数值统一到 4 位小数、NULL 统一成 <NULL>、行内排序 -> 供投票比对。"""
    normalized: list[tuple] = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            if cell is None:
                cells.append("<NULL>")
            elif isinstance(cell, float):
                cells.append(f"{round(cell, 4):.4f}")
            elif isinstance(cell, (int,)) and not isinstance(cell, bool):
                cells.append(str(cell))
            else:
                cells.append(str(cell).strip())
        normalized.append(tuple(sorted(cells)))
    normalized.sort()
    return normalized


def _demo() -> None:  # pragma: no cover - 手工验证用
    cases: list[tuple[str, bool]] = [
        ("SELECT COUNT(*) FROM orders", True),
        ("SELECT * FROM orders", True),
        ("WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t", True),
        ("SELECT product_id FROM order_item UNION SELECT product_id FROM product", True),
        ("DROP TABLE orders", False),
        ("DELETE FROM orders", False),
        ("UPDATE orders SET total_amount=0", False),
        ("SELECT 1; DROP TABLE orders", False),
        ("PRAGMA table_info(orders)", False),
        ("ATTACH DATABASE 'x' AS y", False),
        ("SELECT * FROM sqlite_master", False),
        ("SELECT load_extension('evil')", False),
        ("", False),
    ]
    for sql, expect_ok in cases:
        res = validate_sql(sql, allowed_tables={"orders", "order_item", "product", "customer"})
        flag = "OK " if res.ok == expect_ok else "!! "
        print(f"{flag}{sql[:52]:<54} ok={res.ok} -> {res.sql[:60] if res.ok else res.feedback()}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
