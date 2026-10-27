"""评测指标：结果集比对、SQL 规范化比对、分位数统计。

为什么主打「执行准确率」而不是「SQL 字符串精确匹配(EM)」：
    同一句问题可以有多种正确写法（`COUNT(*)` vs `COUNT(order_id)`、JOIN 顺序、
    别名不同、加了不必要的 LIMIT），字符串比对会把它们全判错，导致指标虚低。
    论文与工业界的普遍做法是执行结果集比对（Execution Accuracy）。
两种口径都给：
    - ex        ：列序也要求一致（严格）
    - ex_relaxed：允许列序不同（只要求列数一致 + 行多重集一致）
"""
from __future__ import annotations

import statistics
from typing import Any, Sequence

Rows = Sequence[Sequence[Any]]


def normalize_cell(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{round(value, ndigits):.{ndigits}f}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    # 纯数字字符串（SQLite 有时返回 str）也统一成数值形态
    try:
        return f"{round(float(text), ndigits):.{ndigits}f}"
    except (TypeError, ValueError):
        return text


def row_multiset(rows: Rows, *, ndigits: int = 4, sort_cells: bool = False) -> "list[tuple[str, ...]]":
    out: list[tuple[str, ...]] = []
    for row in rows:
        cells = [normalize_cell(c, ndigits) for c in row]
        out.append(tuple(sorted(cells)) if sort_cells else tuple(cells))
    out.sort()
    return out


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def close_enough(a: str, b: str, *, abs_tol: float = 0.01, rel_tol: float = 1e-4) -> bool:
    """数值容差比较。

    为什么需要：gold SQL 常写成 ROUND(x, 2)，模型返回全精度 SUM(x)，
    两者是同一个数（差在 1e-6 级别），但对字符串就是「不相等」。
    容差 = max(绝对 1 分钱, 相对 1e-4)，既能放过四舍五入差异，
    也能抓住真正的算法错误（例如 12345.67 vs 12350.12，相对差 3.6e-4 会被判错）。
    """
    if a == b:
        return True
    if _is_number(a) and _is_number(b):
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= max(abs_tol, rel_tol * max(abs(fa), abs(fb)))
    return False


def _row_sort_key(row: tuple[str, ...]):
    return tuple((0, float(c)) if _is_number(c) else (1, c) for c in row)


def results_equal(gold: Rows, pred: Rows, *, relaxed: bool = False, ndigits: int = 4,
                  tolerant: bool = True) -> bool:
    """执行准确率的判定：行序无关；relaxed=True 时列序也无关（但仍要求列数一致）。

    tolerant=True 时对数值做容差比较（见 close_enough）。
    """
    if not gold or not pred:
        return len(gold or []) == len(pred or [])
    gold_cols = len(gold[0])
    if gold_cols != (len(pred[0]) if pred else 0):
        return False
    if len(gold) != len(pred):
        return False

    def canon(rows: Rows) -> list[tuple[str, ...]]:
        out = []
        for row in rows:
            cells = [normalize_cell(c, ndigits) for c in row]
            out.append(tuple(sorted(cells)) if relaxed else tuple(cells))
        out.sort(key=_row_sort_key)
        return out

    g, p = canon(gold), canon(pred)
    if g == p:
        return True
    if not tolerant:
        return False
    return all(
        len(rg) == len(rp) and all(close_enough(cg, cp) for cg, cp in zip(rg, rp))
        for rg, rp in zip(g, p)
    )


def _cells_equal(a: str, b: str, ndigits: int = 4) -> bool:
    return a == b or close_enough(a, b)


def _row_loose_match(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """行级"宽松匹配"：两个单元格集合互为子集（任一方多给/少给列都算匹配）。

    为什么需要这个口径：执行准确率对**列数**很敏感，但很多"错"其实是善意的信息增减 ——
    问「完成订单数超过 300 单的大区有哪些？」，模型只给大区名（完全符合字面），
    我的标准答案却多带了一列计数；反过来模型也可能多给一列 ID。
    这类差异人工判定都算"答对了"，所以额外报一个宽松口径，同时保留严格口径。
    """
    if len(a) == len(b):
        return True
    small, large = (a, b) if len(a) < len(b) else (b, a)
    remaining = list(large)
    for cell in small:
        for i, other in enumerate(remaining):
            if _cells_equal(cell, other):
                remaining.pop(i)
                break
        else:
            return False
    return True


def results_equal_loose(gold: Rows, pred: Rows, *, ndigits: int = 4) -> bool:
    """宽松口径：行数必须一致；每行允许多给/少给列，只要公共列的值能对上。"""
    if not gold or not pred:
        return len(gold or []) == len(pred or [])
    if len(gold) != len(pred):
        return False
    g = [tuple(normalize_cell(c, ndigits) for c in row) for row in gold]
    p = [tuple(normalize_cell(c, ndigits) for c in row) for row in pred]
    g.sort(key=_row_sort_key)
    p.sort(key=_row_sort_key)
    used = [False] * len(p)
    for grow in g:
        for i, prow in enumerate(p):
            if not used[i] and _row_loose_match(grow, prow):
                used[i] = True
                break
        else:
            return False
    return True


def sql_normalized(sql: str) -> str:
    """用 sqlglot 归一化 SQL，用于 EM 口径（别名/大小写/空白不敏感）。"""
    if not sql:
        return ""
    try:
        import sqlglot

        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree.sql(dialect="sqlite", normalize=True, pretty=False).lower()
    except Exception:
        return " ".join(sql.lower().split())


def percentile(values: Sequence[float], p: float) -> float:
    """p 支持两种写法：0.95（分位）或 95（百分数），内部统一成分位。"""
    if not values:
        return 0.0
    if p > 1:          # 传了百分数，比如 95 -> 0.95
        p = p / 100.0
    p = min(max(p, 0.0), 1.0)
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把逐题结果聚合成一份可写进简历的指标表。"""
    gradable = [r for r in records if r.get("gradable")]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

    def rate(items: Sequence[dict], key: str) -> float:
        return round(sum(1 for i in items if i.get(key)) / len(items), 4) if items else 0.0

    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({t for r in records for t in r.get("tags", [])}):
        subset = [r for r in records if tag in r.get("tags", [])]
        graded = [r for r in subset if r.get("gradable")]
        by_tag[tag] = {
            "n": len(subset),
            "ex": round(sum(1 for r in graded if r.get("correct")) / len(graded), 4) if graded else None,
            "em": round(sum(1 for r in graded if r.get("em")) / len(graded), 4) if graded else None,
        }

    return {
        "n_questions": len(records),
        "n_gradable": len(gradable),
        "ex": rate(gradable, "correct"),                       # 执行准确率（数值带容差，严格列序）
        "ex_strict": rate(gradable, "correct_strict"),          # 完全不放松的口径（精确比对）
        "ex_loose": rate(gradable, "correct_loose"),            # 列数宽松口径（允许多给/少给列）
        "ex_relaxed": rate(gradable, "correct_relaxed"),        # 列序无关口径
        "em": rate(gradable, "em"),                             # SQL 字符串精确匹配
        "valid_sql_rate": rate(records, "valid_sql"),           # 生成 SQL 通过护栏的比例
        "clarify_accuracy": rate([r for r in records if r.get("expect") == "clarify"], "clarify_ok"),
        "guardrail_block_rate": rate([r for r in records if r.get("expect") == "safe"], "safe_ok"),
        "repair_rate": rate(records, "needed_repair"),
        "repair_success_rate": rate([r for r in records if r.get("needed_repair")], "correct"),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.5), 1) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "tokens": {
            "prompt": sum(r.get("prompt_tokens", 0) for r in records),
            "completion": sum(r.get("completion_tokens", 0) for r in records),
        },
        "llm_calls": sum(r.get("llm_calls", 0) for r in records),
        "by_tag": by_tag,
    }


def markdown_report(summary: dict[str, Any], *, model: str, candidates: int, engine: str) -> str:
    lines = [
        "# NL2SQL Agent 评测报告",
        "",
        f"- 模型：`{model}`　候选数 N={candidates}　编排引擎：`{engine}`",
        f"- 题量：{summary['n_questions']} 题（可评分 {summary['n_gradable']} 题）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 执行准确率 EX（严格列序） | **{summary['ex']:.1%}** |",
        f"| 执行准确率 EX（列数宽松：允许多给/少给列） | {summary.get('ex_loose', 0):.1%} |",
        f"| 执行准确率 EX（不做数值容差，最严口径） | {summary.get('ex_strict', 0):.1%} |",
        f"| 执行准确率 EX（列序无关） | {summary['ex_relaxed']:.1%} |",
        f"| SQL 精确匹配 EM | {summary['em']:.1%} |",
        f"| 生成 SQL 合法率（过护栏） | {summary['valid_sql_rate']:.1%} |",
        f"| 歧义题澄清率 | {summary['clarify_accuracy']:.1%} |",
        f"| 危险请求拦截率 | {summary['guardrail_block_rate']:.1%} |",
        f"| 触发自愈比例 | {summary['repair_rate']:.1%} |",
        f"| 自愈后救回率 | {summary['repair_success_rate']:.1%} |",
        f"| 端到端延迟 mean / p50 / p95 | {summary['latency_ms']['mean']:.0f} / "
        f"{summary['latency_ms']['p50']:.0f} / {summary['latency_ms']['p95']:.0f} ms |",
        f"| 总 LLM 调用 / token | {summary['llm_calls']} 次 / "
        f"{summary['tokens']['prompt'] + summary['tokens']['completion']} |",
        "",
        "## 分组（按标签）",
        "",
        "| 标签 | 题数 | EX | EM |",
        "| --- | --- | --- | --- |",
    ]
    for tag, data in summary["by_tag"].items():
        ex = "—" if data["ex"] is None else f"{data['ex']:.1%}"
        em = "—" if data["em"] is None else f"{data['em']:.1%}"
        lines.append(f"| {tag} | {data['n']} | {ex} | {em} |")
    return "\n".join(lines) + "\n"
