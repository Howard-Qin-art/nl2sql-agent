# -*- coding: utf-8 -*-
"""离线复评：不重跑模型，只把已有评测结果按「最新指标口径」重新判分。

用途：
    - 改了判分逻辑（例如加了数值容差、改了列序口径）后，不必再花一小时重跑 LLM；
    - 评测过程中断，也能用已落盘的逐题结果生成完整报告。

做法：读出 JSONL 里每条记录的 gold_sql / pred_sql，在本地重新执行并比对。

用法：
    python scripts/regrade.py                       # 自动挑最新的一个评测结果
    python scripts/regrade.py --file eval/results/full-xxx.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.db.executor import SQLExecutor  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.guardrails import validate_sql  # noqa: E402
from eval.metrics import (  # noqa: E402
    markdown_report,
    results_equal,
    results_equal_loose,
    sql_normalized,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="按最新指标口径离线复评已有结果")
    parser.add_argument("--file", default=None, help="评测结果 JSONL；默认取最新的 full-*.jsonl")
    parser.add_argument("--tag", default="regraded", help="输出文件前缀")
    args = parser.parse_args()

    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    docs = introspect(SETTINGS.db_path)
    allowed = {d.name for d in docs}
    executor = SQLExecutor(SETTINGS.db_path, timeout_s=SETTINGS.query_timeout_s, max_rows=SETTINGS.max_rows)

    results_dir = ROOT / "eval" / "results"
    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = ROOT / path
    else:
        candidates = sorted(results_dir.glob("full-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("没有找到 full-*.jsonl")
            return 1
        path = candidates[0]
    print("复评文件:", path)

    records: list[dict] = []
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        expect = rec.get("expect", "rows")
        gold_sql, pred_sql = rec.get("gold_sql") or "", rec.get("pred_sql") or ""
        if expect in ("rows", "empty"):
            gold_rows: list[tuple] = []
            if gold_sql:
                g = validate_sql(gold_sql, default_limit=SETTINGS.default_limit, allowed_tables=allowed)
                res = executor.run(g.sql if g.ok else gold_sql)
                gold_rows = [tuple(r) for r in res.rows] if res.ok else []
            pred_rows: list[tuple] = []
            if pred_sql:
                p = validate_sql(pred_sql, default_limit=SETTINGS.default_limit, allowed_tables=allowed)
                res = executor.run(p.sql if p.ok else pred_sql)
                pred_rows = [tuple(r) for r in res.rows] if res.ok else []
            old = bool(rec.get("correct"))
            rec["correct"] = bool(pred_sql) and results_equal(gold_rows, pred_rows)
            rec["correct_strict"] = bool(pred_sql) and results_equal(gold_rows, pred_rows, tolerant=False)
            rec["correct_loose"] = bool(pred_sql) and results_equal_loose(gold_rows, pred_rows)
            rec["correct_relaxed"] = bool(pred_sql) and results_equal(gold_rows, pred_rows, relaxed=True)
            rec["em"] = bool(pred_sql) and sql_normalized(gold_sql) == sql_normalized(pred_sql)
            rec["gold_row_count"] = len(gold_rows)
            rec["row_count"] = len(pred_rows)
            rec["regraded"] = True
            if old != rec["correct"]:
                changed += 1
        elif expect == "clarify":
            rec["gradable"] = False
            rec["clarify_ok"] = bool(rec.get("clarified"))
        elif expect == "safe":
            rec["gradable"] = False
            rec["safe_ok"] = bool(rec.get("blocked_any")) or (
                rec.get("valid_sql") and str(pred_sql).upper().lstrip().startswith(("SELECT", "WITH")))
        records.append(rec)

    summary = summarize(records)
    summary["source_file"] = path.name
    summary["regraded"] = True
    summary["regrade_changed_verdicts"] = changed
    # 模型名从原始 summary 继承（原来硬编码成本地模型名，云端复评会被写错）
    base_model = None
    sibling = path.with_name(path.stem + "_summary.json")
    if sibling.exists():
        try:
            base_model = json.loads(sibling.read_text(encoding="utf-8")).get("model")
        except (json.JSONDecodeError, OSError):
            base_model = None
    summary["model"] = summary.get("model") or base_model or "unknown"

    base = results_dir / f"{args.tag}-{path.stem}"
    out_jsonl = base.with_suffix(".jsonl")
    out_summary = base.parent / f"{base.name}_summary.json"
    out_report = base.parent / f"{base.name}_report.md"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report.write_text(markdown_report(summary, model=summary["model"],
                                          candidates=summary.get("candidates", 3),
                                          engine=summary.get("engine", "core")), encoding="utf-8")

    print(f"复评 {len(records)} 题，其中 {changed} 题判定发生变化")
    print(f"EX(带容差) = {summary['ex']:.1%} | EX(最严) = {summary['ex_strict']:.1%} | "
          f"EX(列序无关) = {summary['ex_relaxed']:.1%} | EM = {summary['em']:.1%}")
    print(f"澄清率 = {summary['clarify_accuracy']:.1%} | 拦截率 = {summary['guardrail_block_rate']:.1%}")
    print(f"输出：{out_jsonl}\n     {out_summary}\n     {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
