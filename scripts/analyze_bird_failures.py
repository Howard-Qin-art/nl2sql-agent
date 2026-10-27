# -*- coding: utf-8 -*-
"""把 BIRD 评测的错题按原因归类，避免只报一个百分数。

分类规则（按优先级）：
    1. 生成失败     预测 SQL 为空 / 报错
    2. 执行失败     SQL 语法或列名错误
    3. LIMIT 截断   预测行数恰好等于 LIMIT 且少于 gold（护栏补的 LIMIT 干的好事）
    4. 列选错       列数不同，或列名完全不同
    5. 过滤/聚合错  列对得上但行数不同，或行数相同但值不同
    6. 其他

用法：python scripts/analyze_bird_failures.py <results.jsonl>
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.bird import db_path  # noqa: E402


def columns_of(db: Path, sql: str) -> list[str]:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            cur.fetchall()
            return cols
        finally:
            conn.close()
    except sqlite3.Error:
        return []


_LIMIT_RE = re.compile(r"limit\s+(\d+)\s*$", re.I)


def classify(rec: dict) -> str:
    pred = (rec.get("pred_sql") or "").strip()
    if not pred:
        return "1-生成失败"
    if rec.get("error"):
        return "2-执行失败"
    db = db_path(rec["db_id"])
    gold_cols = columns_of(db, rec["gold_sql"])
    pred_cols = columns_of(db, pred)
    if pred_cols and gold_cols and len(pred_cols) != len(gold_cols):
        return "4-列数不对"
    if pred_cols and gold_cols and [c.lower() for c in pred_cols] != [c.lower() for c in gold_cols]:
        return "4-列选错"
    m = _LIMIT_RE.search(pred)
    if m and rec["pred_rows"] < rec["gold_rows"] and int(m.group(1)) == rec["pred_rows"]:
        return "3-LIMIT截断"
    if rec["pred_rows"] != rec["gold_rows"]:
        return "5-过滤/聚合错"
    return "6-取值差异"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--show", type=int, default=3, help="每类展示几个例子")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    wrong = [r for r in rows if not r["ex"]]
    print(f"总题数 {len(rows)} | 错题 {len(wrong)} | EX {len(rows) - len(wrong)}/{len(rows)} = "
          f"{(len(rows) - len(wrong)) / len(rows) * 100:.1f}%")

    buckets: dict[str, list[dict]] = {}
    for r in wrong:
        buckets.setdefault(classify(r), []).append(r)

    print("\n错题归因：")
    for name in sorted(buckets):
        items = buckets[name]
        print(f"  {name:<12} {len(items):>2} 题  ({len(items) / len(wrong) * 100:.0f}% of 错题)")
        for it in items[: args.show]:
            print(f"      · [{it['db_id']}] {it['question'][:46]}")
            print(f"        gold: {it['gold_sql'][:110].replace(chr(10), ' ')}")
            print(f"        pred: {(it['pred_sql'] or '(空)')[:110].replace(chr(10), ' ')}")
    print("\n按难度：")
    for diff in ("simple", "moderate", "challenging"):
        sub = [r for r in rows if r.get("difficulty") == diff]
        if sub:
            ok = sum(r["ex"] for r in sub)
            print(f"  {diff:<12} {ok}/{len(sub)} = {ok / len(sub) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
