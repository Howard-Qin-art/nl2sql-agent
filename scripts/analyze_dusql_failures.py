# -*- coding: utf-8 -*-
"""DuSQL 错题归因（和 BIRD 那版同一套思路，库路径按 manifest 解析）。

用法：python scripts/analyze_dusql_failures.py <results.jsonl>
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_DIR = Path(r"E:\dddddd\nl2sql数据\dusql_db")
MAN = json.loads((DB_DIR / "_manifest.json").read_text(encoding="utf-8"))

_LIMIT_RE = re.compile(r"limit\s+(\d+)\s*$", re.I)


def columns_of(db: Path, sql: str) -> list[str]:
    try:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            cur.fetchall()
            return cols
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def classify(rec: dict) -> str:
    pred = (rec.get("pred_sql") or "").strip()
    if not pred:
        return "1-生成失败"
    if rec.get("error"):
        return "2-执行失败"
    db = DB_DIR / MAN.get(rec["db_id"], "")
    if not db.exists():
        return "9-无库"
    g_cols = columns_of(db, rec["gold_sql"])
    p_cols = columns_of(db, pred)
    if p_cols and g_cols and len(p_cols) != len(g_cols):
        return "4-列数不对"
    if p_cols and g_cols and [c.lower() for c in p_cols] != [c.lower() for c in g_cols]:
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
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    wrong = [r for r in rows if not r["ex"]]
    print(f"总题数 {len(rows)} | 错题 {len(wrong)} | EX {len(rows) - len(wrong)}/{len(rows)} = "
          f"{(len(rows) - len(wrong)) / len(rows) * 100:.1f}%")

    buckets: dict[str, list[dict]] = {}
    for r in wrong:
        buckets.setdefault(classify(r), []).append(r)
    print("\n错题归因：")
    for name in sorted(buckets):
        items = buckets[name]
        print(f"  {name:<12} {len(items):>2} 题 ({len(items) / len(wrong) * 100:.0f}% of 错题)")
        for it in items[: args.show]:
            print(f"      · [{it['db_id']}] {it['question'][:44]}")
            print(f"        gold({it['gold_rows']}行): {it['gold_sql'][:100].replace(chr(10),' ')}")
            print(f"        pred({it['pred_rows']}行): {(it['pred_sql'] or '(空)')[:100].replace(chr(10),' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
