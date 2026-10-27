# -*- coding: utf-8 -*-
"""DuSQL gold SQL 规范化 v2（带 schema 知识，目标：可执行率 >93%）。

修四类问题：
    1. `from 表 as T1` / `from 表 T1` 别名后仍用原表名引用 -> 去别名并统一引用；
    2. `( select A ) except ( select B )` 顶层括号 -> SQLite 不认，去掉外层括号；
    3. 标识符（表名/列名）没加引号，尤其 `2018年营业额` 这种数字开头 -> 用 schema 精确加引号；
    4. `TIME_NOW` 是 DuSQL 自定义伪函数，语义无法忠实还原 -> 标记为不可评测（跳过）。

输出：E:\\dddddd\\nl2sql数据\\dusql_eval\\dev_executable.json（覆盖 v1）
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

SRC = Path(r"E:\dddddd\workbuddy中文数据集\DuSQL")
DB = Path(r"E:\dddddd\nl2sql数据\dusql_db")
OUT = Path(r"E:\dddddd\nl2sql数据\dusql_eval")

_ALIAS_AS = re.compile(r'\b(\S+)\s+as\s+(T\d+)\b', re.I)
_ALIAS_BARE = re.compile(r'\b(from|join)\s+(\S+)\s+(T\d+)\b(?!\s*[,)]*\s*(on|where|join|left|inner))', re.I)
_TOP_PAREN = re.compile(r'^\s*\(\s*(select\b.*?)\s*\)\s*(except|union|intersect)\s*\(\s*(select\b.*?)\s*\)\s*$', re.I | re.S)


def names_of(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    tables, cols = set(), set()
    for (t,) in conn.execute("select name from sqlite_master where type='table'"):
        tables.add(t)
        for r in conn.execute(f'PRAGMA table_info("{t}")'):
            cols.add(r[1])
    return tables, cols


def strip_aliases(sql: str) -> str:
    mapping: dict[str, str] = {}
    for m in _ALIAS_AS.finditer(sql):
        mapping[m.group(2)] = m.group(1).strip('"')
    for m in _ALIAS_BARE.finditer(sql):
        mapping.setdefault(m.group(3), m.group(2).strip('"'))
    if not mapping:
        return sql
    if len(set(mapping.values())) != len(mapping.values()):
        raise ValueError("self-join")
    s = sql
    for alias, table in mapping.items():
        s = re.sub(rf'\b{alias}\.', f'{table}.', s)
        s = re.sub(rf'\s+as\s+{alias}\b', '', s, flags=re.I)
        s = re.sub(rf'(\b(?:from|join)\s+{re.escape(table)})\s+{alias}\b', r'\1', s, flags=re.I)
    return s


def quote_names(sql: str, tables: set[str], cols: set[str]) -> str:
    names = sorted(tables | cols, key=len, reverse=True)
    out = sql
    for name in names:
        pat = re.compile(rf'(?<![\w"\u4e00-\u9fa5]){re.escape(name)}(?![\w"\u4e00-\u9fa5])')
        out = pat.sub(f'"{name}"', out)
    # 已经带引号的不要重复加（"\"表\"" -> "\"表\""）
    out = re.sub(r'""([^"]+)""', r'"\1"', out)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((DB / "_manifest.json").read_text(encoding="utf-8"))
    dev = json.loads((SRC / "dev.json").read_text(encoding="utf-8"))
    schema_cache: dict[str, tuple[set[str], set[str]]] = {}

    ok, bad, reasons = [], [], {}
    for q in dev:
        db_id = q["db_id"]
        f = man.get(db_id)
        if not f:
            reasons["没有库"] = reasons.get("没有库", 0) + 1
            continue
        if "TIME_NOW" in q["query"].upper():
            reasons["TIME_NOW 伪函数（语义不可还原）"] = reasons.get("TIME_NOW 伪函数（语义不可还原）", 0) + 1
            continue
        path = DB / f
        if db_id not in schema_cache:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            schema_cache[db_id] = names_of(conn)
            conn.close()
        tables, cols = schema_cache[db_id]
        try:
            sql = q["query"].strip().rstrip(";")
            m = _TOP_PAREN.match(sql)
            if m:
                sql = f"{m.group(1)} {m.group(2)} {m.group(3)}"
            sql = strip_aliases(sql)
            sql = quote_names(sql, tables, cols)
        except ValueError as exc:
            reasons[f"规范化失败:{exc}"] = reasons.get(f"规范化失败:{exc}", 0) + 1
            continue
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                rows = conn.execute(sql).fetchall()
            finally:
                conn.close()
            ok.append({
                "question_id": q.get("question_id"),
                "db_id": db_id,
                "db_file": f,
                "question": q["question"],
                "sql_gold": sql,
                "sql_raw": q["query"],
                "gold_rows": len(rows),
            })
        except sqlite3.Error as exc:
            key = f"{type(exc).__name__}: {str(exc)[:44]}"
            reasons[key] = reasons.get(key, 0) + 1
            bad.append({"db_id": db_id, "sql": sql, "error": str(exc)[:140]})

    total = len(dev)
    print(f"可执行 {len(ok)}/{total} = {len(ok) / total * 100:.1f}%")
    print("未纳入原因 top6：")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:6]:
        print(f"   {v:>4}x {k}")
    (OUT / "dev_executable.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "dev_failed.json").write_text(json.dumps(bad, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT / 'dev_executable.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
