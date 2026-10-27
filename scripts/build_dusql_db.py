# -*- coding: utf-8 -*-
"""把 DuSQL 的 schema + content JSON 建成 SQLite 库（DuSQL 官方没发 .sqlite，只发了 JSON）。

输入（workbuddy 下载）：
    E:\\dddddd\\workbuddy中文数据集\\DuSQL\\
        db_schema.json    # 208 个库：db_id / table_names / column_names
        db_content.json   # 208 个库：{db_id: {tables: {<表名>: {cell: [[值...], ...]}}}}
        dev.json          # 中文题 + 可执行 SQL 字符串（字段 query）

输出：
    E:\\dddddd\\nl2sql数据\\dusql_db\\<db_id>.sqlite
    E:\\dddddd\\nl2sql数据\\dusql_db\\_manifest.json   # db_id -> 文件名（处理非法字符）

类型推断：按每列内容判断 INTEGER / REAL / TEXT，并依赖 SQLite 的列亲和性把
"1000" 这类字符串自动转成整数 —— 这样 gold SQL 和预测 SQL 在同一套语义下比较。
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

SRC = Path(r"E:\dddddd\workbuddy中文数据集\DuSQL")
OUT = Path(r"E:\dddddd\nl2sql数据\dusql_db")

_ILLEGAL = re.compile(r'[\\/:*?"<>|]')


def safe_name(db_id: str) -> str:
    return _ILLEGAL.sub("_", db_id).strip() or "db"


def infer_type(values: list[str]) -> str:
    seen_num = seen_float = False
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            int(s)
            seen_num = True
            continue
        except ValueError:
            pass
        try:
            float(s)
            seen_num = seen_float = True
            continue
        except ValueError:
            return "TEXT"
    if seen_float:
        return "REAL"
    return "INTEGER" if seen_num else "TEXT"


def build_one(schema: dict, content: dict, out_dir: Path) -> tuple[str, int, int]:
    db_id = schema["db_id"]
    tables: list[str] = schema.get("table_names") or []
    cols: list[list] = schema.get("column_names") or []

    per_table: dict[str, list[str]] = {}
    for item in cols:
        tidx, cname = item[0], item[1]
        if tidx < 0:
            continue
        per_table.setdefault(tables[tidx], []).append(cname)

    data = (content.get("tables") or {})
    path = out_dir / f"{safe_name(db_id)}.sqlite"
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    n_tables = n_rows = 0
    try:
        for t in tables:
            tcols = per_table.get(t) or []
            if not tcols:
                continue
            # DuSQL 少数库里同一张表出现了重复列名（如「投用日期」两次）：
            # 只保留第一次出现，建表/插数据都按这个映射来，避免 CREATE TABLE 直接报错。
            kept: list[tuple[int, str]] = []
            seen: set[str] = set()
            for idx, cname in enumerate(tcols):
                if cname in seen:
                    continue
                seen.add(cname)
                kept.append((idx, cname))
            cells = ((data.get(t) or {}).get("cell")) or []
            # 按列推断类型
            types = []
            for src_idx, _ in kept:
                col_vals = [row[src_idx] if src_idx < len(row) else None for row in cells]
                types.append(infer_type(col_vals))
            ddl = ", ".join(f'"{c}" {ty}' for (_, c), ty in zip(kept, types))
            conn.execute(f'CREATE TABLE "{t}" ({ddl})')
            if cells:
                placeholders = ", ".join("?" for _ in kept)
                rows = []
                for row in cells:
                    rows.append(tuple(row[src_idx] if src_idx < len(row) else None for src_idx, _ in kept))
                conn.executemany(f'INSERT INTO "{t}" VALUES ({placeholders})', rows)
                n_rows += len(rows)
            n_tables += 1
        conn.commit()
    finally:
        conn.close()
    return path.name, n_tables, n_rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    schema_list = json.loads((SRC / "db_schema.json").read_text(encoding="utf-8"))
    content_list = json.loads((SRC / "db_content.json").read_text(encoding="utf-8"))
    content_by_id = {d["db_id"]: d for d in content_list}
    print(f"库数 {len(schema_list)} | 内容条目 {len(content_by_id)}")

    manifest: dict[str, str] = {}
    total_tables = total_rows = 0
    for i, s in enumerate(schema_list, 1):
        db_id = s["db_id"]
        try:
            fname, nt, nr = build_one(s, content_by_id.get(db_id, {}), OUT)
        except Exception as exc:  # noqa: BLE001  单个库坏了不影响整体
            print(f"  ⚠️ 跳过 {db_id}: {type(exc).__name__}: {exc}")
            continue
        manifest[db_id] = fname
        total_tables += nt
        total_rows += nr
        if i % 40 == 0 or i == len(schema_list):
            print(f"  [{i}/{len(schema_list)}] {db_id} -> {fname} ({nt} 表 / {nr} 行)")
    (OUT / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 建库完成：{len(manifest)} 个库，{total_tables} 张表，{total_rows} 行 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
