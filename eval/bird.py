# -*- coding: utf-8 -*-
"""BIRD mini_dev 适配器：把公开数据集变成项目能用的「表画像 + 题库」。

BIRD 的数据布局（本机）：
    E:\\dddddd\\nl2sql数据\\bird_mini_dev\\minidev\\MINIDEV\\
        mini_dev_sqlite.json                 # 500 题：question / db_id / SQL / evidence / difficulty
        dev_tables.json                      # 11 个库的 schema（表名、列名、类型、主外键）
        dev_databases\\<db_id>\\<db_id>.sqlite
        dev_databases\\<db_id>\\database_description\\<table>.csv   # 列说明、取值说明

这里只做「读数据 + 造 TableDoc」，不碰 LLM；真正的评测在 scripts/eval_bird.py。
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.db.schema import ColumnDoc, TableDoc

BIRD_DIR = Path(r"E:\dddddd\nl2sql数据\bird_mini_dev\minidev\MINIDEV")

# database_description 里列名可能的写法（不同库的 CSV 表头略有差异）
_DESC_KEYS = {
    "column": ("original_column_name", "column_name", "column"),
    "description": ("column_description", "description"),
    "value": ("value_description",),
    "format": ("data_format",),
}


def _pick(row: dict[str, str], names: tuple[str, ...]) -> str:
    for n in names:
        for k, v in row.items():
            if k and k.strip().lower() == n:
                return (v or "").strip()
    return ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, OSError):
            continue
    return []


def load_questions(bird_dir: Path | None = None, *, limit: int | None = None,
                   difficulty: str | None = None) -> list[dict[str, Any]]:
    base = Path(bird_dir or BIRD_DIR)
    data = json.loads((base / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    if difficulty:
        data = [q for q in data if str(q.get("difficulty", "")).lower() == difficulty.lower()]
    if limit:
        data = data[:limit]
    return data


def load_schema_meta(bird_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """dev_tables.json -> {db_id: schema}"""
    base = Path(bird_dir or BIRD_DIR)
    tables = json.loads((base / "dev_tables.json").read_text(encoding="utf-8"))
    return {t["db_id"]: t for t in tables}


def db_path(db_id: str, bird_dir: Path | None = None) -> Path:
    base = Path(bird_dir or BIRD_DIR)
    return base / "dev_databases" / db_id / f"{db_id}.sqlite"


def _table_descriptions(db_id: str, bird_dir: Path | None = None) -> dict[str, dict[str, dict[str, str]]]:
    """{table: {column: {description, value, format}}}"""
    base = Path(bird_dir or BIRD_DIR)
    desc_dir = base / "dev_databases" / db_id / "database_description"
    out: dict[str, dict[str, dict[str, str]]] = {}
    if not desc_dir.exists():
        return out
    for csv_path in desc_dir.glob("*.csv"):
        table = csv_path.stem
        per_col: dict[str, dict[str, str]] = {}
        for row in _read_rows(csv_path):
            col = _pick(row, _DESC_KEYS["column"])
            if not col:
                continue
            per_col[col] = {
                "description": _pick(row, _DESC_KEYS["description"]),
                "value": _pick(row, _DESC_KEYS["value"]),
                "format": _pick(row, _DESC_KEYS["format"]),
            }
        out[table] = per_col
    return out


def _row_count(conn: sqlite3.Connection, table: str, cap: int = 5_000_000) -> int:
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except sqlite3.Error:
        return 0


def _samples(conn: sqlite3.Connection, table: str, column: str, limit: int = 3) -> list[str]:
    try:
        rows = conn.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?',
                            (limit,)).fetchall()
        return [str(r[0]) for r in rows]
    except sqlite3.Error:
        return []


def build_table_docs(db_id: str, bird_dir: Path | None = None, meta: dict[str, Any] | None = None,
                     *, with_samples: bool = True, sample_row_cap: int = 200_000) -> list[TableDoc]:
    """按 BIRD 的 schema + CSV 列说明，构造项目里的 TableDoc 列表。"""
    meta = meta or load_schema_meta(bird_dir)[db_id]
    path = db_path(db_id, bird_dir)
    descs = _table_descriptions(db_id, bird_dir)

    table_names: list[str] = meta["table_names_original"]
    col_pairs: list[list[Any]] = meta["column_names_original"]      # [[table_idx, col], ...]
    col_types: list[str] = meta["column_types"]
    # 主键可能是「单列 idx」也可能是「复合主键的 idx 列表」，统一摊平
    pk_idx: set[int] = set()
    for item in (meta.get("primary_keys") or []):
        if isinstance(item, list):
            pk_idx.update(int(x) for x in item)
        else:
            pk_idx.add(int(item))
    fk_pairs = meta.get("foreign_keys") or []                        # [[from_idx, to_idx], ...]

    # 列的索引 -> (表名, 列名)
    def _col(idx: int) -> tuple[str, str]:
        tidx, cname = col_pairs[idx]
        return ("*" if tidx < 0 else table_names[tidx]), cname

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        docs: list[TableDoc] = []
        for tidx, tname in enumerate(table_names):
            cols: list[ColumnDoc] = []
            for cidx in range(len(col_pairs)):
                owner, cname = _col(cidx)      # [表索引, 列名] -> (表名, 列名)
                if owner != tname:
                    continue
                fk_table = fk_col = None
                for f, t in fk_pairs:
                    if f == cidx:
                        fk_table, fk_col = _col(t)
                        break
                d = (descs.get(tname) or {}).get(cname) or {}
                note_bits = [d.get("description", ""), d.get("value", ""), d.get("format", "")]
                cols.append(ColumnDoc(
                    name=cname,
                    type=col_types[cidx] if cidx < len(col_types) else "",
                    notnull=False,
                    pk=cidx in pk_idx,
                    fk_table=fk_table,
                    fk_column=fk_col,
                    samples=[],
                    note="；".join(b for b in note_bits if b)[:200],
                ))
            rows = _row_count(conn, tname)
            if with_samples and rows <= sample_row_cap:
                for c in cols:
                    c.samples = _samples(conn, tname, c.name)
            table_note = ""
            for c in cols:
                if c.note:
                    table_note = c.note[:120]
                    break
            docs.append(TableDoc(
                name=tname,
                row_count=rows,
                columns=cols,
                note=table_note,
                fks=[(c.name, c.fk_table, c.fk_column) for c in cols if c.fk_table],
            ))
        return docs
    finally:
        conn.close()


def difficulty_of(question: dict[str, Any]) -> str:
    return str(question.get("difficulty") or "unknown").lower()
