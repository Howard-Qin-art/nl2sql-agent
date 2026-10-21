"""库表自省：把 SQLite 的元数据 + 数据画像 + 业务术语，变成可以喂给 LLM 的「表/列文档」。

Schema Linking 的召回质量 90% 取决于这里：光有 DDL 不够，还要有
    - 行数、空值率、去重值数量（帮模型判断该不该用某个列做聚合/过滤）
    - 抽样值（帮模型把「华东」「app」这类业务词映射到列）
    - 业务注释（业务术语表，如「GMV」「客单价」「复购」对应哪些表和口径）
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 业务术语表
TABLE_NOTES: dict[str, str] = {
    "orders": "订单主表。GMV/成交额口径=status='completed' 的 total_amount 之和；order_date 是下单日。",
    "order_item": "订单明细（一个订单多行）。销售额=quantity*unit_price*(1-discount)。",
    "customer": "客户表。is_active=1 表示活跃客户；level 是会员等级；signup_date 是注册日。",
    "product": "商品表。list_price 是标价、cost 是成本、status='off_shelf' 表示已下架；launched_on 可能为空。",
    "category": "商品类目，两级结构：parent_id 为空的是大类，其余是小类。",
    "payment": "支付流水。status='success' 才算真正收到钱；paid_at 为支付时间。",
    "refund": "退款记录，只对 status='refunded' 的订单存在；reason 是退款原因。",
    "supplier": "供应商。rating 是 1~5 的评分。",
    "region": "销售大区，region_name 是中文大区名。",
}

COLUMN_NOTES: dict[str, str] = {
    "orders.status": "completed=已完成成交 / cancelled=已取消 / pending=待支付 / refunded=已退款",
    "orders.channel": "渠道：app / web / offline / mini_program",
    "orders.total_amount": "订单金额（未扣退款）",
    "customer.level": "会员等级：bronze / silver / gold / platinum",
    "customer.is_active": "1=活跃，0=已流失",
    "product.status": "on_sale=在售 / off_shelf=已下架",
    "payment.status": "success / failed / pending",
    "refund.reason": "quality 质量问题 / wrong_item 错发 / late_delivery 迟到 / no_reason 无理由",
    "order_item.discount": "折扣率，0.1 表示打 9 折",
}


@dataclass
class ColumnDoc:
    name: str
    type: str
    notnull: bool
    pk: bool
    default: Any = None
    fk_table: str | None = None
    fk_column: str | None = None
    null_ratio: float = 0.0
    distinct: int = 0
    samples: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def is_categorical(self) -> bool:
        """低基数文本列 = 适合做分组/过滤的维度。"""
        return self.type.upper().startswith(("TEXT", "VARCHAR")) and 0 < self.distinct <= 20

    def render(self) -> str:
        bits = [f"{self.name} {self.type}"]
        if self.pk:
            bits.append("PK")
        if self.fk_table:
            bits.append(f"FK->{self.fk_table}.{self.fk_column}")
        if self.null_ratio > 0.01:
            bits.append(f"空值{self.null_ratio:.0%}")
        if self.is_categorical and self.samples:
            bits.append("取值: " + "/".join(self.samples[:8]))
        if self.note:
            bits.append(self.note)
        return "  - " + " | ".join(bits)


@dataclass
class TableDoc:
    name: str
    row_count: int
    columns: list[ColumnDoc] = field(default_factory=list)
    note: str = ""
    fks: list[tuple[str, str, str]] = field(default_factory=list)  # (col, ref_table, ref_col)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def render(self, with_samples: bool = True) -> str:
        head = f"TABLE {self.name}  ({self.row_count} 行)"
        if self.note:
            head += f"\n  业务说明: {self.note}"
        lines = [head]
        for col in self.columns:
            lines.append(col.render() if with_samples else "  - " + col.name + " " + col.type)
        return "\n".join(lines)

    def search_text(self) -> str:
        """给 BM25 / 向量召回用的检索文本。"""
        parts = [self.name, self.note]
        for c in self.columns:
            parts.append(f"{c.name} {c.note} " + " ".join(c.samples[:8]))
        return " ".join(p for p in parts if p)


def _safe_rows(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.Error:
        return []


def introspect(db_path: Path, *, sample_limit: int = 8) -> list[TableDoc]:
    """读取表结构 + 每列画像。只读、不会修改库。"""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in _safe_rows(
            conn, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        docs: list[TableDoc] = []
        for tname in tables:
            total = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
            fk_map = {r["from"]: (r["table"], r["to"]) for r in conn.execute(f'PRAGMA foreign_key_list("{tname}")')}
            cols: list[ColumnDoc] = []
            for row in conn.execute(f'PRAGMA table_info("{tname}")'):
                cname, ctype = row["name"], (row["type"] or "TEXT")
                doc = ColumnDoc(
                    name=cname,
                    type=ctype,
                    notnull=bool(row["notnull"]),
                    pk=bool(row["pk"]),
                    default=row["dflt_value"],
                    fk_table=fk_map.get(cname, (None, None))[0],
                    fk_column=fk_map.get(cname, (None, None))[1],
                    note=COLUMN_NOTES.get(f"{tname}.{cname}", ""),
                )
                if total:
                    stats = conn.execute(
                        f'SELECT COUNT(DISTINCT "{cname}") AS d, '
                        f'SUM(CASE WHEN "{cname}" IS NULL THEN 1 ELSE 0 END) AS n FROM "{tname}"'
                    ).fetchone()
                    doc.distinct = int(stats["d"] or 0)
                    doc.null_ratio = float(stats["n"] or 0) / total
                    if doc.is_categorical or doc.distinct <= 25:
                        samples = conn.execute(
                            f'SELECT DISTINCT "{cname}" FROM "{tname}" WHERE "{cname}" IS NOT NULL LIMIT ?',
                            (sample_limit,),
                        ).fetchall()
                        doc.samples = [str(s[0]) for s in samples]
                cols.append(doc)
            docs.append(TableDoc(name=tname, row_count=total, columns=cols, note=TABLE_NOTES.get(tname, ""),
                                 fks=[(c, c2[0], c2[1]) for c, c2 in fk_map.items() if c2[0]]))
        return docs
    finally:
        conn.close()


def schema_prompt(docs: list[TableDoc], *, max_tables: int | None = None) -> str:
    """渲染成 Prompt 里的 schema 文本；只给给定表（Schema Linking 之后调用）。"""
    chosen = docs if max_tables is None else docs[:max_tables]
    return "\n\n".join(d.render() for d in chosen)


def all_tables_prompt(docs: list[TableDoc]) -> str:
    """全库概览（表名 + 行数 + 一句话），用于给模型一个全局视野但不塞满上下文。"""
    lines = ["可用表（共 %d 张）：" % len(docs)]
    for d in docs:
        lines.append(f"  - {d.name} ({d.row_count} 行): {d.note or ' / '.join(d.column_names[:6])}")
    return "\n".join(lines)
