"""只读 SQL 执行器。

要点（都是面试会追问的工程细节）：
    1. 用 `file:...?mode=ro` + `PRAGMA query_only=1` 双保险，从连接层面禁止写；
    2. `set_progress_handler` 做**可中断**的超时（SQLite 没有 statement_timeout，
       必须在 VM 回调里抛异常才能真正打断一条慢查询）；
    3. 行数上限：`fetchmany(max_rows + 1)` 判断是否被截断，避免把大结果集读进内存；
    4. 报错原文原样返回 —— 这是执行自愈（self-repair）的输入。
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    sql: str = ""

    def preview(self, n: int = 5) -> list[tuple]:
        return self.rows[:n]

    def as_dicts(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.rows if limit is None else self.rows[:limit]
        return [dict(zip(self.columns, r)) for r in rows]

    def result_signature(self) -> str:
        """结果集指纹：用于自一致性投票（与行序、列序、列别名无关）。"""
        normalized = []
        for row in self.rows:
            cells = []
            for cell in row:
                if cell is None:
                    cells.append("<NULL>")
                elif isinstance(cell, float):
                    cells.append(f"{round(cell, 4):.4f}")
                else:
                    cells.append(str(cell))
            normalized.append(tuple(sorted(cells)))
        normalized.sort()
        return repr(normalized)


class SQLExecutor:
    def __init__(self, db_path: Path, *, timeout_s: float = 10.0, max_rows: int = 500):
        self.db_path = Path(db_path)
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    # ---------- 连接 ----------
    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}（先跑 python -m app.db.seed）")
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, timeout=self.timeout_s)
        conn.execute("PRAGMA query_only=1")
        # 排序/去重产生的临时表放内存，不落到系统临时目录（Windows 下默认在 C 盘）。
        # 顺带也更快：大数据集上避免临时文件读写。
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _deadline_handler(self, deadline: float):
        def handler() -> int:
            return 1 if time.perf_counter() > deadline else 0
        return handler

    # ---------- 执行 ----------
    def run(self, sql: str) -> ExecResult:
        t0 = time.perf_counter()
        conn = None
        try:
            conn = self._connect()
            deadline = t0 + self.timeout_s
            conn.set_progress_handler(self._deadline_handler(deadline), 2000)
            cur = conn.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
            return ExecResult(True, columns, [tuple(r) for r in rows], len(rows), truncated,
                              latency_ms=(time.perf_counter() - t0) * 1000, sql=sql)
        except sqlite3.OperationalError as exc:
            message = str(exc)
            error_type = "timeout" if "interrupted" in message.lower() else "operational"
            return ExecResult(False, error=message, error_type=error_type,
                              latency_ms=(time.perf_counter() - t0) * 1000, sql=sql)
        except sqlite3.Error as exc:
            return ExecResult(False, error=f"{type(exc).__name__}: {exc}", error_type="sqlite",
                              latency_ms=(time.perf_counter() - t0) * 1000, sql=sql)
        except Exception as exc:  # noqa: BLE001 - 任何异常都要变成可回注的文本，不能炸掉 Agent
            return ExecResult(False, error=f"{type(exc).__name__}: {exc}", error_type="python",
                              latency_ms=(time.perf_counter() - t0) * 1000, sql=sql)
        finally:
            if conn is not None:
                conn.close()

    # ---------- 便捷 ----------
    def scalar(self, sql: str) -> Any:
        res = self.run(sql)
        if not res.ok or not res.rows:
            return None
        return res.rows[0][0]

    def explain_plan(self, sql: str) -> list[str]:
        """把执行计划也暴露出来：面试里「慢查询怎么定位」的素材。"""
        conn = self._connect()
        try:
            return [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql)]
        finally:
            conn.close()
