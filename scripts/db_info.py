# -*- coding: utf-8 -*-
"""打印示例库概况：文件位置、大小、每张表的行数、SQLite 版本。

用法：
    python scripts/db_info.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402


def main() -> int:
    db = Path(SETTINGS.db_path)
    print(f"数据库类型 : SQLite（单文件，无需服务进程）")
    print(f"文件位置   : {db}")
    if not db.exists():
        print("文件不存在，先跑：python -m app.db.seed")
        return 1
    print(f"文件大小   : {db.stat().st_size / 1024:.0f} KB")
    print(f"SQLite 版本: {sqlite3.sqlite_version}")

    # 只读打开（和 Agent 执行器一样的方式）
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        total = 0
        print(f"\n共 {len(tables)} 张表：")
        for name in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            total += count
            cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{name}")')]
            print(f"  {name:<12} {count:>6} 行   列: {', '.join(cols[:6])}"
                  + (" …" if len(cols) > 6 else ""))
        print(f"合计 {total} 行")
        index_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index'").fetchone()[0]
        print(f"\n索引数: {index_count}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
