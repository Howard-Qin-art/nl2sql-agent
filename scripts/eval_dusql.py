# -*- coding: utf-8 -*-
"""在 DuSQL（中文 Text-to-SQL）上评测：中文提问 → SQL，库是本地从官方 JSON 建出来的。

数据来源：
    E:\\dddddd\\workbuddy中文数据集\\DuSQL\\{db_schema,db_content,dev}.json
    -> scripts/build_dusql_db.py 建库到 E:\\dddddd\\nl2sql数据\\dusql_db
    -> scripts/fix_dusql_gold.py 规范化 gold SQL，筛出可执行子集 dev_executable.json（2267/2482）

用法：
    python scripts/eval_dusql.py --limit 20
    python scripts/eval_dusql.py --per-db 1 --tag dusql-strat
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent          # noqa: E402
from app.config import SETTINGS                 # noqa: E402
from app.db.schema import TableDoc, introspect  # noqa: E402
from app.llm import LLMClient                   # noqa: E402
from app.tracing import Tracer                  # noqa: E402
from eval.metrics import percentile, results_equal, results_equal_loose  # noqa: E402

DATA = Path(r"E:\dddddd\nl2sql数据")
DB_DIR = DATA / "dusql_db"
DEV = DATA / "dusql_eval" / "dev_executable.json"
OUT_DIR = DATA / "dusql_results"


def gold_rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--per-db", type=int, default=0, help="每个库取 N 题（分层抽样）")
    ap.add_argument("--shuffle", action="store_true", help="先随机打乱再取前 N 题（推荐，duSQL 的题按模板排序）")
    ap.add_argument("--seed", type=int, default=42, help="打乱用的随机种子，保证可复现")
    ap.add_argument("--tag", default="dusql")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    items = json.loads(DEV.read_text(encoding="utf-8"))
    if args.per_db:
        seen: dict[str, int] = {}
        picked = []
        for it in items:
            n = seen.get(it["db_id"], 0)
            if n < args.per_db:
                picked.append(it)
                seen[it["db_id"]] = n + 1
        items = picked
    # DuSQL 的 dev.json 是按查询模板排序的（前 20 题几乎全是「没有…」这种反连接），
    # 直接取前 N 题会得到虚高的准确率，所以默认先打乱。
    if args.shuffle or args.per_db:
        rng = random.Random(args.seed)
        rng.shuffle(items)
    items = items[: args.limit] if not args.per_db else items
    print(f"DuSQL 可执行题 {len(items)} 道 | {len({i['db_id'] for i in items})} 个库"
          + (f" | 已打乱 seed={args.seed}" if (args.shuffle or args.per_db) else " | 未打乱（按文件顺序）"))

    settings = SETTINGS
    if args.model:
        settings = dataclasses.replace(settings, model=args.model)
    settings = dataclasses.replace(settings, default_limit=10_000_000)   # 评测不截断结果

    tracer = Tracer(enabled=False)
    llm = LLMClient(settings=settings, tracer=tracer)
    docs_cache: dict[str, list[TableDoc]] = {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M")
    out_path = OUT_DIR / f"{args.tag}-{stamp}.jsonl"

    records: list[dict] = []
    t_start = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fout:
        for i, it in enumerate(items, 1):
            db_id = it["db_id"]
            db_file = DB_DIR / it["db_file"]
            if db_id not in docs_cache:
                docs_cache[db_id] = introspect(db_file)
            st = dataclasses.replace(settings, db_path=db_file)
            agent = NL2SQLAgent(settings=st, llm=llm, docs=docs_cache[db_id], tracer=tracer,
                                lang="zh-general", use_clarify=False, use_summary=False)
            t0 = time.perf_counter()
            err = None
            try:
                res = agent.run(it["question"])
                pred_sql, pred_rows = res.sql or "", res.rows or []
            except Exception as exc:  # noqa: BLE001
                pred_sql, pred_rows, err = "", [], f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - t0) * 1000
            g_rows = gold_rows(db_file, it["sql_gold"])
            ex = results_equal(g_rows, pred_rows, tolerant=True)
            rec = {
                "question_id": it.get("question_id"),
                "db_id": db_id,
                "question": it["question"],
                "gold_sql": it["sql_gold"],
                "pred_sql": pred_sql,
                "gold_rows": len(g_rows),
                "pred_rows": len(pred_rows),
                "ex": bool(ex),
                "ex_loose": bool(results_equal_loose(g_rows, pred_rows)),
                "latency_ms": round(latency_ms, 1),
                "error": err,
            }
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(items)}] {'✅' if ex else '❌'} {db_id:<16} {latency_ms / 1000:5.1f}s "
                  f"{it['question'][:44]}")

    n = len(records)
    ok = sum(r["ex"] for r in records)
    summary = {
        "dataset": "DuSQL dev (gold 可执行子集)",
        "n": n,
        "ex": round(ok / n * 100, 1) if n else 0,
        "ex_loose": round(sum(r["ex_loose"] for r in records) / n * 100, 1) if n else 0,
        "latency_p50_s": round(percentile([r["latency_ms"] for r in records], 0.5) / 1000, 1),
        "latency_p95_s": round(percentile([r["latency_ms"] for r in records], 0.95) / 1000, 1),
        "elapsed_min": round((time.perf_counter() - t_start) / 60, 1),
        "model": settings.model,
    }
    (OUT_DIR / f"{args.tag}-{stamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"DuSQL：{n} 题  EX {summary['ex']}%  宽松 {summary['ex_loose']}%  "
          f"p50 {summary['latency_p50_s']}s  用时 {summary['elapsed_min']} 分钟")
    print(f"结果：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
