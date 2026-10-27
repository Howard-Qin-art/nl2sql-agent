# -*- coding: utf-8 -*-
"""在 BIRD mini_dev 上跑评测（公开数据集，不是自建题库）。

和 eval/run_eval.py 的区别：
    - 每个问题自带 db_id，库文件从 BIRD 目录取（11 个库，最大 583MB）
    - 题面是英文，所以 agent 用 lang="en"（不带中文电商业务口径）
    - BIRD 官方允许把 evidence 当提示，这里默认带上，并在结果里标明（walkthrough 可复现）
    - gold SQL 直接在该库上执行，和预测结果比（行序无关 + 数值容差）

用法：
    python scripts/eval_bird.py --limit 10                    # 先跑 10 题冒烟
    python scripts/eval_bird.py --limit 50 --difficulty simple
    python scripts/eval_bird.py --all --tag bird-full
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent          # noqa: E402
from app.config import SETTINGS                 # noqa: E402
from app.db.schema import TableDoc, schema_prompt  # noqa: E402
from app.llm import LLMClient                   # noqa: E402
from app.tracing import Tracer                  # noqa: E402
from eval.bird import BIRD_DIR, build_table_docs, db_path, difficulty_of, load_questions  # noqa: E402
from eval.metrics import percentile, results_equal, results_equal_loose  # noqa: E402

OUT_DIR = Path(r"E:\dddddd\nl2sql数据\bird_results")


def gold_rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="跑全部 500 题")
    ap.add_argument("--difficulty", default=None, choices=["simple", "moderate", "challenging", "unknown"])
    ap.add_argument("--tag", default="bird-smoke")
    ap.add_argument("--model", default=None, help="覆盖模型名（默认用 .env 里的）")
    ap.add_argument("--no-evidence", action="store_true", help="不给 evidence 提示（更严格）")
    ap.add_argument("--bird-dir", default=str(BIRD_DIR))
    args = ap.parse_args()

    bird_dir = Path(args.bird_dir)
    qs = load_questions(bird_dir, limit=None if args.all else args.limit, difficulty=args.difficulty)
    print(f"题目 {len(qs)} 道 | 库目录 {bird_dir}")

    settings = SETTINGS
    if args.model:
        settings = dataclasses.replace(settings, model=args.model)
    tracer = Tracer(enabled=False)
    llm = LLMClient(settings=settings, tracer=tracer)

    # 每个库的表画像只建一次（大库的 COUNT(*) 和抽样比较贵）
    docs_cache: dict[str, list[TableDoc]] = {}
    metas: dict[str, dict] = {}
    from eval.bird import load_schema_meta
    all_meta = load_schema_meta(bird_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M")
    out_path = OUT_DIR / f"{args.tag}-{stamp}.jsonl"

    records: list[dict] = []
    t_start = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fout:
        for i, q in enumerate(qs, 1):
            db_id = q["db_id"]
            if db_id not in docs_cache:
                t0 = time.perf_counter()
                docs_cache[db_id] = build_table_docs(db_id, bird_dir, meta=all_meta.get(db_id))
                print(f"  [建表画像] {db_id}: {len(docs_cache[db_id])} 表 / {time.perf_counter() - t0:.1f}s")
            docs = docs_cache[db_id]
            db_file = db_path(db_id, bird_dir)

            question = q["question"]
            if q.get("evidence") and not args.no_evidence:
                question = f"{question}\n\n(Hints: {q['evidence']})"

            st = dataclasses.replace(settings, db_path=db_file)
            agent = NL2SQLAgent(settings=st, llm=llm, docs=docs, tracer=tracer,
                                lang="en", use_clarify=False, use_summary=False)
            t0 = time.perf_counter()
            try:
                res = agent.run(question)
                pred_rows = res.rows or []
                pred_sql = res.sql or ""
                err = res.error
            except Exception as exc:  # noqa: BLE001
                pred_rows, pred_sql, err = [], "", f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - t0) * 1000

            try:
                g_rows = gold_rows(db_file, q["SQL"])
                g_err = None
            except Exception as exc:  # noqa: BLE001
                g_rows, g_err = [], str(exc)

            ex = results_equal(g_rows, pred_rows, tolerant=True) if g_err is None else False
            ex_relaxed = results_equal(g_rows, pred_rows, relaxed=True, tolerant=True) if g_err is None else False
            ex_loose = results_equal_loose(g_rows, pred_rows) if g_err is None else False

            rec = {
                "question_id": q.get("question_id"),
                "db_id": db_id,
                "difficulty": difficulty_of(q),
                "question": q["question"],
                "evidence": q.get("evidence", ""),
                "gold_sql": q["SQL"],
                "pred_sql": pred_sql,
                "gold_rows": len(g_rows),
                "pred_rows": len(pred_rows),
                "ex": bool(ex),
                "ex_relaxed": bool(ex_relaxed),
                "ex_loose": bool(ex_loose),
                "latency_ms": round(latency_ms, 1),
                "gold_error": g_err,
                "error": err,
                "answer": (res.answer or "")[:200] if "res" in dir() else "",
            }
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            mark = "✅" if ex else ("~" if ex_loose else "❌")
            print(f"[{i}/{len(qs)}] {mark} {db_id:<22} {q['difficulty']:<11} {latency_ms/1000:5.1f}s  {q['question'][:56]}")

    total = len(records)
    ok = sum(r["ex"] for r in records)
    ok_loose = sum(r["ex_loose"] for r in records)
    by_diff: dict[str, list[bool]] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r["ex"])
    summary = {
        "dataset": "BIRD mini_dev (sqlite)",
        "tag": args.tag,
        "n": total,
        "ex_strict": round(ok / total * 100, 1) if total else 0,
        "ex_loose": round(ok_loose / total * 100, 1) if total else 0,
        "by_difficulty": {k: {"n": len(v), "ex": round(sum(v) / len(v) * 100, 1)} for k, v in by_diff.items()},
        "latency_p50_s": round(percentile([r["latency_ms"] for r in records], 50) / 1000, 1),
        "latency_p95_s": round(percentile([r["latency_ms"] for r in records], 95) / 1000, 1),
        "elapsed_min": round((time.perf_counter() - t_start) / 60, 1),
        "with_evidence": not args.no_evidence,
        "model": settings.model,
    }
    (OUT_DIR / f"{args.tag}-{stamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"BIRD mini_dev：{total} 题  EX {summary['ex_strict']}%  宽松 {summary['ex_loose']}%")
    print(f"分难度：{summary['by_difficulty']}")
    print(f"延迟 p50/p95 = {summary['latency_p50_s']}/{summary['latency_p95_s']}s | 用时 {summary['elapsed_min']} 分钟")
    print(f"结果：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
