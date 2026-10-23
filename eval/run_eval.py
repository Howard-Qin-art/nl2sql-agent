# -*- coding: utf-8 -*-
"""评测执行器：跑题库、判分、输出报告（结果逐题落盘，随时可中断）。

用法：
    python -m eval.run_eval --limit 10                 # 先小样本冒烟
    python -m eval.run_eval                            # 全量 60 题
    python -m eval.run_eval --engine graph             # 用 LangGraph 引擎跑同一套题
    python -m eval.run_eval --candidates 1             # 消融：关掉自一致性投票
    python -m eval.run_eval --model qwen2.5:3b         # 换小模型对比

产物：
    eval/results/<tag>.jsonl    逐题结果（可中断、可续跑分析）
    eval/results/<tag>_summary.json
    eval/results/<tag>_report.md   （可直接贴进 README 的指标表）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.agent.graph import GraphAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.executor import SQLExecutor  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database, table_counts  # noqa: E402
from app.guardrails import validate_sql  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.tracing import Tracer  # noqa: E402
from eval.manifest import build_manifest, describe as manifest_describe  # noqa: E402
from eval.metrics import markdown_report, results_equal, sql_normalized, summarize  # noqa: E402

CLARIFY_ANSWER = "按全部时间范围、以成交额和订单数作为指标统计"


def load_questions(path: Path, *, tags: list[str] | None, limit: int | None,
                   ids: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        wanted = {i.strip() for i in ids}
        rows = [r for r in rows if r["id"] in wanted]
    if tags:
        rows = [r for r in rows if set(r.get("tags", [])) & set(tags)]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description="NL2SQL Agent 评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="只跑指定题号（如 q006 q012），用于针对失败题集做定向回归")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--engine", choices=["core", "graph"], default="core")
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true", help="关闭 LLM 重排（消融）")
    parser.add_argument("--with-summary", action="store_true", help="额外调用模型生成中文结论（更慢）")
    parser.add_argument("--max-repair", type=int, default=None)
    parser.add_argument("--tag", default="run", help="结果文件前缀")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)

    overrides = {
        "llm_backend": SETTINGS.llm_backend,
        "n_candidates": args.candidates or SETTINGS.n_candidates,
        "max_repair_rounds": args.max_repair if args.max_repair is not None else SETTINGS.max_repair_rounds,
        # 只做内存态 trace（统计 token/调用次数），不落 JSONL
        "trace_enabled": True,
    }
    if args.model:
        overrides["model"] = args.model
    settings = dataclasses.replace(SETTINGS, **overrides)

    docs = introspect(settings.db_path)
    allowed = {d.name for d in docs}
    executor = SQLExecutor(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.max_rows)

    questions = load_questions(ROOT / "eval" / "questions.jsonl", tags=args.tags, limit=args.limit,
                               ids=args.ids)
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M")
    # 生效的模型名：云端后端用 openai_model，别把本地的 qwen2.5:7b 写进报告
    effective_model = settings.openai_model if settings.llm_backend == "openai" else settings.model
    tag = f"{args.tag}-{effective_model.replace(':', '_')}-n{settings.n_candidates}-{stamp}"
    jsonl_path = out_dir / f"{tag}.jsonl"

    print(f"评测开始：{len(questions)} 题 | 后端 {settings.llm_backend} | 模型 {effective_model} | "
          f"候选 {settings.n_candidates} | "
          f"引擎 {args.engine} | 重排 {'关' if args.no_rerank else '开'} | 结果 -> {jsonl_path.name}")
    if settings.llm_backend == "ollama":
        print("提示：本机 GPU 与 Ollama CUDA 内核不兼容，走 CPU 推理，单题约 1~3 分钟。")

    def build_agent():
        # 关键：LLM 与 Agent 必须共用同一个 tracer，否则 token/调用次数记不到 span 上
        tracer = Tracer(enabled=True)
        llm = LLMClient(settings=settings, tracer=tracer)
        return NL2SQLAgent(settings=settings, llm=llm, docs=docs, tracer=tracer,
                           use_llm_rerank=not args.no_rerank, use_summary=args.with_summary, use_clarify=True)

    agent = build_agent()
    graph_agent = GraphAgent(agent=agent) if args.engine == "graph" else None

    counts_before = table_counts(settings.db_path)
    records: list[dict] = []
    t_start = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(questions, start=1):
            qid, question = item["id"], item["question"]
            expect = item.get("expect", "rows")
            gold_sql = item.get("gold_sql", "")
            gold_rows: list[tuple] = []
            if gold_sql:
                gguard = validate_sql(gold_sql, default_limit=settings.default_limit, allowed_tables=allowed)
                gold_exec = executor.run(gguard.sql if gguard.ok else gold_sql)
                gold_rows = gold_exec.rows if gold_exec.ok else []

            asked: list[str] = []

            def clarify_handler(q: str) -> str:  # 模拟被打断后用户的回答
                asked.append(q)
                return CLARIFY_ANSWER

            # 关键：逐题归零 trace，否则 token / LLM 调用次数会跨题累积
            agent.tracer.reset()
            t0 = time.time()
            if graph_agent is not None:
                out = graph_agent.run(question)
                if out.get("interrupted"):
                    asked.append(str(out["clarify"].get("question", "")))
                    out = graph_agent.resume(CLARIFY_ANSWER, thread_id=out["thread_id"])
                res = dict(out)
                res.setdefault("rejected", [])
                res.setdefault("candidates", {})
                res.setdefault("trace", {})
                res.setdefault("state", {})
                repaired = any(h.get("ok") for h in (res["state"].get("repair_history") or []))
            else:
                agent_res = agent.run(question, clarify_handler=clarify_handler)
                res = agent_res.to_dict()
                repaired = agent_res.repaired

            latency_ms = (time.time() - t0) * 1000
            pred_sql = res.get("sql") or ""
            pred_rows = [tuple(r) for r in (res.get("rows") or [])]
            valid = bool(pred_sql) and validate_sql(pred_sql, default_limit=settings.default_limit,
                                                    allowed_tables=allowed).ok
            blocked_any = bool(res.get("rejected"))

            record: dict = {
                "id": qid,
                "question": question,
                "tags": item.get("tags", []),
                "expect": expect,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "linked_tables": res.get("linked_tables", []),
                "row_count": res.get("row_count", 0),
                "gold_row_count": len(gold_rows),
                "answer": (res.get("answer") or "")[:300],
                "error": res.get("error"),
                "repair_rounds": res.get("repair_rounds", 0),
                "needed_repair": bool(res.get("repair_rounds", 0) or repaired),
                "agreement": (res.get("candidates") or {}).get("agreement"),
                "blocked_any": blocked_any,
                "valid_sql": valid,
                "latency_ms": latency_ms,
                "llm_calls": (res.get("trace") or {}).get("llm_calls", 0),
                "prompt_tokens": (res.get("trace") or {}).get("prompt_tokens", 0),
                "completion_tokens": (res.get("trace") or {}).get("completion_tokens", 0),
                "clarified": bool(asked),
                "clarify_question": asked[0] if asked else "",
            }

            if expect in ("rows", "empty"):
                record["gradable"] = True
                record["correct"] = bool(pred_sql) and results_equal(gold_rows, pred_rows)
                record["correct_strict"] = bool(pred_sql) and results_equal(gold_rows, pred_rows, tolerant=False)
                record["correct_relaxed"] = bool(pred_sql) and results_equal(gold_rows, pred_rows, relaxed=True)
                record["em"] = bool(pred_sql) and sql_normalized(gold_sql) == sql_normalized(pred_sql)
            elif expect == "clarify":
                record["gradable"] = False
                record["clarify_ok"] = bool(asked) and bool(record["clarify_question"])
            elif expect == "safe":
                record["gradable"] = False
                record["safe_ok"] = bool(blocked_any or (valid and pred_sql.upper().lstrip().startswith(("SELECT", "WITH"))))

            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()

            if not args.quiet:
                done = idx
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(questions) - done)
                mark = ""
                if record.get("gradable"):
                    mark = "EX=1" if record["correct"] else "EX=0"
                elif expect == "clarify":
                    mark = "澄清=1" if record.get("clarify_ok") else "澄清=0"
                elif expect == "safe":
                    mark = "拦截=1" if record.get("safe_ok") else "拦截=0"
                print(f"[{done:>2}/{len(questions)}] {qid} {mark:<8} {latency_ms/1000:5.1f}s  "
                      f"eta {eta/60:5.1f}min  {question[:34]}")

    counts_after = table_counts(settings.db_path)
    integrity = counts_before == counts_after

    summary = summarize(records)
    summary["model"] = effective_model
    summary["candidates"] = settings.n_candidates
    summary["engine"] = args.engine
    summary["llm_rerank"] = not args.no_rerank
    summary["data_integrity_ok"] = integrity
    summary["wall_clock_s"] = round(time.time() - t_start, 1)
    # 现场快照：代码版本 + 关键配置 + 关键文件指纹，保证「这个数字对应哪一版」可查
    summary["manifest"] = build_manifest(settings, ROOT / "eval" / "questions.jsonl")

    summary_path = out_dir / f"{tag}_summary.json"
    report_path = out_dir / f"{tag}_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_body = markdown_report(summary, model=effective_model,
                                  candidates=settings.n_candidates, engine=args.engine)
    # 报告末尾附一行可复现信息
    report_body = report_body.rstrip() + "\n\n> " + manifest_describe(summary["manifest"]) + "\n"
    report_path.write_text(report_body, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"完成：{summary['n_questions']} 题，用时 {summary['wall_clock_s']/60:.1f} 分钟")
    print(f"执行准确率 EX = {summary['ex']:.1%}（列序无关 {summary['ex_relaxed']:.1%}）| EM = {summary['em']:.1%}")
    print(f"合法率 {summary['valid_sql_rate']:.1%} | 澄清率 {summary['clarify_accuracy']:.1%} | "
          f"拦截率 {summary['guardrail_block_rate']:.1%} | 自愈率 {summary['repair_rate']:.1%}")
    print(f"延迟 mean/p50/p95 = {summary['latency_ms']['mean']:.0f}/"
          f"{summary['latency_ms']['p50']:.0f}/{summary['latency_ms']['p95']:.0f} ms | "
          f"LLM 调用 {summary['llm_calls']} 次 | token {summary['tokens']['prompt']+summary['tokens']['completion']}")
    print(f"数据完整性（评测前后所有表行数不变）: {'OK' if integrity else '被改动!'}")
    print(f"结果：{jsonl_path}\n汇总：{summary_path}\n报告：{report_path}")
    return 0 if integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
