# -*- coding: utf-8 -*-
"""稳定性评测：同一个问题换 5 种说法，看答案是否稳定。

衡量的是"语义不变、表述不同"时的鲁棒性 —— 这是单次准确率看不出来的东西。
每组 5 个说法共享同一个标准 SQL（从 eval/questions.jsonl 按 base_id 取，避免手抄出错）。

输出：
    - 每题：5 种说法各自的 EX、答案指纹一致性
    - 总览：组内平均一致率、全一致组数、组均 EX、EX 极差

用法：
    python scripts/stability_eval.py                        # 按 .env 的后端
    python scripts/stability_eval.py --backend openai       # 强制云端
    python scripts/stability_eval.py --limit 2              # 只跑前 2 组（冒烟）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.executor import SQLExecutor  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.tracing import Tracer  # noqa: E402


def cell(v):
    if v is None:
        return "<NULL>"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return str(v).strip()


def fp_strict(rows) -> tuple:
    """单元格排序 + 行排序：列序列序都无关的最严指纹。"""
    return tuple(sorted(tuple(sorted(cell(c) for c in row)) for row in rows))


def fp_values(rows) -> tuple:
    """只比数值集合，忽略行列结构与列数差异。"""
    return tuple(sorted(cell(c) for row in rows for c in row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="cloud")
    args = ap.parse_args()

    settings = SETTINGS
    if args.backend:
        settings = dataclasses.replace(SETTINGS, llm_backend=args.backend)
    print(f"后端={settings.llm_backend} 模型={settings.openai_model if settings.llm_backend == 'openai' else settings.model}")

    questions = {}
    for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            questions[r["id"]] = r

    groups = []
    for line in (ROOT / "eval" / "stability.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            groups.append(json.loads(line))
    if args.limit:
        groups = groups[: args.limit]

    docs = introspect(settings.db_path)
    executor = SQLExecutor(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.max_rows)
    tracer = Tracer(enabled=False)
    agent = NL2SQLAgent(settings=settings, llm=LLMClient(settings=settings, tracer=tracer),
                        docs=docs, tracer=tracer, use_summary=False)

    # 预检：先发一个极小请求。否则额度/认证类错误会被 Agent 的 120 秒预算吞掉，
    # 变成"每题都超时、0 行"，白跑很久还看不出原因（真实踩过）。
    try:
        LLMClient(settings=settings).chat([{"role": "user", "content": "ok"}], max_tokens=5)
        print("预检: 模型后端可用")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "403" in msg or "quota" in low or "arrearage" in low:
            hint = ("\n  多半是免费额度用尽 / 账户欠费：去百炼控制台关闭「仅用免费额度」模式或充值，"
                    "或换一个有额度的 key。")
        elif "401" in msg:
            hint = "\n  key 无效，检查 .env 里的 NL2SQL_OPENAI_KEY。"
        elif "timeout" in low or "timed out" in low:
            hint = "\n  连不上模型服务，检查网络或代理。"
        else:
            hint = ""
        print(f"预检失败，已中止（没有开始跑题）：{msg[:200]}{hint}")
        return 2

    report_rows = []
    t_start = time.perf_counter()
    for g in groups:
        base = questions[g["base_id"]]
        gold = executor.run(base["gold_sql"])
        gold_rows = [tuple(r) for r in gold.rows]
        gold_fp, gold_vals = fp_strict(gold_rows), fp_values(gold_rows)

        variants = []
        for q in g["variants"]:
            t0 = time.perf_counter()
            try:
                out = agent.run(q)
                rows = [tuple(r) for r in (out.rows or [])]
                exc = rows and fp_strict(rows) == gold_fp
                vals_ok = rows and fp_values(rows) == gold_vals
                variants.append({
                    "question": q,
                    "ok": bool(exc),
                    "vals_ok": bool(vals_ok),
                    "sql": getattr(out, "sql", "") or "",
                    "rows": len(rows),
                    "row_count": getattr(out, "row_count", 0),
                    "fp": repr(fp_strict(rows)) if rows else "",
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "error": getattr(out, "error", None),
                    "clarify": bool(getattr(out, "clarify_question", None)),
                })
            except Exception as exc:  # 单题失败不打断整轮
                msg = f"{type(exc).__name__}: {exc}"
                # 认证/额度类错误重试也没用，直接停掉整轮，别白等
                if any(k in msg for k in ("401", "invalid_api_key", "Incorrect API key",
                                          "403", "Arrearage", "quota")):
                    print(f"\n[中止] 模型接口认证/额度有问题，已停止：{msg[:200]}")
                    print("       检查 .env 里的 NL2SQL_OPENAI_KEY 是否有效、账户是否有余额。")
                    sys.exit(2)
                variants.append({"question": q, "ok": False, "vals_ok": False, "sql": "",
                                 "rows": 0, "fp": "", "latency_ms": (time.perf_counter() - t0) * 1000,
                                 "error": msg, "clarify": False})
            v = variants[-1]
            print(f"  [{g['group']}/{g['base_id']}] {'OK ' if v['ok'] else ('~  ' if v['vals_ok'] else 'ERR')} "
                  f"{v['latency_ms']/1000:5.1f}s  {v['rows']:>3} 行  {v['question'][:34]}", flush=True)

        fps = [v["fp"] for v in variants if v["fp"]]
        counter = Counter(fps)
        mode_share = (counter.most_common(1)[0][1] / len(fps)) if fps else 0.0
        report_rows.append({
            "group": g["group"],
            "base_id": g["base_id"],
            "tags": base.get("tags"),
            "gold_question": base["question"],
            "variants": variants,
            "ex_count": sum(1 for v in variants if v["ok"]),
            "distinct_answers": len(counter),
            "mode_share": mode_share,
            "all_same": len(counter) == 1,
            "clarify_count": sum(1 for v in variants if v["clarify"]),
        })

    total_s = time.perf_counter() - t_start
    lines = [f"# 稳定性评测（{settings.llm_backend} / "
             f"{settings.openai_model if settings.llm_backend == 'openai' else settings.model}）", ""]
    lines.append(f"共 {len(report_rows)} 组 × 5 种说法 = {len(report_rows) * 5} 次查询，用时 {total_s / 60:.1f} 分钟。")
    lines.append("")
    lines.append("| 组 | 原题 | 5 种说法 EX | 不同答案数 | 最常见答案占比 | 全部一致 | 触发澄清 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report_rows:
        lines.append(f"| {r['group']} | {r['gold_question'][:26]} | {r['ex_count']}/5 | {r['distinct_answers']} | "
                     f"{r['mode_share']:.0%} | {'是' if r['all_same'] else '否'} | {r['clarify_count']} |")

    ex_all = sum(r["ex_count"] for r in report_rows) / (len(report_rows) * 5)
    avg_share = sum(r["mode_share"] for r in report_rows) / len(report_rows)
    all_same = sum(1 for r in report_rows if r["all_same"])
    ex_by_group = [r["ex_count"] / 5 for r in report_rows]
    spread = max(ex_by_group) - min(ex_by_group)

    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 逐说法平均 EX：**{ex_all:.1%}**")
    lines.append(f"- 组内平均一致率：**{avg_share:.1%}**（同一问题的 5 种说法里，答案完全相同的比例）")
    lines.append(f"- 5 种说法答案全一致的组：**{all_same}/{len(report_rows)}**")
    lines.append(f"- 组间 EX 极差：**{spread:.0%}**（最小 {min(ex_by_group):.0%} ~ 最大 {max(ex_by_group):.0%}）")
    clar = sum(r["clarify_count"] for r in report_rows)
    if clar:
        lines.append(f"- 触发澄清的说法：{clar} 条（同样的意图，换了说法就让模型反问，也算不稳定）")

    text = "\n".join(lines)
    print()
    print(text)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M")
    md = out_dir / f"stability-{args.tag}-{stamp}.md"
    md.write_text(text, encoding="utf-8")
    raw = out_dir / f"stability-{args.tag}-{stamp}.json"
    raw.write_text(json.dumps(report_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入: {md.name} / {raw.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
