# -*- coding: utf-8 -*-
"""软指标（quality metrics）计算：把"硬准确率"之外、但能自动量化的质量维度算出来。

包含四组：
    1. 置信度校准：投票一致度（agreement）分桶后的真实准确率。
       回答的是"它自己知不知道自己可能错" —— 如果一致度 1.0 的题错误率远低于 0.67 的，
       说明这个置信度是有区分度的，可以拿来做"不确定就提示用户"。
    2. 回答可读性：结论是否非空、长度是否适中、是否带关键数字（规则判定，不引模型）。
    3. 失败可定位性：失败样本里有多大比例同时留下了可定位的信息（SQL / 错误 / 候选明细）。
    4. 体验相关：澄清打扰率、自愈救回率、首行耗时占比。

用法：
    python scripts/soft_metrics.py --file eval/results/full2-*.jsonl
    python scripts/soft_metrics.py --file eval/results/cloud-qwen-max-*.jsonl --tag cloud-soft
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d")

BUCKETS = [
    ("一致度 1.00（三条候选全一致）", lambda a: a >= 0.999),
    ("一致度 0.67（两条一致）", lambda a: 0.66 <= a < 0.999),
    ("一致度 < 0.67（三条各不相同）", lambda a: a < 0.66),
]


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        found = sorted(glob.glob(str(ROOT / path)))
        if not found:
            raise SystemExit(f"找不到评测文件：{path}")
        p = Path(found[-1])
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"读取: {p.name}  共 {len(recs)} 题")
    return recs, p


def auc(records: list[dict]) -> float | None:
    """用 agreement 作为分数、correct 作为标签算 AUC（纯 Python，处理并列）。"""
    pos = [r["agreement"] for r in records if r.get("correct")]
    neg = [r["agreement"] for r in records if not r.get("correct")]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif abs(a - b) < 1e-9:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def calibration(records: list[dict]) -> list[tuple[str, int, float, float, float]]:
    out = []
    for label, pred in BUCKETS:
        group = [r for r in records if pred(float(r.get("agreement") or 0))]
        if not group:
            continue
        acc = sum(1 for r in group if r.get("correct")) / len(group)
        lat = sum(float(r.get("latency_ms") or 0) for r in group) / len(group) / 1000
        rows = sum(int(r.get("row_count") or 0) for r in group) / len(group)
        out.append((label, len(group), acc, lat, rows))
    return out


def readability(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("answer")]
    total = len(records)
    stats = {
        "有结论": len(ok),
        "结论非空率": len(ok) / total if total else 0,
        "带数字": sum(1 for r in ok if NUM.search(str(r["answer"]))),
        "长度适中(10~120字)": sum(1 for r in ok if 10 <= len(str(r["answer"])) <= 120),
        "过短(<10字)": sum(1 for r in ok if len(str(r["answer"])) < 10),
        "过长(>120字)": sum(1 for r in ok if len(str(r["answer"])) > 120),
    }
    return stats


def locate(records: list[dict]) -> dict:
    failed = [r for r in records if not r.get("correct")]
    has_sql = sum(1 for r in failed if r.get("pred_sql"))
    has_error = sum(1 for r in failed if r.get("error"))
    return {
        "失败题数": len(failed),
        "带生成SQL": has_sql,
        "带错误信息": has_error,
        "可定位率": (sum(1 for r in failed if r.get("pred_sql") or r.get("error")) / len(failed)) if failed else 1.0,
    }


def experience(records: list[dict]) -> dict:
    clarified = [r for r in records if r.get("clarified")]
    repair = [r for r in records if r.get("needed_repair")]
    repaired_ok = sum(1 for r in repair if r.get("correct"))
    fast = [float(r.get("latency_ms") or 0) / 1000 for r in records]
    fast.sort()
    p50 = fast[len(fast) // 2] if fast else 0
    return {
        "澄清题数": len(clarified),
        "澄清占比": len(clarified) / len(records) if records else 0,
        "触发自愈题数": len(repair),
        "自愈后答对": repaired_ok,
        "自愈救回率": (repaired_ok / len(repair)) if repair else 0.0,
        "延迟 p50": p50,
        "最慢单题": max(fast) if fast else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tag", default="soft")
    args = ap.parse_args()

    records, path = load(args.file)
    graded = [r for r in records if r.get("gradable", True)]

    cal = calibration(records)
    read = readability(records)
    loc = locate(records)
    exp = experience(records)
    a = auc(records)

    lines: list[str] = []
    lines.append(f"# 软指标报告 · {path.name}\n")
    lines.append(f"样本：{len(records)} 题（其中可评分 {len(graded)} 题）\n")

    lines.append("## 1. 置信度校准（投票一致度 vs 真实正确率）\n")
    lines.append("| 一致度档位 | 题数 | 真实准确率 | 平均耗时 | 平均返回行数 |")
    lines.append("|---|---|---|---|---|")
    for label, n, acc, lat, rows in cal:
        lines.append(f"| {label} | {n} | **{acc:.1%}** | {lat:.1f}s | {rows:.1f} |")
    if a is not None:
        lines.append(f"\n置信度区分度 AUC = **{a:.3f}**（0.5 = 没有区分能力，1.0 = 完全能区分）。")
        over = next((acc for lbl, _, acc, _, _ in cal if lbl.startswith("一致度 1.00")), None)
        if over is not None:
            lines.append(f"高置信组（一致度 1.00）里仍然答错的比例 = **{1 - over:.1%}**，"
                         f"这部分就是「自信但错」的风险区。")
    lines.append("")

    lines.append("## 2. 回答可读性（规则判定，不引入模型）\n")
    lines.append("| 检查项 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    for k in ["有结论", "带数字", "长度适中(10~120字)", "过短(<10字)", "过长(>120字)"]:
        v = read[k]
        base = len(records) if k == "有结论" else max(read["有结论"], 1)
        lines.append(f"| {k} | {v} | {v / base:.1%} |")
    lines.append("")

    lines.append("## 3. 失败可定位性\n")
    lines.append(f"- 失败题数：**{loc['失败题数']}**")
    lines.append(f"- 其中留下了生成 SQL：{loc['带生成SQL']}　留下了错误信息：{loc['带错误信息']}")
    lines.append(f"- 可定位率：**{loc['可定位率']:.1%}**（失败样本能追溯到具体原因的比例）")
    lines.append("")

    lines.append("## 4. 体验相关\n")
    lines.append(f"- 触发澄清：{exp['澄清题数']} 题（{exp['澄清占比']:.1%}）")
    lines.append(f"- 触发自愈：{exp['触发自愈题数']} 题，其中答对 {exp['自愈后答对']} 题"
                 f"（救回率 {exp['自愈救回率']:.1%}）")
    lines.append(f"- 延迟：p50 {exp['延迟 p50']:.1f}s，最慢单题 {exp['最慢单题']:.1f}s")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"soft-{args.tag}-{path.stem}.md"
    out.write_text(text, encoding="utf-8")
    print(f"已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
