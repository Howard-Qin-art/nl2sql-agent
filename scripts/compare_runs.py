# -*- coding: utf-8 -*-
"""对比两次评测结果（例如「本地 7B」vs「云端 qwen-max」），输出逐题差异与分组表。

用法：
    python scripts/compare_runs.py                       # 自动取最新的两个 60 题满跑结果
    python scripts/compare_runs.py --a <本地.jsonl> --b <云端.jsonl> --label-a 本地7B --label-b 百炼qwen-max
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def load(path: str) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def grade(rows: dict[str, dict]) -> tuple[int, int]:
    graded = [r for r in rows.values() if r.get("gradable")]
    return sum(1 for r in graded if r.get("correct")), len(graded)


def stats(rows: dict[str, dict]) -> dict:
    graded = [r for r in rows.values() if r.get("gradable")]
    lat = [r["latency_ms"] / 1000 for r in rows.values() if r.get("latency_ms")]
    prompt = sum(r.get("prompt_tokens", 0) for r in rows.values())
    completion = sum(r.get("completion_tokens", 0) for r in rows.values())
    return {
        "ex": (sum(1 for r in graded if r.get("correct")) / len(graded)) if graded else 0,
        "p50": statistics.median(lat) if lat else 0,
        "mean": statistics.fmean(lat) if lat else 0,
        "calls": sum(r.get("llm_calls", 0) for r in rows.values()),
        "prompt": prompt,
        "completion": completion,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None, help="第一个结果（基线，通常是本地）")
    parser.add_argument("--b", default=None, help="第二个结果（对照，通常是云端）")
    parser.add_argument("--label-a", default="基线")
    parser.add_argument("--label-b", default="对照")
    args = parser.parse_args()

    if args.a and args.b:
        path_a, path_b = args.a, args.b
    else:
        # 自动挑：本地 60 题满跑（full2-*）与云端 60 题满跑（cloud-qwen*）
        local = sorted(glob.glob(str(RESULTS / "full2-*.jsonl")))
        cloud = sorted(glob.glob(str(RESULTS / "cloud-qwen*.jsonl")))
        if not local or not cloud:
            print("找不到 full2-*.jsonl 或 cloud-qwen*.jsonl，请用 --a/--b 指定")
            return 1
        path_a, path_b = local[-1], cloud[-1]

    A, B = load(path_a), load(path_b)
    common = sorted(set(A) & set(B))
    print(f"A = {args.label_a:<14} {os.path.basename(path_a)}")
    print(f"B = {args.label_b:<14} {os.path.basename(path_b)}")
    print(f"共同题数：{len(common)}\n")

    sa, sb = stats({k: A[k] for k in common}), stats({k: B[k] for k in common})
    print(f"{'指标':<14}{args.label_a:>14}{args.label_b:>16}{'变化':>12}")
    print("-" * 60)
    exa, exb = sa["ex"] * 100, sb["ex"] * 100
    print(f"{'执行准确率 EX':<14}{exa:>13.1f}%{exb:>15.1f}%{exb - exa:>+11.1f}pt")
    print(f"{'延迟 p50':<14}{sa['p50']:>13.1f}s{sb['p50']:>15.1f}s{'':>12}")
    print(f"{'延迟 mean':<14}{sa['mean']:>13.1f}s{sb['mean']:>15.1f}s{'':>12}")
    print(f"{'LLM 调用':<14}{sa['calls']:>14}{sb['calls']:>16}{'':>12}")
    print(f"{'token 合计':<14}{sa['prompt'] + sa['completion']:>14}{sb['prompt'] + sb['completion']:>16}{'':>12}")

    graded = [k for k in common if A[k].get("gradable")]
    fixed = [k for k in graded if not A[k].get("correct") and B[k].get("correct")]
    broke = [k for k in graded if A[k].get("correct") and not B[k].get("correct")]
    print(f"\n由错转对 {len(fixed)} 题：{' '.join(fixed) or '（无）'}")
    print(f"由对转错 {len(broke)} 题：{' '.join(broke) or '（无）'}")

    print("\n仍答错的题（两版都错）：")
    still = [k for k in graded if not A[k].get("correct") and not B[k].get("correct")]
    for k in still:
        print(f"  [{k}] {A[k]['question'][:36]}")

    print("\n分组（按标签）：")
    tags = sorted({t for k in common for t in A[k].get("tags", [])})
    print(f"  {'标签':<12}{'题数':>6}{args.label_a:>12}{args.label_b:>14}")
    for tag in tags:
        ks = [k for k in graded if tag in A[k].get("tags", [])]
        if not ks:
            continue
        ea = sum(1 for k in ks if A[k].get("correct")) / len(ks) * 100
        eb = sum(1 for k in ks if B[k].get("correct")) / len(ks) * 100
        print(f"  {tag:<12}{len(ks):>6}{ea:>11.0f}%{eb:>13.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
