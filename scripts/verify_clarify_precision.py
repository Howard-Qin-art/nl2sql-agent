# -*- coding: utf-8 -*-
"""歧义澄清的「准召」验证：只调歧义判断节点，不生成 SQL，所以很快（每题 1 次模型调用）。

要回答两个问题：
    1. 该澄清的题（题库里 expect=clarify 的 6 题）是否都澄清了？        -> 召回率
    2. 本来能直接答的题有多少被误澄清？                                 -> 误澄清率（越低越好）

用法：
    python scripts/verify_clarify_precision.py               # 全部可答题 + 6 道歧义题
    python scripts/verify_clarify_precision.py --sample 20   # 可答题只抽样 20 道（省时间）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.tracing import Tracer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="可直答题抽样数量（0=全部）")
    parser.add_argument("--out", default="eval/results/clarify_precision.json")
    args = parser.parse_args()

    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    clarify_cases = [q for q in questions if q.get("expect") == "clarify"]
    answerable = [q for q in questions if q.get("expect") in ("rows", "empty")]
    if args.sample:
        answerable = answerable[: args.sample]

    agent = NL2SQLAgent(settings=SETTINGS, llm=LLMClient(settings=SETTINGS, tracer=Tracer(enabled=False)),
                        docs=introspect(SETTINGS.db_path), tracer=Tracer(enabled=False),
                        use_llm_rerank=False)  # 关掉重排：只需 schema 文本，省一次调用

    def judge(question: str) -> tuple[bool, str]:
        state = {"question": question, "repair_rounds": 0}
        state.update(agent.link_schema(state))
        out = agent.check_ambiguity(state)
        return bool(out.get("needs_clarification")), str(out.get("clarify_question", ""))

    print(f"该澄清 {len(clarify_cases)} 题 / 可直答 {len(answerable)} 题")
    results = {"clarify": [], "answerable": []}
    t0 = time.time()

    for q in clarify_cases:
        asked, text = judge(q["question"])
        results["clarify"].append({"id": q["id"], "question": q["question"], "asked": asked, "clarify": text})
        print(f"  [该澄清] {q['id']} {'✅ 澄清' if asked else '❌ 漏了'}  {q['question'][:28]}")
    for q in answerable:
        asked, text = judge(q["question"])
        results["answerable"].append({"id": q["id"], "question": q["question"], "asked": asked, "clarify": text})
        mark = "❌ 误澄清" if asked else "✅ 直答"
        print(f"  [可直答] {q['id']} {mark}  {q['question'][:28]}")

    n_clar = len(results["clarify"])
    hit = sum(1 for r in results["clarify"] if r["asked"])
    n_ans = len(results["answerable"])
    false_ask = sum(1 for r in results["answerable"] if r["asked"])
    summary = {
        "clarify_recall": hit / n_clar if n_clar else None,
        "false_clarify_rate": false_ask / n_ans if n_ans else None,
        "n_clarify_cases": n_clar,
        "n_answerable_cases": n_ans,
        "elapsed_s": round(time.time() - t0, 1),
        "model": SETTINGS.model,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "detail": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"该澄清召回率 : {hit}/{n_clar} = {100 * summary['clarify_recall']:.0f}%" if n_clar else "无歧义题")
    print(f"误澄清率     : {false_ask}/{n_ans} = {100 * summary['false_clarify_rate']:.1f}%  （越低越好）")
    print(f"耗时 {summary['elapsed_s']}s｜明细 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
