# -*- coding: utf-8 -*-
"""把 BIRD mini_dev 的英文题翻成中文，生成「中文提问 → SQL」的评测集。

为什么这么做：
    - BIRD 的库和 gold SQL 是真实的、公开的、可执行的；
    - 但题面是英文，而本项目要验证的是「中文提问」；
    - 所以只翻译题面（question + evidence），gold SQL 和库完全不动 —— 执行准确率依然是标准口径。

产出：E:\\dddddd\\nl2sql数据\\bird_zh\\zh_bird_<n>.json
    [{"question_id", "db_id", "difficulty", "question_en", "question_zh",
      "evidence_en", "evidence_zh", "gold_sql"}, ...]

用法：python scripts/make_zh_bird.py --per-db 4 --model qwen2.5:7b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS          # noqa: E402
from app.llm import LLMClient            # noqa: E402
from eval.bird import BIRD_DIR, load_questions  # noqa: E402

OUT_DIR = Path(r"E:\dddddd\nl2sql数据\bird_zh")

SYS = ("你是数据库领域的中英翻译。把用户给的英文问题翻成自然、口语化的中文提问，"
       "保留业务术语的准确含义，不要解释、不要补全信息、不要输出任何多余文字。")

PROMPT = """【英文问题】{q}
{hints}
只输出中文翻译，不要引号、不要解释。"""


def translate(llm: LLMClient, text: str, hints: str = "") -> str:
    if not text:
        return ""
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": PROMPT.format(q=text, hints=f"【业务提示】{hints}\n" if hints else "")},
    ]
    res = llm.chat(msgs, temperature=0.0, max_tokens=200)
    return (res.text or "").strip().strip('"').strip()


def sample_stratified(items: list[dict], per_db: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for q in items:
        buckets.setdefault(q["db_id"], []).append(q)
    out: list[dict] = []
    for db_id in sorted(buckets):
        out.extend(buckets[db_id][:per_db])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-db", type=int, default=4)
    ap.add_argument("--total", type=int, default=0, help="覆盖：总共取多少题")
    ap.add_argument("--model", default=None)
    ap.add_argument("--bird-dir", default=str(BIRD_DIR))
    args = ap.parse_args()

    qs = load_questions(Path(args.bird_dir))
    picked = sample_stratified(qs, args.per_db)
    if args.total:
        picked = picked[: args.total]

    settings = SETTINGS
    if args.model or True:
        # 翻译用本地 Ollama（云端 key 当前不可用）
        import dataclasses
        settings = dataclasses.replace(
            settings, llm_backend="ollama",
            model=args.model or getattr(settings, "model", "qwen2.5:7b"),
        )
    llm = LLMClient(settings=settings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"zh_bird_{len(picked)}.json"
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, q in enumerate(picked, 1):
        zh = translate(llm, q["question"])
        ev_zh = translate(llm, q.get("evidence") or "")
        rows.append({
            "question_id": q.get("question_id"),
            "db_id": q["db_id"],
            "difficulty": str(q.get("difficulty", "")).lower(),
            "question_en": q["question"],
            "question_zh": zh,
            "evidence_en": q.get("evidence", ""),
            "evidence_zh": ev_zh,
            "gold_sql": q["SQL"],
        })
        print(f"[{i}/{len(picked)}] {q['db_id']:<22} {zh[:48]}")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 生成 {len(rows)} 条 -> {out_path}  用时 {(time.perf_counter() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
