# -*- coding: utf-8 -*-
"""云端后端（阿里云百炼 / OpenAI / DeepSeek 等 OpenAI 兼容服务）一键自检。

为什么单独写这个脚本：切云端的失败点集中在四处 —— base_url 写错、key 与地域不匹配、
模型名没开通、以及「服务不支持 response_format / seed 这类参数」。
逐项试一遍比直接跑评测（几十次调用）快得多，也不会白白烧 token。

用法：
    python scripts/check_cloud_backend.py                      # 只做连通性 + JSON 能力自检
    python scripts/check_cloud_backend.py --ask "退款原因各占多少比例？"   # 顺便跑一次完整 Agent
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.llm import LLMClient, LLMError  # noqa: E402


def mask(key: str) -> str:
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "(未设置)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", default=None, help="自检通过后，顺便走一次完整 Agent 提问")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("当前配置")
    print(f"  backend   : {SETTINGS.llm_backend}")
    print(f"  base_url  : {SETTINGS.openai_base}")
    print(f"  model     : {SETTINGS.openai_model}")
    print(f"  api_key   : {mask(SETTINGS.openai_key)}")
    print(f"  超时/重试 : {SETTINGS.request_timeout:.0f}s / {SETTINGS.max_retries} 次｜"
          f"temperature 下限 {SETTINGS.min_temperature}")
    print("=" * 70)

    if SETTINGS.llm_backend != "openai":
        print("⚠️  当前 NL2SQL_LLM_BACKEND 不是 openai —— 这个脚本是给云端后端用的。")
        print("   请在 .env 里设 NL2SQL_LLM_BACKEND=openai 后再跑本脚本。")
        return 2
    if not SETTINGS.openai_key:
        print("❌ 没有读到 API Key。请在 .env 里填 NL2SQL_OPENAI_KEY=sk-xxx 后重试。")
        return 2

    try:
        llm = LLMClient(settings=SETTINGS)
    except LLMError as exc:
        print(f"❌ 初始化失败：{exc}")
        return 1

    # ---------- 1) 最简连通性：一次极小的对话 ----------
    print("\n[1/4] 连通性（一次极小的对话）")
    try:
        t0 = time.perf_counter()
        res = llm.chat([{"role": "user", "content": "只回答两个字：可以"}], temperature=0.0, max_tokens=args.max_tokens)
        dt = time.perf_counter() - t0
        print(f"  ✅ 成功  用时 {dt:.2f}s")
        print(f"     返回：{res.text.strip()[:60]!r}")
        print(f"     用量：prompt {res.prompt_tokens} / completion {res.completion_tokens} token（模型 {res.model}）")
        single_latency = dt
    except LLMError as exc:
        print(f"  ❌ 失败：{exc}")
        print("\n排查顺序：① base_url 是否以 /compatible-mode/v1 结尾；"
              "② key 与 base_url 是否同地域；③ 模型名是否已在百炼控制台开通。")
        return 1

    # ---------- 2) JSON 结构化输出（response_format 兼容性） ----------
    print("\n[2/4] JSON 结构化输出（关键：百炼官方参数清单里没有 response_format）")
    backend = llm.backend
    try:
        data, res2 = llm.chat_json(
            [{"role": "user", "content": '把这句话转成 JSON：{"sql": "SELECT 1", "tables": ["t"]}'}],
            temperature=0.0, max_tokens=200,
        )
        print(f"  ✅ 成功，解析出：{data}")
        supported = getattr(backend, "_json_mode_supported", None)
        if supported is False:
            print(f"  ℹ️  该服务不支持 response_format，已自动回退为「提示词要求 JSON + 本地解析」。")
            print(f"     原始报错：{getattr(backend, 'last_json_mode_error', '')[:160]}")
        else:
            print("  ℹ️  response_format 可用（原生 JSON 模式）")
    except LLMError as exc:
        print(f"  ❌ 失败：{exc}")
        return 1

    # ---------- 3) seed 兼容性（可选参数） ----------
    print("\n[3/4] seed 参数兼容性（不支持也没关系，代码会自动去掉）")
    try:
        llm.chat([{"role": "user", "content": "回答：1"}], temperature=0.0, max_tokens=8, seed=42)
        print("  ✅ seed 可用")
    except LLMError as exc:
        print(f"  ⚠️  seed 不可用（不影响使用）：{str(exc)[:120]}")

    # ---------- 4) 端到端（可选） ----------
    if args.ask:
        print(f"\n[4/4] 完整 Agent 提问：{args.ask}")
        from app.agent.core import NL2SQLAgent
        from app.db.schema import introspect
        from app.tracing import Tracer

        tracer = Tracer(enabled=True)
        agent = NL2SQLAgent(settings=SETTINGS, llm=LLMClient(settings=SETTINGS, tracer=tracer),
                            docs=introspect(SETTINGS.db_path), tracer=tracer)
        t0 = time.perf_counter()
        out = agent.run(args.ask)
        dt = time.perf_counter() - t0
        print(f"  选中表：{out.linked_tables}")
        print(f"  SQL   ：{out.sql or '(无)'}")
        print(f"  结果  ：{out.row_count} 行 {out.rows[:3]}")
        print(f"  结论  ：{out.answer[:160]}")
        print(f"  耗时  ：{dt:.1f}s  调用 {out.trace.get('llm_calls')} 次  "
              f"token {out.trace.get('total_tokens')}  "
              f"估算成本 {out.trace.get('cost')}")
    else:
        print("\n[4/4] 跳过端到端（加 --ask \"你的问题\" 可跑一次完整链路）")

    print("\n" + "=" * 70)
    print(f"自检通过 ✅  单次最小调用 {single_latency:.2f}s —— "
          f"按每题约 6 次调用估算，单题约 {single_latency * 6:.0f}s（本地 CPU 约 90~100s）")
    print("下一步：python -m eval.run_eval --limit 5 --tag cloud-smoke   # 小样本评测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
