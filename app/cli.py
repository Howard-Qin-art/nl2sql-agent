"""命令行入口：ask / serve / ui / seed / schema / eval。

    python -m app.cli ask "华东大区2024年每个月的成交额是多少？"
    python -m app.cli ask "订单情况怎么样？" --clarify      # 演示歧义反问
    python -m app.cli serve --port 8000                    # 起 API
    python -m app.cli ui                                   # 起 Streamlit
    python -m app.cli seed                                 # 重建示例库
    python -m app.cli schema                               # 打印库表画像
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from app.agent.core import NL2SQLAgent
from app.agent.graph import GraphAgent
from app.config import SETTINGS
from app.db.schema import all_tables_prompt, introspect
from app.db.seed import seed_database
from app.llm import LLMClient
from app.tracing import Tracer


def _agent(engine: str):
    tracer = Tracer(enabled=SETTINGS.trace_enabled)
    agent = NL2SQLAgent(settings=SETTINGS, llm=LLMClient(settings=SETTINGS, tracer=tracer), tracer=tracer)
    return GraphAgent(agent=agent) if engine == "graph" else agent


def _cmd_ask_tools(args: argparse.Namespace) -> int:
    """Plan B：让模型自己决定调哪个工具（tools=本地工具 / mcp=通过 MCP 协议连工具）。"""
    from app.agent.tool_loop import local_tools, run_tool_agent

    tracer = Tracer(enabled=SETTINGS.trace_enabled)
    llm = LLMClient(settings=SETTINGS, tracer=tracer)
    mode = args.mode

    if mode == "mcp":
        from pathlib import Path

        from app.agent.mcp_client import mcp_tools

        server = Path(__file__).resolve().parent.parent / "mcp_server.py"
        with mcp_tools(command=sys.executable, args=[str(server)],
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as tools:
            print(f"[MCP] 从 server 动态发现工具：{[t.name for t in tools]}")
            result = run_tool_agent(args.question, llm=llm, tools=tools,
                                    max_steps=args.steps, tracer=tracer)
    else:
        tools = local_tools(SETTINGS)
        print(f"[tools] 本地工具：{[t.name for t in tools]}")
        result = run_tool_agent(args.question, llm=llm, tools=tools,
                                max_steps=args.steps, tracer=tracer)

    print("=" * 72)
    print(f"问题   : {args.question}")
    print(f"模式   : {mode}（模型自主选择工具，最大 {args.steps} 轮）")
    for c in result.tool_calls:
        args_preview = json.dumps(c.arguments, ensure_ascii=False)[:100]
        print(f"  ↳ 第{c.step}轮 {c.name}({args_preview})  {c.latency_ms / 1000:.1f}s"
              + ("" if c.ok else "  [工具报错]"))
    print(f"结论   : {result.answer or '（无）'}")
    if result.sql:
        print(f"SQL    : {result.sql}")
    if result.columns:
        print(f"结果   : {result.row_count} 行  列={result.columns}")
    print(f"步数   : {result.steps}    耗时 : {result.latency_ms / 1000:.1f}s")
    if result.error:
        print(f"错误   : {result.error}")
    return 0 if result.ok else 1


def _read_answer(prompt: str, default: str) -> str:
    """读用户输入；非交互环境（管道/重定向时 input 会抛 EOFError）必须兜住 ——
    否则一次「Agent 反问」会把 CLI 直接弄崩，这是实际踩到的坑。"""
    try:
        return input(prompt).strip() or default
    except EOFError:
        print("（非交互环境，使用默认澄清答案）")
        return default


def cmd_ask(args: argparse.Namespace) -> int:
    if getattr(args, "mode", "pipeline") in ("tools", "mcp"):
        return _cmd_ask_tools(args)
    engine = _agent(args.engine)
    default_answer = args.answer or "按全部时间范围统计"

    def clarify(question: str) -> str:
        print(f"\n[Agent 反问] {question}")
        return _read_answer("你的补充（直接回车=用默认答案）: ", default_answer)

    if args.engine == "graph":
        out = engine.run(args.question)
        if out.get("interrupted"):
            print(f"\n[Agent 反问] {out['clarify']['question']}")
            if args.answer:
                answer = args.answer
                print(f"（使用 --answer：{answer}）")
            else:
                answer = _read_answer("你的补充: ", default_answer)
            out = engine.resume(answer, thread_id=out["thread_id"])
        _print_result(out)
    else:
        res = engine.run(args.question, clarify_handler=clarify if args.clarify else None)
        _print_result(res.to_dict())
    return 0


def _print_result(res: dict) -> None:
    print("\n" + "=" * 72)
    print("问题   :", res.get("question", ""))
    print("选中表 :", ", ".join(res.get("linked_tables") or []))
    print("SQL    :", res.get("sql") or "(无)")
    result = res.get("result") or {}
    rows = res.get("rows") or result.get("rows") or []
    columns = res.get("columns") or result.get("columns") or []
    print(f"结果   : {res.get('row_count', len(rows))} 行  列={columns}")
    for row in rows[:8]:
        print("        ", row)
    if len(rows) > 8:
        print(f"         ... 其余 {len(rows) - 8} 行省略")
    print("结论   :", res.get("answer", ""))
    cand = res.get("candidates") or {}
    if cand:
        print(f"投票   : 一致度 {cand.get('agreement')}  候选 {cand.get('n')} 条，"
              f"成功执行 {cand.get('ok')} 条")
        for c in cand.get("candidates", [])[:5]:
            print(f"         #{c['index']} ok={c['ok']} rows={c['rows']} {str(c['sql'])[:80]}")
    if res.get("repair_rounds"):
        print("自愈   : 第", res["repair_rounds"], "轮修复成功")
    if res.get("error"):
        print("错误   :", res["error"])
    trace = res.get("trace") or {}
    if trace:
        print(f"耗时   : {trace.get('elapsed_ms')} ms | LLM {trace.get('llm_calls')} 次 | "
              f"token {trace.get('total_tokens')} | 成本 {trace.get('cost')}")
    if res.get("trace_text"):
        print("Trace  :")
        print(res["trace_text"])
    print("=" * 72)


def cmd_seed(_args: argparse.Namespace) -> int:
    path = seed_database(SETTINGS.db_path)
    print("示例库已生成:", path)
    return 0


def cmd_schema(_args: argparse.Namespace) -> int:
    docs = introspect(SETTINGS.db_path)
    print(all_tables_prompt(docs))
    print()
    for d in docs:
        print(d.render())
        print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    ui_path = str((SETTINGS.db_path.parent.parent / "ui" / "streamlit_app.py"))
    return subprocess.call([sys.executable, "-m", "streamlit", "run", ui_path, "--server.port", str(args.port)])


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.run_eval import main as eval_main

    argv = ["--limit", str(args.limit)] if args.limit else []
    sys.argv = ["run_eval", *argv]
    return eval_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2sql", description="NL2SQL 智能查询 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="问一个业务问题")
    p_ask.add_argument("question")
    p_ask.add_argument("--engine", choices=["core", "graph"], default="graph")
    p_ask.add_argument("--mode", choices=["pipeline", "tools", "mcp"], default="pipeline",
                       help="pipeline=固定流程(默认) | tools=模型自主调工具 | mcp=通过 MCP 协议连工具")
    p_ask.add_argument("--steps", type=int, default=6, help="tools/mcp 模式的最大工具调用轮数")
    p_ask.add_argument("--clarify", action="store_true", help="core 引擎下开启交互式澄清")
    p_ask.add_argument("--answer", default=None, help="预置澄清答案（非交互环境用，避免等待输入）")
    p_ask.set_defaults(func=cmd_ask)

    p_seed = sub.add_parser("seed", help="生成/重建示例库")
    p_seed.set_defaults(func=cmd_seed)

    p_schema = sub.add_parser("schema", help="打印库表画像")
    p_schema.set_defaults(func=cmd_schema)

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_ui = sub.add_parser("ui", help="启动 Streamlit 界面")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.set_defaults(func=cmd_ui)

    p_eval = sub.add_parser("eval", help="跑评测")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
