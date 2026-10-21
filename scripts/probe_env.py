# -*- coding: utf-8 -*-
"""环境探测：确认 LangGraph API、openai SDK 版本、Ollama 可用性与 JSON 模式。"""
import importlib
import json
import sys
import time
import urllib.request

print("python:", sys.version.split()[0], sys.executable)

# ---------- 1. LangGraph API 面 ----------
try:
    import langgraph
    print("langgraph:", getattr(langgraph, "__version__", "?"))
except Exception as e:
    print("langgraph import ERR:", e)

for mod, names in [
    ("langgraph.graph", ["StateGraph", "START", "END", "MessagesState"]),
    ("langgraph.checkpoint.memory", ["MemorySaver", "InMemorySaver"]),
    ("langgraph.types", ["interrupt", "Command"]),
]:
    try:
        m = importlib.import_module(mod)
        got = [n for n in names if hasattr(m, n)]
        print("  %-32s %s" % (mod, got or "none of " + str(names)))
    except Exception as e:
        print("  %-32s ERR %s" % (mod, e))

# ---------- 2. openai SDK ----------
try:
    import openai
    print("openai SDK:", openai.__version__)
    from openai import OpenAI  # noqa
    print("  OpenAI client: import OK")
except Exception as e:
    print("openai ERR:", e)

# ---------- 3. Ollama ----------
models = []
try:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=8) as r:
        models = [m["name"] for m in json.load(r).get("models", [])]
    print("ollama models:", models)
except Exception as e:
    print("ollama ERR:", e)

# ---------- 4. 真的跑一次对话 + JSON 模式 ----------
if models:
    target = "qwen2.5:7b" if any(m.startswith("qwen2.5:7b") for m in models) else models[0]
    try:
        from openai import OpenAI

        c = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
        t0 = time.time()
        r = c.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": "只回答两个字：可以"}],
            temperature=0,
            max_tokens=16,
        )
        print("chat  %s  %.1fs  -> %r" % (target, time.time() - t0, r.choices[0].message.content))
        print("      usage:", r.usage)
    except Exception as e:
        print("chat ERR:", type(e).__name__, e)

    try:
        from openai import OpenAI

        c = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
        t0 = time.time()
        r = c.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": '输出 JSON：{"sql": "SELECT 1", "tables": ["t"]}'}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=80,
        )
        raw = r.choices[0].message.content
        print("json  %.1fs  -> %r" % (time.time() - t0, raw))
        try:
            print("      parsed:", json.loads(raw))
        except Exception as e:
            print("      JSON 解析失败:", e)
    except Exception as e:
        print("json ERR:", type(e).__name__, e)

# ---------- 5. 关键库 ----------
for n in ["sqlglot", "fastapi", "pytest", "tiktoken", "sqlalchemy", "streamlit", "pandas", "pydantic"]:
    try:
        importlib.import_module(n)
        print("  %-12s OK" % n)
    except Exception:
        print("  %-12s MISSING" % n)
