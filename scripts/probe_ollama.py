# -*- coding: utf-8 -*-
"""探测 Ollama 原生 API：GPU 崩了的话，能否用 num_gpu=0 走 CPU。"""
import json
import time

import requests

BASE = "http://127.0.0.1:11434"


def chat(model, num_gpu, prompt="只回答两个字：可以", num_predict=16):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict, "num_gpu": num_gpu, "num_ctx": 4096},
    }
    t0 = time.time()
    r = requests.post(BASE + "/api/chat", json=body, timeout=600)
    dt = time.time() - t0
    print("%-12s num_gpu=%-2s -> HTTP %s  %.1fs" % (model, num_gpu, r.status_code, dt))
    if r.status_code == 200:
        j = r.json()
        content = (j.get("message") or {}).get("content")
        print("     content      :", repr(content)[:160])
        ev = j.get("eval_count") or 0
        ed = j.get("eval_duration") or 1
        print("     eval_count   :", ev, " tokens/s:", round(ev / ed * 1e9, 1))
        print("     load_duration: %.1fs" % ((j.get("load_duration") or 0) / 1e9))
    else:
        print("     body:", r.text[:300].replace("\n", " "))
    return r.status_code == 200


for model, gpu in [("qwen2.5:3b", 0), ("qwen2.5:7b", 0)]:
    try:
        chat(model, gpu)
    except Exception as e:
        print("ERR", model, gpu, type(e).__name__, str(e)[:200])

# JSON 结构化输出能力（format=json，Agent 里要用）
try:
    body = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "你只输出 JSON。"},
            {"role": "user", "content": '把这句话转成 JSON：{"sql": "SELECT 1", "tables": ["t"]}'},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 120, "num_gpu": 0, "num_ctx": 4096},
    }
    t0 = time.time()
    r = requests.post(BASE + "/api/chat", json=body, timeout=600)
    print("json mode -> HTTP %s %.1fs" % (r.status_code, time.time() - t0))
    if r.status_code == 200:
        raw = r.json()["message"]["content"]
        print("     raw   :", raw[:200])
        print("     parsed:", json.loads(raw))
    else:
        print("     body:", r.text[:300])
except Exception as e:
    print("json ERR", type(e).__name__, str(e)[:200])
