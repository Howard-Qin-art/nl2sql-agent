# -*- coding: utf-8 -*-
"""探测 Ollama 的 embedding 接口：bge-m3 走哪个端点、向量维度、批量与耗时。

用法：python scripts/probe_embed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402

BASE = SETTINGS.ollama_base
MODEL = "bge-m3"
TEXTS = ["华东大区的成交额", "退款原因分布", "每个大类销量最高的商品"]

print(f"Ollama: {BASE}   模型: {MODEL}")


def try_endpoint(name: str, url: str, payload: dict, pick):
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        dt = time.time() - t0
        if resp.status_code != 200:
            print(f"  [{name}] HTTP {resp.status_code}  {dt:.2f}s  {resp.text[:200]}")
            return None
        vecs = pick(resp.json())
        dim = len(vecs[0]) if vecs else 0
        print(f"  [{name}] OK  {dt:.2f}s  向量数={len(vecs)}  维度={dim}")
        return vecs
    except Exception as exc:
        print(f"  [{name}] ERR {type(exc).__name__}: {str(exc)[:160]}")
        return None


print("\n1) 新端点 /api/embed（批量，推荐）")
v_new = try_endpoint(
    "embed/batch", f"{BASE}/api/embed",
    {"model": MODEL, "input": TEXTS},
    lambda j: j.get("embeddings") or [],
)

print("\n2) 老端点 /api/embeddings（单条，兼容用）")
v_old = try_endpoint(
    "embeddings", f"{BASE}/api/embeddings",
    {"model": MODEL, "prompt": TEXTS[0]},
    lambda j: [j["embedding"]] if j.get("embedding") else [],
)

print("\n3) 单条延迟（新端点，连打 3 次，看是否已加载常驻）")
for i in range(3):
    t0 = time.time()
    r = requests.post(f"{BASE}/api/embed", json={"model": MODEL, "input": TEXTS[i % len(TEXTS)]}, timeout=120)
    print(f"  第{i + 1}次: {time.time() - t0:.2f}s  HTTP {r.status_code}")

if v_new and v_old:
    a, b = v_new[0], v_old[0]
    same = len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    print(f"\n两个端点结果一致: {same}")

# ---------------------------------------------------------------- CPU 模式
# 本机 GPU 驱动与 Ollama 的 CUDA 内核不兼容（qwen 也是同样问题），
# 所以 embedding 也必须强制 CPU 跑：给接口传 options.num_gpu = 0。
print("\n4) 强制 CPU（options.num_gpu=0）")
cpu_new = try_endpoint(
    "embed+num_gpu=0", f"{BASE}/api/embed",
    {"model": MODEL, "input": TEXTS, "options": {"num_gpu": 0}},
    lambda j: j.get("embeddings") or [],
)
cpu_old = try_endpoint(
    "embeddings+num_gpu=0", f"{BASE}/api/embeddings",
    {"model": MODEL, "prompt": TEXTS[0], "options": {"num_gpu": 0}},
    lambda j: [j["embedding"]] if j.get("embedding") else [],
)
if cpu_new:
    print("\n5) CPU 模式批量耗时（9 张表，模拟真实 Schema Linking）")
    docs = [f"表 t{i} 的业务说明与列名" for i in range(9)]
    t0 = time.time()
    r = requests.post(f"{BASE}/api/embed", json={"model": MODEL, "input": docs, "options": {"num_gpu": 0}}, timeout=300)
    print(f"  9 条文本: {time.time() - t0:.2f}s  HTTP {r.status_code}")

print(f"\n结论：代码里应使用 {'/api/embed' if cpu_new else '/api/embeddings'} 端点"
      f"{'（并带 options.num_gpu=0）' if cpu_new else '，且 CPU 模式仍不可用'}")

