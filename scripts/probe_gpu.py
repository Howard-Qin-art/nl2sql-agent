# -*- coding: utf-8 -*-
"""测 Ollama 在 GPU 上到底能不能跑，以及比 CPU 快多少。

对比三组：
    1) qwen2.5:7b  CPU（num_gpu=0） vs GPU（num_gpu=-1，全部层上显存）
    2) qwen2.5:3b  GPU（7B 显存吃紧时的备选）
    3) bge-m3 向量模型 GPU（之前和 llama-server 一起崩过，重点验证）

指标用 Ollama 自己的 eval_count / eval_duration 算 tok/s，不用墙钟，避免把加载时间算进去。
用法：python scripts/probe_gpu.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402

BASE = SETTINGS.ollama_base.rstrip("/")
PROMPT = "用一句话解释什么是数据库索引。"


def unload(model: str) -> None:
    """先把模型从显存/内存卸载，避免上一轮的 num_gpu 设置影响下一轮。"""
    try:
        requests.post(f"{BASE}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def ps() -> list[dict]:
    try:
        return requests.get(f"{BASE}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return []


def chat(model: str, num_gpu: int, num_ctx: int = 4096) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {"num_gpu": num_gpu, "num_ctx": num_ctx, "num_predict": 128, "temperature": 0.2},
        "keep_alive": "5m",
    }
    r = requests.post(f"{BASE}/api/chat", json=payload, timeout=600)
    if r.status_code >= 400:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    eval_count = data.get("eval_count") or 0
    eval_ns = data.get("eval_duration") or 0
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns else 0
    loaded = ps()
    proc = loaded[0].get("size_vram") and "GPU" or "CPU" if loaded else "?"
    vram = (loaded[0].get("size_vram") or 0) / 1e9 if loaded else 0
    return {"ok": True, "tok_s": round(tps, 1), "eval_count": eval_count, "processor": proc,
            "vram_gb": round(vram, 2), "text": (data.get("message") or {}).get("content", "")[:40]}


def embed(num_gpu: int) -> dict:
    payload = {"model": "bge-m3", "input": ["数据库索引", "退款金额合计"], "options": {"num_gpu": num_gpu}}
    try:
        r = requests.post(f"{BASE}/api/embed", json=payload, timeout=300)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if r.status_code >= 400:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    vecs = data.get("embeddings") or []
    return {"ok": bool(vecs), "dim": len(vecs[0]) if vecs else 0}


def show(label: str, res: dict) -> None:
    if res.get("ok"):
        extra = f" dim={res['dim']}" if "dim" in res else f" {res['tok_s']} tok/s  {res['processor']}  VRAM={res['vram_gb']}GB"
        print(f"  [PASS] {label}{extra}")
    else:
        print(f"  [FAIL] {label}  {res.get('error')}")


def main() -> int:
    print(f"Ollama: {BASE}\n")

    print("=== 1) qwen2.5:7b：CPU vs GPU ===")
    unload("qwen2.5:7b")
    cpu = chat("qwen2.5:7b", num_gpu=0)
    show("7b CPU (num_gpu=0)", cpu)
    unload("qwen2.5:7b")
    gpu = chat("qwen2.5:7b", num_gpu=-1, num_ctx=2048)
    show("7b GPU (num_gpu=-1)", gpu)
    if cpu.get("ok") and gpu.get("ok") and cpu["tok_s"]:
        print(f"      → 加速 {gpu['tok_s'] / cpu['tok_s']:.1f}×")

    print("\n=== 2) qwen2.5:3b：GPU（显存吃紧时的备选）===")
    unload("qwen2.5:3b")
    small = chat("qwen2.5:3b", num_gpu=-1, num_ctx=4096)
    show("3b GPU", small)

    print("\n=== 3) bge-m3 向量模型（之前和 llama-server 一起崩过）===")
    unload("bge-m3")
    e_cpu = embed(num_gpu=0)
    show("bge-m3 CPU", e_cpu)
    unload("bge-m3")
    e_gpu = embed(num_gpu=-1)
    show("bge-m3 GPU", e_gpu)

    print("\n=== 结论 ===")
    issues = []
    if not gpu.get("ok"):
        issues.append("7B 上不了 GPU（显存或驱动）")
    if not e_gpu.get("ok"):
        issues.append("bge-m3 上不了 GPU")
    if issues:
        print("  ⚠️ " + "；".join(issues) + " → 保持 num_gpu=0，或改用 3b")
    else:
        print("  ✅ 7B 与 bge-m3 都能走 GPU；建议设 NL2SQL_OLLAMA_NUM_GPU=-1")
        if gpu.get("ok") and cpu.get("ok") and cpu["tok_s"]:
            print(f"     实测加速 {gpu['tok_s'] / cpu['tok_s']:.1f}×（{cpu['tok_s']} → {gpu['tok_s']} tok/s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
