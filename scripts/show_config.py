# -*- coding: utf-8 -*-
"""显示当前生效的配置（API Key 打码，不泄露到日志/终端）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS, ROOT as PROJECT_ROOT  # noqa: E402


def mask(key: str) -> str:
    if not key:
        return "❌ 未设置"
    return f"{key[:6]}…{key[-4:]}（长度 {len(key)}）"


print("配置文件 :", PROJECT_ROOT / ".env", "存在" if (PROJECT_ROOT / ".env").exists() else "不存在")
print("LLM 后端 :", SETTINGS.llm_backend)
print("模型     :", SETTINGS.model, "| 云端模型:", SETTINGS.openai_model)
print("base_url :", SETTINGS.openai_base)
print("api_key  :", mask(SETTINGS.openai_key))
print("向量模型 :", SETTINGS.embed_model or "（关闭）")
print("候选数 N :", SETTINGS.n_candidates, "| 自愈轮数:", SETTINGS.max_repair_rounds)
print("时间预算 :", SETTINGS.request_budget_s, "s | 单次超时:", SETTINGS.request_timeout, "s | 重试:", SETTINGS.max_retries)
