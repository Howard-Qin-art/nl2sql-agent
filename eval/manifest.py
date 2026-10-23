# -*- coding: utf-8 -*-
"""评测清单（manifest）：把一次评测的「现场信息」固化下来。

解决的问题：跑完只记了模型名，过几天改了提示词，就说不清"83.7% 到底对应哪一版"。
所以每次评测都落一份 manifest：代码版本、关键配置、题库文件指纹、环境信息。

有了它，"这个数字还能复现吗"就能直接回答，而不是靠记忆。
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# 会显著影响评测结果的配置项（按这些判断"两次评测能不能横向比"）
KEY_SETTINGS = [
    "llm_backend", "model", "openai_model", "n_candidates", "temperature",
    "use_llm_rerank", "max_repair_rounds", "strict", "embed_model",
    "request_budget_s", "fast_path",
]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # git 不存在/超时都不该让评测挂掉
        return ""


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_manifest(settings: Any, questions_path: Path | None = None) -> dict[str, Any]:
    """收集一次评测的可复现信息。"""
    cfg = {}
    for name in KEY_SETTINGS:
        if hasattr(settings, name):
            cfg[name] = getattr(settings, name)

    prompt_file = ROOT / "app" / "prompts.py"
    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "--short", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # 有未提交改动时标记 dirty —— 这种情况下数字严格来说不可复现
            "dirty": bool(_git("status", "--porcelain")),
        },
        "settings": cfg,
        "fingerprints": {
            "prompts.py": _sha256(prompt_file),
            "schema_linker.py": _sha256(ROOT / "app" / "schema_linker.py"),
            "guardrails.py": _sha256(ROOT / "app" / "guardrails.py"),
            "questions": _sha256(questions_path) if questions_path else "",
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    return manifest


def describe(manifest: dict[str, Any]) -> str:
    """把 manifest 压成一行人类可读的复现信息（写进 markdown 报告用）。"""
    g = manifest.get("git", {})
    s = manifest.get("settings", {})
    f = manifest.get("fingerprints", {})
    dirty = "（有未提交改动，结果不严格可复现）" if g.get("dirty") else ""
    model = s.get("openai_model") if s.get("llm_backend") == "openai" else s.get("model")
    return (
        f"复现信息：commit `{g.get('commit') or '?'}`@{g.get('branch') or '?'}{dirty}｜"
        f"模型 `{model}`｜backend `{s.get('llm_backend')}`｜N={s.get('n_candidates')}｜"
        f"strict={s.get('strict')}｜prompts.py#`{f.get('prompts.py')}`｜题库#`{f.get('questions')}`"
    )
