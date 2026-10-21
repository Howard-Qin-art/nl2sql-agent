# -*- coding: utf-8 -*-
"""MCP Client 的三条硬测试（对应"简历上写 MCP Client 站不站得住"）：

    1. 动态发现：连上后拿到的工具列表 == server 声明的，不是硬编码。
       证明方式：连一个**工具集完全不同**的假 server（whoami/add_numbers/echo），
       如果客户端报出来的就是这三个，说明工具清单来自协议，而不是代码里的常量。
    2. 调用转发：同步 call() 能拿到**子进程里真实计算**的结果（加法由子进程算，SQL 由子进程查）。
    3. 关闭不泄漏：close() 之后子进程**真的退出**（拿 pid 判定进程存活），
       并且反复开关不会留下僵尸进程。

注意：这三条都需要给子进程创建管道。某些受限沙箱禁止创建命名管道（Windows 下报
WinError 5），此时会自动 skip —— 在 GitHub Actions（Linux）和普通开发机上是真跑的。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="需要 mcp SDK")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.mcp_client import MCPToolset  # noqa: E402

MAIN_SERVER = ROOT / "mcp_server.py"
DUMMY_SERVER = ROOT / "tests" / "fixtures" / "dummy_mcp_server.py"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def _start(server_path: Path) -> MCPToolset:
    toolset = MCPToolset(sys.executable, [str(server_path)], cwd=str(ROOT), env=ENV, timeout=60)
    try:
        return toolset.start()
    except (PermissionError, OSError) as exc:
        pytest.skip(f"当前环境不允许创建子进程管道：{exc}")
    except RuntimeError as exc:
        msg = str(exc)
        if "PermissionError" in msg or "WinError 5" in msg or "拒绝访问" in msg:
            pytest.skip(f"当前沙箱禁止创建命名管道：{msg[:120]}")
        raise


def _pid_alive(pid: int) -> bool:
    """跨平台判断进程是否还活着。"""
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=15).stdout
        except Exception:                      # 判断不了就当作还活着，避免误报通过
            return True
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------- 1) 动态发现
def test_discovers_tools_from_dummy_server():
    """工具集完全不同的假 server —— 客户端报出来的必须是它声明的工具。"""
    ts = _start(DUMMY_SERVER)
    try:
        names = sorted(t.name for t in ts.tools())
        assert names == ["add_numbers", "echo", "whoami"], names
        # 主项目的工具名一个都不该出现，否则说明是硬编码
        assert "run_sql" not in names and "get_schema" not in names
        for spec in ts.tools():
            assert spec.description, f"{spec.name} 缺少说明"
            assert spec.parameters.get("type") == "object"
    finally:
        ts.close()


def test_discovers_tools_from_main_server():
    ts = _start(MAIN_SERVER)
    try:
        names = sorted(t.name for t in ts.tools())
        assert names == ["ask", "get_schema", "list_tables", "run_sql"], names
        by_name = {t.name: t for t in ts.tools()}
        assert "sql" in (by_name["run_sql"].parameters.get("properties") or {})
        assert "tables" in (by_name["get_schema"].parameters.get("properties") or {})
    finally:
        ts.close()


# ---------------------------------------------------------------- 2) 调用转发
def test_call_forwarding_computes_in_child_process():
    """加法是**子进程**算的，不是本地算的 —— 证明调用真的转发过去了。"""
    ts = _start(DUMMY_SERVER)
    try:
        raw = ts.call("add_numbers", {"a": 17, "b": 25})
        payload = json.loads(raw)
        assert payload["sum"] == 42, payload
        assert payload["a"] == 17 and payload["b"] == 25
        assert ts.call("echo", {"text": "你好"}) == "echo:你好"
    finally:
        ts.close()


def test_call_forwarding_real_sql_result():
    """真实 server 的 run_sql：返回的是数据库真查出来的行，并且护栏补了 LIMIT。"""
    ts = _start(MAIN_SERVER)
    try:
        payload = json.loads(ts.call("run_sql", {"sql": "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status"}))
        assert payload["ok"] is True
        assert payload["row_count"] > 0
        assert payload["columns"] == ["status", "cnt"]
        assert "LIMIT" in payload["sql"].upper()
        assert len(payload["rows"]) == payload["row_count"]
    finally:
        ts.close()


def test_call_forwarding_guardrail_rejection():
    """危险 SQL 的拒绝理由是**子进程**给出的，也要能原样传回。"""
    ts = _start(MAIN_SERVER)
    try:
        payload = json.loads(ts.call("run_sql", {"sql": "DROP TABLE orders"}))
        assert payload["ok"] is False
        assert payload["reasons"], payload
    finally:
        ts.close()


# ---------------------------------------------------------------- 3) 关闭不泄漏
def test_close_really_kills_child_process():
    """close() 之后子进程必须真的退出（这是最容易写错的地方）。"""
    ts = _start(DUMMY_SERVER)
    pid = int(ts.call("whoami", {}))
    assert _pid_alive(pid), f"刚启动时 pid {pid} 应该活着"
    ts.close()
    assert _wait_dead(pid, timeout=10), f"close() 之后 pid {pid} 仍然存活 —— 有进程泄漏"
    # 关闭后不能再调用（会话已销毁），否则就是 use-after-close
    with pytest.raises(RuntimeError):
        ts.call("whoami", {})


def test_repeated_open_close_leaves_no_zombies():
    """连续开关 3 次，每次的子进程都要退出，不能越积越多。"""
    pids: list[int] = []
    for _ in range(3):
        ts = _start(DUMMY_SERVER)
        pids.append(int(ts.call("whoami", {})))
        ts.close()
    assert len(set(pids)) == 3, f"每次应当是新的子进程，实际 {pids}"
    for pid in pids:
        assert _wait_dead(pid, timeout=10), f"pid {pid} 泄漏"
