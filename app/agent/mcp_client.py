# -*- coding: utf-8 -*-
"""MCP Client：把任意 MCP server 的工具接进工具循环（Plan B 的"连外部工具"那半）。

实现要点（都不是教科书上的简单写法）：
    1. mcp SDK 的客户端 API 是 async 的，而本项目的工具循环是同步的；
       所以在**后台线程**里跑一个独立事件循环，长期持有 ClientSession，
       同步侧用 `run_coroutine_threadsafe` 把调用投递过去 —— 这样一次连接能服务多轮工具调用，
       而不是每次调用都重启一个 server 子进程。
    2. 工具列表**动态发现**：server 报什么就有什么，不需要在客户端写死；
       每个工具的 `input_schema` 直接转成 OpenAI tool schema 给模型。
    3. 只做"能力发现 + 转发"，不碰业务逻辑；安全仍由 server 侧（本项目是 run_sql 的护栏）保证。

用法（配合 app/agent/tool_loop.py）：
    with mcp_tools(command=sys.executable, args=["mcp_server.py"]) as tools:
        result = run_tool_agent("华东上月成交额", llm=llm, tools=tools)
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from app.agent.tool_loop import ToolSpec


class MCPToolset:
    """一个 MCP 会话，暴露 tools() 与同步 call()。"""

    def __init__(self, command: str, args: Sequence[str], *, cwd: str | None = None,
                 env: dict[str, str] | None = None, timeout: float = 120.0):
        self.command = command
        self.args = list(args)
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._tools: list[Any] = []

    # ---------- 启动 / 关闭 ----------
    def start(self) -> "MCPToolset":
        self._thread = threading.Thread(target=self._thread_main, name="mcp-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout):
            raise TimeoutError("MCP 连接超时（server 没有完成握手）")
        if self._error is not None:
            raise RuntimeError(f"MCP 连接失败：{type(self._error).__name__}: {self._error}")
        return self

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(command=self.command, args=self.args, cwd=self.cwd, env=self.env)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._tools = (await session.list_tools()).tools
                    self._stop = asyncio.Event()
                    self._ready.set()
                    await self._stop.wait()          # 一直挂着，直到 close()
                    self._session = None
        except BaseException as exc:                 # noqa: BLE001 - 要把错误带回同步侧
            self._error = exc
            self._ready.set()

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> "MCPToolset":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------- 工具 ----------
    def tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for tool in self._tools:
            schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
            # MCP 的 schema 里常带 title，模型不需要，去掉让 prompt 更干净
            schema = {k: v for k, v in schema.items() if k != "title"}
            specs.append(ToolSpec(
                name=tool.name,
                description=(tool.description or "")[:500],
                parameters=schema,
                func=self._make_caller(tool.name),
            ))
        return specs

    def _make_caller(self, name: str):
        def caller(**kwargs: Any) -> str:
            return self.call(name, kwargs)
        return caller

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """同步版工具调用：投递到后台事件循环执行，等结果。"""
        if self._loop is None or self._session is None:
            raise RuntimeError("MCP 会话未就绪（先 start()）")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop)
        result = future.result(timeout=self.timeout)
        parts = []
        for block in (result.content or []):
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)


@contextmanager
def mcp_tools(command: str, args: Sequence[str], *, cwd: str | None = None,
              env: dict[str, str] | None = None) -> Iterator[list[ToolSpec]]:
    """上下文管理器：打开 MCP 会话 -> 拿到工具列表 -> 用完关闭。"""
    toolset = MCPToolset(command, args, cwd=cwd, env=env).start()
    try:
        yield toolset.tools()
    finally:
        toolset.close()
