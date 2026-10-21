# -*- coding: utf-8 -*-
"""测试专用的迷你 MCP server（工具集和主项目完全不同）。

存在的唯一目的：证明 MCP Client 是**真的动态发现工具**，而不是把主项目的工具名写死。
如果客户端连上这个 server 后发现的是 whoami/add_numbers/echo，就说明工具列表来自协议，
而不是来自代码里的常量；同时它还被用来验证「关闭后子进程真的退出」（whoami 返回真实 pid）。

不要在生产里用。
"""
from __future__ import annotations

import os
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="dummy-mcp-server",
    description="测试用迷你 server：whoami / add_numbers / echo",
    version="0.1.0",
)


@server.tool(name="whoami", description="返回当前 server 进程的 PID（用于验证关闭后子进程确实退出）",
             structured_output=False)
def whoami() -> str:
    return str(os.getpid())


@server.tool(name="add_numbers", description="把两个整数相加，返回 JSON",
             structured_output=False)
def add_numbers(a: int, b: int) -> str:
    return f'{{"a": {a}, "b": {b}, "sum": {a + b}}}'


@server.tool(name="echo", description="原样回显传入的文本", structured_output=False)
def echo(text: str) -> str:
    return f"echo:{text}"


if __name__ == "__main__":
    server.run(transport="stdio")
