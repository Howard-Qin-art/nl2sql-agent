"""LangGraph 编排：StateGraph + 条件边 + Checkpointer + interrupt 澄清。

拓扑：
                    ┌──────────────┐
    START ─────────►│ guard_input  │──(空问题)──► END
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐      (有歧义)     ┌───────────┐
                    │ link_schema  │◄──────────────────│  clarify  │◄─ interrupt 挂起
                    └──────┬───────┘   澄清后重跑链接   └───────────┘
                           ▼
                    ┌──────────────┐
                    │check_ambiguity│──(无歧义)──┐
                    └──────────────┘            ▼
                                         ┌──────────────┐
                                         │   generate   │  N 候选采样
                                         └──────┬───────┘
                                                ▼
                                         ┌──────────────┐
                                         │   execute    │  护栏 + 只读执行 + 投票
                                         └──┬────────┬──┘
                                   (有结果) │        │ (全失败)
                                            ▼        ▼
                                   ┌────────────┐  ┌────────┐
                                   │ synthesize │◄─│ repair │◄─ 报错原文回注自愈
                                   └─────┬──────┘  └────────┘
                                         ▼
                                        END

Checkpointer 的作用：同一个 thread_id 下状态被持久化，多轮「那上个月呢」这类
指代追问不需要把历史重新拼一遍；interrupt 挂起后也能靠它原地恢复。
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.core import NL2SQLAgent, build_agent
from app.agent.nodes import (
    make_nodes,
    route_after_ambiguity,
    route_after_execute,
    route_after_guard,
    route_after_repair,
)
from app.agent.state import AgentState, new_state


def build_graph(agent: NL2SQLAgent | None = None, *, checkpointer: Any = None):
    agent = agent or build_agent()
    nodes = make_nodes(agent)

    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", route_after_guard, {"ok": "link_schema", "end": END})
    g.add_edge("link_schema", "check_ambiguity")
    g.add_conditional_edges("check_ambiguity", route_after_ambiguity,
                            {"clarify": "clarify", "generate": "generate"})
    g.add_edge("clarify", "link_schema")      # 澄清后带着补充信息重新链接
    g.add_edge("generate", "execute")
    g.add_conditional_edges("execute", lambda s: route_after_execute(s, agent),
                            {"synthesize": "synthesize", "repair": "repair"})
    g.add_conditional_edges("repair", lambda s: route_after_repair(s, agent),
                            {"synthesize": "synthesize", "repair": "repair"})
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


class GraphAgent:
    """图引擎的调用封装：对外暴露和 NL2SQLAgent.run 同样的返回类型。"""

    def __init__(self, agent: NL2SQLAgent | None = None, **kwargs: Any):
        self.agent = agent or build_agent(**kwargs)
        self.graph = build_graph(self.agent)
        self._thread_seq = 0

    def new_thread_id(self) -> str:
        self._thread_seq += 1
        return f"thread-{self._thread_seq}"

    def run(self, question: str, *, thread_id: str | None = None,
            history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """跑一轮；如果图被 interrupt 挂起，返回 {"interrupted": True, ...}。"""
        thread_id = thread_id or self.new_thread_id()
        config = {"configurable": {"thread_id": thread_id}}
        self.agent.tracer.reset()   # 逐轮归零，保证 trace 统计是「本轮」的
        result = self.graph.invoke(new_state(question, history), config=config)
        return self._render(result, thread_id, config)

    def resume(self, answer: str, *, thread_id: str, config: dict | None = None) -> dict[str, Any]:
        """用用户对澄清问题的回答恢复被挂起的图。"""
        cfg = config or {"configurable": {"thread_id": thread_id}}
        from langgraph.types import Command as _Command

        result = self.graph.invoke(_Command(resume=answer), config=cfg)
        return self._render(result, thread_id, cfg)

    def state(self, thread_id: str) -> dict[str, Any]:
        snap = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return dict(snap.values or {})

    # ---------- 内部 ----------
    def _render(self, result: dict[str, Any], thread_id: str, config: dict) -> dict[str, Any]:
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            payload = getattr(interrupts[0], "value", interrupts[0])
            return {"interrupted": True, "thread_id": thread_id, "clarify": payload, "state": result}
        return self.agent_result(result, thread_id)

    def agent_result(self, state: dict[str, Any], thread_id: str) -> dict[str, Any]:
        """把图状态转成与顺序引擎一致的输出结构（API/UI/评测共用），
        trace 用 Agent 真实 tracer 的汇总 —— 图引擎同样有完整的 token/耗时统计。"""
        from app.agent.core import _to_result

        tracer = self.agent.tracer
        prices = (self.agent.s.price_per_1k_prompt, self.agent.s.price_per_1k_completion)
        out = _to_result(dict(state), tracer, 0.0, prices).to_dict()
        out["thread_id"] = thread_id
        out["engine"] = "langgraph"
        return out
