"""LangGraph 节点：把 core.NL2SQLAgent 的方法包成图的节点，并写路由条件。

刻意保持「节点极薄」：业务逻辑留在 core 里，节点只做状态适配，
这样两条编排路径（顺序循环 / StateGraph）行为一致，可互相校验。
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.agent.core import NL2SQLAgent
from app.agent.state import AgentState


def make_nodes(agent: NL2SQLAgent) -> dict[str, Any]:
    def guard_input(state: AgentState) -> dict:
        return agent.guard_input(state)

    def link_schema(state: AgentState) -> dict:
        return agent.link_schema(state)

    def check_ambiguity(state: AgentState) -> dict:
        return agent.check_ambiguity(state)

    def clarify(state: AgentState) -> dict:
        """Human-in-the-loop：interrupt 会挂起整张图，把问题抛给调用方；
        恢复时（Command(resume=...)）从本节点重放，拿到用户答案后继续。
        这是「多轮澄清 + 可恢复执行」的关键，也是面试最容易被追问的点。"""
        answer = interrupt({
            "type": "clarification",
            "question": state.get("clarify_question") or state.get("question", ""),
            "hint": "问题存在歧义，需要用户补充信息（例如时间范围、指标口径）",
        })
        return agent.apply_clarification(state, str(answer))

    def generate(state: AgentState) -> dict:
        return agent.generate(state)

    def execute(state: AgentState) -> dict:
        return agent.execute_candidates(state)

    def repair(state: AgentState) -> dict:
        return agent.repair(state)

    def synthesize(state: AgentState) -> dict:
        return agent.synthesize(state)

    return {
        "guard_input": guard_input,
        "link_schema": link_schema,
        "check_ambiguity": check_ambiguity,
        "clarify": clarify,
        "generate": generate,
        "execute": execute,
        "repair": repair,
        "synthesize": synthesize,
    }


# ---------------------------------------------------------------- 路由条件
def route_after_guard(state: AgentState) -> str:
    """空输入、或走了寒暄快速通道 -> 直接结束，不进入检索/生成。"""
    return "end" if (state.get("error") or state.get("early_exit")) else "ok"


def route_after_ambiguity(state: AgentState) -> str:
    # 已经澄清过一次就不再打断（避免死循环）
    if state.get("needs_clarification") and not state.get("clarified"):
        return "clarify"
    return "generate"


def route_after_execute(state: AgentState, agent: NL2SQLAgent) -> str:
    if state.get("result"):
        return "synthesize"
    if state.get("repair_rounds", 0) < agent.s.max_repair_rounds:
        return "repair"
    return "synthesize"


def route_after_repair(state: AgentState, agent: NL2SQLAgent) -> str:
    if state.get("result"):
        return "synthesize"
    if state.get("repair_rounds", 0) < agent.s.max_repair_rounds:
        return "repair"
    return "synthesize"
