"""Agent 状态定义。

两个考点都在这里：
    1. TypedDict + `Annotated[..., operator.add]` reducer —— 并行/多轮节点往同一个 key 写时，
       到底是覆盖还是追加，由 reducer 决定。`notes` / `attempts` 用追加语义，
       避免自愈重试把前几轮的失败证据覆盖掉（线上排查全靠它）。
    2. 状态里只放**可序列化**的东西（dict/list/str/float），
       这样 Checkpointer 才能把状态存下来做多轮对话与中断恢复。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dict(left: dict | None, right: dict | None) -> dict:
    """dict 的 reducer：合并而不是覆盖（各节点只更新自己那几个字段）。"""
    merged: dict[str, Any] = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    # ---------- 输入 ----------
    question: str
    history: Annotated[list[dict[str, str]], operator.add]   # 多轮对话（供指代消解）

    # ---------- Schema Linking ----------
    linked_tables: list[str]
    schema_text: str
    link_debug: dict

    # ---------- 澄清（Human-in-the-loop） ----------
    needs_clarification: bool
    clarify_question: str
    clarified: bool
    clarifications: Annotated[list[dict[str, str]], operator.add]

    # ---------- 生成与执行 ----------
    candidates: dict                                  # CandidateSet.to_dict()
    candidate_plans: list[dict]
    chosen_sql: str
    result: dict                                      # 结果集（列/行/行数/耗时）
    rejected: list[dict]                              # 被护栏拒绝或执行失败的候选
    confidence: float

    # ---------- 自愈 ----------
    repair_rounds: int
    repair_history: Annotated[list[dict], operator.add]
    attempts: Annotated[list[dict], operator.add]

    # ---------- 输出 ----------
    answer: str
    explanation: str
    error: str | None
    early_exit: bool                                  # 快速通道/空输入等：图直接结束，不再往下跑

    # ---------- 观测 ----------
    stats: Annotated[dict, _merge_dict]
    notes: Annotated[list[str], operator.add]


def new_state(question: str, history: list[dict[str, str]] | None = None) -> AgentState:
    return AgentState(
        question=question,
        history=list(history or []),
        notes=[],
        attempts=[],
        repair_history=[],
        clarifications=[],
        stats={},
        repair_rounds=0,
        needs_clarification=False,
        clarified=False,
        confidence=1.0,
        error=None,
    )
