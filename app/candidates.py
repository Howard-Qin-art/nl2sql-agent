"""候选生成 + 自一致性投票（Self-Consistency）。

NL2SQL 里「同一句问题让模型采样 N 次、执行后按结果集投票」比只跑一次稳得多：
    - 采样温度 0.7、不同随机种子，得到 N 条**语义可能不同**的 SQL；
    - 每条都过一遍 AST 护栏（非法直接淘汰，不浪费执行）；
    - 都执行一遍，把结果集规范化（列序/行序/别名无关）后做指纹；
    - 指纹出现次数最多的即胜出（多数投票）；平票时取执行更快、行数更少的那条。

注意投票比的是**结果**而不是 SQL 文本：`COUNT(*)` 和 `COUNT(order_id)` 写法不同但结果一致，
按文本比对会误判，这也是评估里不能用字符串精确匹配的原因。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field

from app.db.executor import ExecResult, SQLExecutor
from app.guardrails import GuardResult, validate_sql


class SQLPlan(BaseModel):
    """模型结构化输出的契约（pydantic 校验，字段缺失也能兜住）。"""

    sql: str = ""
    tables_used: list[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = 1.0
    needs_clarification: bool = False
    clarify_question: str = ""


@dataclass
class Candidate:
    index: int
    plan: SQLPlan
    guard: GuardResult | None = None
    exec: ExecResult | None = None
    seed: int | None = None

    @property
    def sql(self) -> str:
        if self.guard and self.guard.ok:
            return self.guard.sql
        return self.plan.sql

    @property
    def ran_ok(self) -> bool:
        return bool(self.exec and self.exec.ok)

    @property
    def rejected_reason(self) -> str | None:
        if self.guard and not self.guard.ok:
            return self.guard.feedback()
        if self.exec and not self.exec.ok:
            return self.exec.error
        return None

    def brief(self) -> str:
        state = "OK" if self.ran_ok else ("被护栏拒绝" if self.guard and not self.guard.ok else "执行失败")
        rows = self.exec.row_count if self.exec else 0
        return f"#{self.index} {state} rows={rows} sql={self.sql[:90]}"


@dataclass
class CandidateSet:
    candidates: list[Candidate] = field(default_factory=list)
    winner: Candidate | None = None
    votes: dict[str, int] = field(default_factory=dict)
    agreement: float = 0.0

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.candidates if c.ran_ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.candidates),
            "ok": self.ok_count,
            "agreement": round(self.agreement, 3),
            "votes": self.votes,
            "winner_index": self.winner.index if self.winner else None,
            "candidates": [
                {
                    "index": c.index,
                    "sql": c.sql,
                    "ok": c.ran_ok,
                    "rows": c.exec.row_count if c.exec else 0,
                    "error": c.rejected_reason,
                    "confidence": c.plan.confidence,
                }
                for c in self.candidates
            ],
        }


def build_candidate(index: int, plan: SQLPlan, executor: SQLExecutor, *, allowed_tables: set[str] | None,
                    default_limit: int, seed: int | None = None) -> Candidate:
    cand = Candidate(index=index, plan=plan, seed=seed)
    if not plan.sql.strip():
        cand.guard = GuardResult(ok=False, errors=["模型没有给出 sql 字段"])
        return cand
    cand.guard = validate_sql(plan.sql, default_limit=default_limit, allowed_tables=allowed_tables)
    if cand.guard.ok:
        cand.exec = executor.run(cand.guard.sql)
    return cand


def majority_vote(candidates: Sequence[Candidate]) -> tuple[Candidate | None, dict[str, int], float]:
    """按结果集指纹投票；只有成功执行的候选参与。"""
    successful = [c for c in candidates if c.ran_ok]
    if not successful:
        return None, {}, 0.0
    signatures = Counter(c.exec.result_signature() for c in successful)
    votes = {sig[:60] + ("..." if len(sig) > 60 else ""): n for sig, n in signatures.most_common()}
    best_sig, best_count = signatures.most_common(1)[0]
    tied = [c for c in successful if c.exec.result_signature() == best_sig]
    # 平票：先看模型自评置信度，再看执行耗时（稳定、可解释的 tie-break）
    tied.sort(key=lambda c: (-c.plan.confidence, c.exec.latency_ms))
    agreement = best_count / len(successful)
    return tied[0], votes, agreement


def run_candidates(plans: Sequence[tuple[int, SQLPlan, int | None]], executor: SQLExecutor, *,
                   allowed_tables: set[str] | None, default_limit: int) -> CandidateSet:
    cands = [build_candidate(i, plan, executor, allowed_tables=allowed_tables, default_limit=default_limit, seed=seed)
             for i, plan, seed in plans]
    winner, votes, agreement = majority_vote(cands)
    return CandidateSet(candidates=list(cands), winner=winner, votes=votes, agreement=agreement)
