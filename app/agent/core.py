"""Agent 主流程（框架无关的核心实现）。

流水线：
    输入检查 -> Schema Linking -> 歧义检查(-interrupt->澄清) -> N 候选生成
    -> 护栏校验 -> 只读执行 -> 自一致性投票
        ├─ 有结果 -> 结果解释 -> 结束
        └─ 全失败 -> 执行自愈(报错原文回注) -> 重试（有上限）-> 结束/放弃

同一套方法被两个引擎复用：
    - `core.NL2SQLAgent.run()`：顺序循环（CLI / 评测 / 单元测试；不依赖 LangGraph）
    - `graph.build_graph(agent)`：LangGraph StateGraph（条件边 / Checkpointer / interrupt）
这样「框架挂了也能跑」，也方便对两条路径做一致性测试。
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from app.candidates import Candidate, SQLPlan, CandidateSet, majority_vote, build_candidate
from app.config import SETTINGS, Settings
from app.db.executor import SQLExecutor
from app.db.schema import TableDoc, introspect, schema_prompt
from app.guardrails import check_question, validate_sql
from app.llm import LLMClient, LLMError
from app.prompts import (
    build_answer_messages,
    build_clarify_messages,
    build_generation_messages,
    build_repair_messages,
)
from app.schema_linker import SchemaLinker
from app.tracing import Tracer

ClarifyHandler = Callable[[str], str]

# 寒暄/非数据问题快速通道：这类输入不该花 3~5 分钟走完整链路（6 次模型调用）。
_CHITCHAT_PATTERNS = [
    re.compile(r"^\s*(你好|您好|hi|hello|hey|嗨|哈喽|在吗|在么|早上好|中午好|下午好|晚上好|晚安)\s*[!！。.~～?？]*\s*$", re.I),
    re.compile(r"^\s*(谢谢|多谢|感谢|thanks|thank\s*you|thx)\s*[!！。.~～]*\s*$", re.I),
    re.compile(r"^\s*(你是谁|你叫什么|你能做什么|你能干什么|你会什么|帮助|help|怎么用|如何使用)\s*[?？!！。]*\s*$", re.I),
]
_FAST_PATH_REPLY = (
    "你好 👋 我是数据查询助手：你用一句中文描述想看的业务数据，我会把它翻成 SQL 去查库并给结论。\n\n"
    "试试这些（都基于内置的电商订单库）：\n"
    "· 华东大区 2024 年每个月的成交额是多少？\n"
    "· 各渠道的客单价从高到低排一下\n"
    "· 每个大类里销量最高的商品是哪个？\n"
    "· 退款原因各占多少比例？"
)


def _is_chitchat(question: str) -> bool:
    text = (question or "").strip()
    if not text or len(text) > 20:
        return False
    return any(p.match(text) for p in _CHITCHAT_PATTERNS)


# 与 AgentState 中带 `Annotated[..., operator.add]` reducer 的字段保持一致（追加语义）
APPEND_KEYS = {"notes", "attempts", "repair_history", "clarifications", "history"}


@dataclass
class AgentResult:
    question: str
    ok: bool = False
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    answer: str = ""
    explanation: str = ""
    linked_tables: list[str] = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    repair_rounds: int = 0
    repaired: bool = False
    rejected: list[dict] = field(default_factory=list)
    needs_clarification: bool = False
    clarify_question: str = ""
    confidence: float = 0.0
    error: str | None = None
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)
    trace_text: str = ""
    state: dict = field(default_factory=dict)

    def to_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_rows:
            data.pop("rows", None)
            data.pop("state", None)
        return data


class NL2SQLAgent:
    def __init__(self, *, settings: Settings | None = None, llm: LLMClient | None = None,
                 docs: Sequence[TableDoc] | None = None, tracer: Tracer | None = None,
                 use_llm_rerank: bool | None = None, use_clarify: bool = True, use_summary: bool = True):
        self.s = settings or SETTINGS
        self.tracer = tracer or Tracer(enabled=self.s.trace_enabled)
        self.llm = llm or LLMClient(settings=self.s, tracer=self.tracer)
        self.docs = list(docs) if docs is not None else introspect(self.s.db_path)
        self.by_name = {d.name: d for d in self.docs}
        self.allowed_tables = set(self.by_name)
        self.executor = SQLExecutor(self.s.db_path, timeout_s=self.s.query_timeout_s, max_rows=self.s.max_rows)
        self.linker = SchemaLinker(
            self.docs, llm=self.llm, top_k=self.s.top_k_tables,
            embed_model=self.s.embed_model or None,   # 配了 NL2SQL_EMBED_MODEL 才启用向量召回（如 bge-m3）
            embed_cache_dir=self.s.embed_cache_dir,
            num_gpu=self.s.ollama_num_gpu,
            ollama_base=self.s.ollama_base,
            use_llm_rerank=self.s.top_k_tables < len(self.docs) if use_llm_rerank is None else use_llm_rerank,
        )
        self.use_clarify = use_clarify
        self.use_summary = use_summary

    # ================================================================ 各阶段
    def guard_input(self, state: dict) -> dict:
        question = (state.get("question") or "").strip()
        with self.tracer.span("guard_input", question=question[:200]) as sp:
            if not question:
                sp.outputs["ok"] = False
                return {"error": "问题为空", "notes": ["输入为空，直接拒绝"]}
            # 寒暄/非数据问题：两条编排引擎都要走这条快速通道，
            # 否则「你好」会白跑 6 次模型调用（CPU 上要好几分钟）。
            if self.s.fast_path and _is_chitchat(question):
                sp.outputs["fast_path"] = True
                return {"answer": _FAST_PATH_REPLY, "early_exit": True,
                        "notes": ["寒暄/非数据问题，跳过整条链路（0 次模型调用）"],
                        "stats": {"fast_path": True}}
            hints = check_question(question)
            notes = [f"输入检查命中：{h}" for h in hints]
            sp.outputs["hints"] = hints
            return {"notes": notes, "stats": {"input_hints": len(hints)}}

    def link_schema(self, state: dict) -> dict:
        question = state.get("question", "")
        with self.tracer.span("link_schema", question=question[:200]) as sp:
            linked = self.linker.link(question)
            text = schema_prompt(linked.tables)
            sp.outputs["tables"] = linked.table_names
            sp.outputs["value_hits"] = linked.value_hits
            return {
                "linked_tables": linked.table_names,
                "schema_text": text,
                "link_debug": linked.debug,
                "stats": {"linked_tables": len(linked.tables), "value_hits": len(linked.value_hits)},
            }

    def check_ambiguity(self, state: dict) -> dict:
        """让模型判断问题是否有歧义/信息不足；这一步决定要不要打断去问用户。"""
        if not self.use_clarify:
            return {"needs_clarification": False, "confidence": 1.0}
        question = state.get("question", "")
        with self.tracer.span("check_ambiguity", question=question[:200]) as sp:
            try:
                data, _ = self.llm.chat_json(
                    build_clarify_messages(question, state.get("schema_text", "")),
                    temperature=0.0, max_tokens=200,
                )
            except LLMError as exc:
                sp.outputs["fallback"] = str(exc)[:120]
                return {"needs_clarification": False, "confidence": 1.0,
                        "notes": [f"歧义检查失败，跳过澄清：{exc}"]}
            ambiguous = bool(data.get("ambiguous"))
            conf = float(data.get("confidence") or (0.3 if ambiguous else 0.9))
            sp.outputs["ambiguous"] = ambiguous
            sp.outputs["confidence"] = conf
            out: dict[str, Any] = {
                "needs_clarification": ambiguous,
                "clarify_question": str(data.get("question") or "").strip(),
                "confidence": conf,
            }
            if ambiguous:
                out["notes"] = [f"判定为有歧义：{data.get('question', '')[:80]}"]
            return out

    def apply_clarification(self, state: dict, answer: str) -> dict:
        """把用户的澄清拼回问题里，再走一遍生成（这是 HITL 的落地点）。"""
        merged = f"{state.get('question', '')}（补充说明：{answer}）"
        return {
            "question": merged,
            "needs_clarification": False,
            "clarified": True,
            "clarifications": [{"question": state.get("clarify_question", ""), "answer": answer}],
            "notes": [f"用户澄清：{answer[:80]}"],
        }

    def generate(self, state: dict, *, temperature: float | None = None, n: int | None = None,
                 retry_hint: str | None = None, deadline: float | None = None) -> dict:
        question = state.get("question", "")
        n = n or self.s.n_candidates
        temperature = self.s.sample_temperature if temperature is None else temperature
        messages = build_generation_messages(question, state.get("schema_text", ""),
                                            history=state.get("history"), retry_hint=retry_hint,
                                            memory_text=state.get("memory_text"))
        plans: list[dict] = []
        with self.tracer.span("generate", n=n, temperature=temperature) as sp:
            for i in range(n):
                # 时间预算控制：已经攒到候选了就不再继续采样（宁可少投票，也别让用户干等）
                if deadline is not None and plans and time.perf_counter() > deadline:
                    sp.outputs["budget_cut"] = f"生成到第 {i} 条时超过预算，停止采样"
                    break
                seed = 1000 + i * 17 + state.get("repair_rounds", 0) * 101
                try:
                    data, _res = self.llm.chat_json(messages, model_cls=SQLPlan, temperature=temperature,
                                                    max_tokens=self.s.max_tokens, seed=seed)
                    plan = SQLPlan(**data)
                    plans.append({"index": i, "seed": seed, "plan": plan, "error": None})
                except Exception as exc:  # 单条候选失败不影响其它候选
                    plans.append({"index": i, "seed": seed, "plan": None,
                                  "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            sp.outputs["plans"] = [p["plan"].sql[:120] if p["plan"] else p["error"] for p in plans]
        return {"candidate_plans": plans,
                "attempts": [{"stage": "generate", "round": state.get("repair_rounds", 0),
                              "n_ok": sum(1 for p in plans if p["plan"])}]}

    def execute_candidates(self, state: dict) -> dict:
        """护栏校验 + 执行 + 自一致性投票。"""
        plans = state.get("candidate_plans") or []
        with self.tracer.span("execute_candidates", n=len(plans)) as sp:
            cands: list[Candidate] = []
            for item in plans:
                plan: SQLPlan | None = item.get("plan")
                if plan is None:
                    continue
                cands.append(build_candidate(item["index"], plan, self.executor,
                                             allowed_tables=self.allowed_tables,
                                             default_limit=self.s.default_limit, seed=item.get("seed")))
            winner, votes, agreement = majority_vote(cands)
            rejected = [{"index": c.index, "sql": c.sql, "reason": c.rejected_reason}
                        for c in cands if not c.ran_ok]
            cset = CandidateSet(candidates=cands, winner=winner, votes=votes, agreement=agreement)
            sp.outputs["ok"] = bool(winner)
            sp.outputs["agreement"] = round(agreement, 2)
            sp.outputs["rejected"] = len(rejected)
            return {
                "candidates": cset.to_dict(),
                "chosen_sql": winner.sql if winner else "",
                "result": _result_payload(winner.exec) if winner else {},
                "confidence": winner.plan.confidence if winner else 0.0,
                "rejected": rejected,
                "stats": {"candidate_ok": cset.ok_count, "agreement": round(agreement, 3)},
            }

    def repair(self, state: dict) -> dict:
        """执行自愈：把「失败的 SQL + 数据库报错原文」回注，要求最小改动。"""
        rejected = state.get("rejected") or []
        # 优先用执行报错（信息量最大），没有就用护栏拒绝原因
        pick = next((r for r in rejected if r.get("reason") and "解析" not in r["reason"]), None) or (rejected[0] if rejected else None)
        if pick is None:
            return {"error": "没有可修复的候选", "notes": ["自愈阶段没有拿到失败信息"]}
        round_no = state.get("repair_rounds", 0) + 1
        with self.tracer.span("repair", round=round_no, error=str(pick.get("reason"))[:160]) as sp:
            messages = build_repair_messages(state.get("question", ""), state.get("schema_text", ""),
                                             pick.get("sql", ""), str(pick.get("reason", "")),
                                             history=state.get("history"))
            try:
                data, _ = self.llm.chat_json(messages, model_cls=SQLPlan, temperature=0.2,
                                             max_tokens=self.s.max_tokens, seed=2000 + round_no)
                plan = SQLPlan(**data)
            except Exception as exc:
                sp.outputs["failed"] = str(exc)[:160]
                return {"repair_rounds": round_no,
                        "repair_history": [{"round": round_no, "ok": False, "error": str(exc)[:200]}],
                        "notes": [f"自愈第 {round_no} 轮调用失败"]}

            cand = build_candidate(99, plan, self.executor, allowed_tables=self.allowed_tables,
                                   default_limit=self.s.default_limit, seed=2000 + round_no)
            sp.outputs["sql"] = cand.sql[:140]
            sp.outputs["ok"] = cand.ran_ok
            out: dict[str, Any] = {
                "repair_rounds": round_no,
                "repair_history": [{
                    "round": round_no, "failed_sql": pick.get("sql", ""), "error": str(pick.get("reason", ""))[:200],
                    "new_sql": cand.sql, "ok": cand.ran_ok,
                }],
                "attempts": [{"stage": "repair", "round": round_no, "ok": cand.ran_ok}],
            }
            if cand.ran_ok:
                prev = state.get("candidates") or {}
                out["chosen_sql"] = cand.sql
                out["result"] = _result_payload(cand.exec)
                out["confidence"] = cand.plan.confidence
                out["candidates"] = {**prev, "repaired_sql": cand.sql, "repair_round": round_no}
                out["notes"] = [f"自愈成功（第 {round_no} 轮）：{cand.sql[:80]}"]
            else:
                out["notes"] = [f"自愈第 {round_no} 轮仍失败：{cand.rejected_reason}"]
                out["rejected"] = [{"index": 99, "sql": cand.sql, "reason": cand.rejected_reason}]
            return out

    def synthesize(self, state: dict) -> dict:
        result = state.get("result") or {}
        if not result:
            return {"answer": "没有查询到结果。", "error": state.get("error") or "全部候选执行失败"}
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        row_count = result.get("row_count", 0)
        explanation = (state.get("candidates") or {}).get("winner_explanation", "")
        answer = ""
        if self.use_summary and row_count:
            with self.tracer.span("synthesize") as sp:
                try:
                    res = self.llm.chat(build_answer_messages(state.get("question", ""), state.get("chosen_sql", ""),
                                                              columns, rows, row_count=row_count),
                                        temperature=0.2, max_tokens=200)
                    answer = (res.text or "").strip()
                    sp.outputs["answer"] = answer[:120]
                except LLMError as exc:
                    sp.outputs["fallback"] = str(exc)[:120]
        if not answer:
            answer = f"查询返回 {row_count} 行。" + (f"列：{', '.join(columns)}" if columns else "")
        return {"answer": answer, "explanation": explanation}

    # ================================================================ 编排（顺序版）
    def _prices(self) -> tuple[float, float]:
        return (self.s.price_per_1k_prompt, self.s.price_per_1k_completion)

    def iter_run(self, question: str, *, history: list[dict[str, str]] | None = None,
                 memory_text: str = "",
                 clarify_handler: ClarifyHandler | None = None, tracer: Tracer | None = None,
                 cancel_event: Any = None, budget_s: float | None = None):
        """逐阶段产出事件，供 API 的 SSE 流式推送给前端画进度。
        最后一个事件的 stage == "done"，其 result 即 AgentResult 的字典形态。

        打断/超时（两层）：
            - `budget_s`：本次提问的总时间预算，每进入一个阶段前检查，超了就带着已有结果返回；
            - `cancel_event`：threading.Event，界面点「停止」时置位，在阶段之间立刻退出
              （模型调用本身没法中断，所以粒度是"阶段之间"，这也是为什么不把它做在模型层）。
        """
        tracer = tracer or self.tracer
        t0 = time.perf_counter()
        budget = self.s.request_budget_s if budget_s is None else budget_s
        deadline = t0 + budget if budget and budget > 0 else float("inf")
        state: dict[str, Any] = {"question": question, "history": list(history or []),
                                 "memory_text": memory_text or "",
                                 "repair_rounds": 0, "notes": [], "attempts": []}
        yield {"stage": "start", "question": question, "budget_s": budget}

        def aborted() -> str | None:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                return "用户手动停止"
            if time.perf_counter() > deadline:
                return f"超过时间预算 {budget:.0f}s"
            return None

        def finish(reason: str | None = None) -> dict:
            if reason:
                state["error"] = reason
                _merge(state, {"notes": [f"提前结束：{reason}"]})
                if not state.get("answer"):
                    state["answer"] = (
                        f"（本次没有跑完：{reason}。已完成的部分："
                        f"{'已生成 SQL 但未执行' if state.get('chosen_sql') else '尚未生成 SQL'}）"
                    )
            return _to_result(state, tracer, t0, self._prices()).to_dict()

        # ---- 0) 寒暄/非数据问题快速通道：不打模型，秒回 ----
        if self.s.fast_path and _is_chitchat(question):
            state["answer"] = _FAST_PATH_REPLY
            yield {"stage": "fast_path", "reason": "寒暄/非数据问题，跳过整条链路"}
            yield {"stage": "done", "result": finish()}
            return

        _merge(state, self.guard_input(state))
        yield {"stage": "guard_input", "ok": not state.get("error"),
               "notes": state.get("notes", [])[-1:]}
        if state.get("error"):
            yield {"stage": "done", "result": finish()}
            return

        reason = aborted()
        if reason:
            yield {"stage": "aborted", "reason": reason}
            yield {"stage": "done", "result": finish(reason)}
            return

        _merge(state, self.link_schema(state))
        yield {"stage": "link_schema", "tables": state.get("linked_tables", []),
               "value_hits": (state.get("link_debug") or {}).get("value_tables", [])}

        # ---- 歧义澄清（Human-in-the-loop）----
        _merge(state, self.check_ambiguity(state))
        yield {"stage": "check_ambiguity", "needs_clarification": bool(state.get("needs_clarification")),
               "question": state.get("clarify_question", "")}
        if state.get("needs_clarification") and clarify_handler is not None and state.get("clarify_question"):
            answer = clarify_handler(state["clarify_question"])
            if answer:
                _merge(state, self.apply_clarification(state, answer))
                _merge(state, self.link_schema(state))
                _merge(state, self.check_ambiguity(state))
                state["needs_clarification"] = False
                yield {"stage": "clarified", "answer": answer}

        reason = aborted()
        if reason:
            yield {"stage": "aborted", "reason": reason}
            yield {"stage": "done", "result": finish(reason)}
            return

        # ---- 生成 -> 执行 -> 自愈循环（候选按剩余预算动态收敛）----
        _merge(state, self.generate(state, deadline=deadline))
        yield {"stage": "generate", "candidates": [
            (p["plan"].sql if p.get("plan") else p.get("error")) for p in state.get("candidate_plans", [])]}

        _merge(state, self.execute_candidates(state))
        yield {"stage": "execute", "ok": bool(state.get("result")),
               "agreement": (state.get("candidates") or {}).get("agreement"),
               "row_count": (state.get("result") or {}).get("row_count", 0)}

        while not state.get("result") and state.get("repair_rounds", 0) < self.s.max_repair_rounds:
            if aborted():
                break
            _merge(state, self.repair(state))
            yield {"stage": "repair", "round": state.get("repair_rounds"), "ok": bool(state.get("result"))}
            if state.get("result"):
                break

        reason = None if state.get("result") else aborted()
        _merge(state, self.synthesize(state))
        yield {"stage": "synthesize", "answer": state.get("answer", "")}
        yield {"stage": "done", "result": finish(reason)}

    def run(self, question: str, *, history: list[dict[str, str]] | None = None,
            memory_text: str = "",
            clarify_handler: ClarifyHandler | None = None, tracer: Tracer | None = None,
            cancel_event: Any = None, budget_s: float | None = None) -> AgentResult:
        """跑完整轮并返回 AgentResult（内部复用 iter_run，保证流式与一次性调用行为一致）。"""
        final: dict[str, Any] | None = None
        for event in self.iter_run(question, history=history, memory_text=memory_text,
                                   clarify_handler=clarify_handler, tracer=tracer,
                                   cancel_event=cancel_event, budget_s=budget_s):
            if event.get("stage") == "done":
                final = event["result"]
        if final is None:  # pragma: no cover - 理论上不可达
            raise RuntimeError("iter_run 未产出 done 事件")
        return AgentResult(**final)


def _merge(state: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    """顺序引擎里的状态合并：追加语义的 key 与 LangGraph 的 reducer 保持一致，
    这样两条编排路径的 notes/attempts 数量相同，可以做一致性测试。"""
    for key, value in partial.items():
        if key in APPEND_KEYS and isinstance(value, list):
            state[key] = list(state.get(key) or []) + value
        else:
            state[key] = value
    return state


def _result_payload(exec_result) -> dict:
    if exec_result is None:
        return {}
    return {
        "columns": exec_result.columns,
        "rows": exec_result.rows,
        "row_count": exec_result.row_count,
        "truncated": exec_result.truncated,
        "latency_ms": round(exec_result.latency_ms, 1),
    }


def _to_result(state: dict, tracer: Tracer, t0: float, prices: tuple[float, float] = (0.0, 0.0)) -> AgentResult:
    result = state.get("result") or {}
    return AgentResult(
        question=state.get("question", ""),
        ok=bool(result),
        sql=state.get("chosen_sql", ""),
        columns=result.get("columns", []),
        rows=result.get("rows", []),
        row_count=result.get("row_count", 0),
        answer=state.get("answer", ""),
        explanation=state.get("explanation", ""),
        linked_tables=state.get("linked_tables", []),
        candidates=state.get("candidates", {}),
        repair_rounds=state.get("repair_rounds", 0),
        repaired=any(h.get("ok") for h in (state.get("repair_history") or [])),
        rejected=state.get("rejected", []),
        needs_clarification=bool(state.get("needs_clarification")),
        clarify_question=state.get("clarify_question", ""),
        confidence=float(state.get("confidence") or 0.0),
        error=state.get("error"),
        latency_ms=(time.perf_counter() - t0) * 1000,
        trace=tracer.summary(*prices),
        trace_text=tracer.pretty(),
        state={k: v for k, v in state.items() if k != "history"},
    )


def build_agent(**kwargs: Any) -> NL2SQLAgent:
    return NL2SQLAgent(**kwargs)
