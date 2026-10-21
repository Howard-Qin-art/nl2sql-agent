"""FastAPI 服务：对外提供 NL2SQL 查询接口（含 SSE 流式进度、歧义澄清回话）。

    uvicorn app.api:app --port 8000
    curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
         -d '{"question":"华东大区2024年每个月的成交额是多少？"}'

接口：
    GET  /health           健康检查（模型/后端/库/候选数）
    GET  /schema           库表画像（Schema Linking 的输入，方便前端展示）
    POST /query            问一次；engine=graph 时若判定有歧义会返回 interrupted=true
    POST /query/clarify    对歧义问题补充信息后继续（LangGraph Command(resume)）
    POST /query/stream     SSE：逐阶段推送 Agent 进度（阶段名 + 中间结果）
    GET  /eval/summary     最近一次评测的指标（跑过 eval/run_eval.py 才有）
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.core import NL2SQLAgent
from app.agent.graph import GraphAgent
from app.config import SETTINGS
from app.llm import LLMClient
from app.tracing import Tracer

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    tracer = Tracer(enabled=SETTINGS.trace_enabled)
    agent = NL2SQLAgent(settings=SETTINGS, llm=LLMClient(settings=SETTINGS, tracer=tracer), tracer=tracer)
    STATE["agent"] = agent
    STATE["graph"] = GraphAgent(agent=agent)
    STATE["docs"] = agent.docs
    yield
    STATE.clear()


app = FastAPI(title="NL2SQL Agent", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------- 请求/响应模型
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="中文业务问题")
    thread_id: str | None = Field(None, description="多轮对话/澄清会话 id，不传则新建")
    engine: str = Field("graph", description="graph=LangGraph 编排；core=顺序循环")


class ClarifyRequest(BaseModel):
    thread_id: str
    answer: str = Field(..., min_length=1, description="用户对澄清问题的补充说明")


class QueryResponse(BaseModel):
    ok: bool = False
    interrupted: bool = False
    thread_id: str | None = None
    clarify_question: str | None = None
    sql: str = ""
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    answer: str = ""
    explanation: str = ""
    linked_tables: list[str] = []
    candidates: dict = {}
    repair_rounds: int = 0
    confidence: float = 0.0
    latency_ms: float = 0.0
    trace: dict = {}
    error: str | None = None


# ---------------------------------------------------------------- 基础接口
@app.get("/health")
def health() -> dict[str, Any]:
    agent: NL2SQLAgent = STATE["agent"]
    # 报「实际在跑的 Agent 的配置」，而不是进程启动时的全局配置（两者可能不同）
    return {
        "status": "ok",
        "llm_backend": agent.s.llm_backend,
        "model": agent.s.model,
        "db": str(agent.s.db_path),
        "tables": len(agent.docs),
        "n_candidates": agent.s.n_candidates,
        "max_repair_rounds": agent.s.max_repair_rounds,
        "query_timeout_s": agent.s.query_timeout_s,
        "llm_rerank": agent.linker.use_llm_rerank,
        "clarify_enabled": agent.use_clarify,
        "embed_model": agent.s.embed_model or None,
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    docs = STATE["docs"]
    return {
        "tables": [
            {
                "name": d.name,
                "row_count": d.row_count,
                "note": d.note,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.type,
                        "pk": c.pk,
                        "fk": f"{c.fk_table}.{c.fk_column}" if c.fk_table else None,
                        "null_ratio": round(c.null_ratio, 3),
                        "distinct": c.distinct,
                        "samples": c.samples[:6],
                        "note": c.note,
                    }
                    for c in d.columns
                ],
            }
            for d in docs
        ]
    }


# ---------------------------------------------------------------- 查询
def _run_graph(req: QueryRequest) -> QueryResponse:
    graph: GraphAgent = STATE["graph"]
    out = graph.run(req.question, thread_id=req.thread_id)
    if out.get("interrupted"):
        clarify = out.get("clarify") or {}
        return QueryResponse(ok=False, interrupted=True, thread_id=out["thread_id"],
                             clarify_question=str(clarify.get("question", "")), trace={"stage": "clarify"})
    return QueryResponse(**{k: v for k, v in out.items() if k in QueryResponse.model_fields})


def _run_core(req: QueryRequest) -> QueryResponse:
    agent: NL2SQLAgent = STATE["agent"]
    res = agent.run(req.question).to_dict()
    res["thread_id"] = req.thread_id
    return QueryResponse(**{k: v for k, v in res.items() if k in QueryResponse.model_fields})


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if STATE.get("agent") is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    return _run_graph(req) if req.engine == "graph" else _run_core(req)


@app.post("/query/clarify", response_model=QueryResponse)
def clarify(req: ClarifyRequest) -> QueryResponse:
    graph: GraphAgent = STATE["graph"]
    try:
        out = graph.resume(req.answer, thread_id=req.thread_id)
    except Exception as exc:  # 会话不存在 / 已经恢复过
        raise HTTPException(status_code=400, detail=f"恢复会话失败：{exc}") from exc
    return QueryResponse(**{k: v for k, v in out.items() if k in QueryResponse.model_fields})


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """SSE 流式：每个 Agent 阶段完成就推一条事件，前端可以画时间线。"""
    agent: NL2SQLAgent = STATE["agent"]

    def event_source() -> Iterator[str]:
        try:
            for event in agent.iter_run(req.question):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:  # 流里也不要把连接搞崩
            yield f"data: {json.dumps({'stage': 'error', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- 评测结果
@app.get("/eval/summary")
def eval_summary() -> dict[str, Any]:
    results = Path(SETTINGS.trace_dir).parent / "eval" / "results"
    files = sorted(results.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="还没有评测结果，先跑 python -m eval.run_eval")
    latest = files[0]
    data = json.loads(latest.read_text(encoding="utf-8"))
    data["_file"] = latest.name
    return data
