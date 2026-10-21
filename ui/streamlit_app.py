"""Streamlit 前端：提问 -> 看 SQL / 结果表 / 候选投票 / 逐节点 Trace。

启动：streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.core import NL2SQLAgent  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.db.schema import introspect  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.tracing import Tracer  # noqa: E402

st.set_page_config(page_title="NL2SQL 智能查询 Agent", page_icon="🗄️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_agent(model: str, n_candidates: int, rerank: bool, clarify: bool,
              backend: str, cloud_model: str) -> NL2SQLAgent:
    import dataclasses

    if not SETTINGS.db_path.exists():
        seed_database(SETTINGS.db_path)
    settings = dataclasses.replace(
        SETTINGS,
        llm_backend=backend,                      # ollama=本地免费 / openai=百炼计费
        model=model,
        openai_model=cloud_model,
        n_candidates=n_candidates,
    )
    tracer = Tracer(enabled=True)
    llm = LLMClient(settings=settings, tracer=tracer)
    return NL2SQLAgent(settings=settings, llm=llm, tracer=tracer,
                       use_llm_rerank=rerank, use_clarify=clarify, use_summary=True)


@st.cache_data(show_spinner=False)
def get_docs():
    return introspect(SETTINGS.db_path)


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 运行配置")
    # ⚠️ 计费开关：本地 Ollama 免费；云端百炼按 token 计费（单题约 9000 token）。
    # 之前默认跟着 .env 走，导致在界面上随手试几下也在花钱，所以做成显式开关。
    source = st.radio("模型来源", ["🖥️ 本地 Ollama（免费）", "☁️ 云端百炼（计费）"], index=0)
    backend = "openai" if source.startswith("☁️") else "ollama"
    if backend == "openai":
        cloud_model = st.selectbox("云端模型", ["qwen-max", "qwen-plus", "qwen-turbo"], index=0)
        st.warning("云端按 token 计费：单题约 9000 token（≈0.01 元），60 题评测约 0.6 元。"
                   "用完想省钱请切回本地。", icon="💸")
        model = SETTINGS.model
    else:
        cloud_model = SETTINGS.openai_model
        model = st.selectbox("本地模型", ["qwen2.5:7b", "qwen2.5:3b"], index=0)
        st.caption("本地推理：零成本、可离线；CPU 下单题约 1.5 分钟（升级显卡驱动后约快 10 倍）")
    n_candidates = st.slider("自一致性候选数 N", 1, 5, SETTINGS.n_candidates,
                             help="N=1 相当于关掉投票；N 越大越稳但越慢")
    rerank = st.toggle("Schema Linking LLM 重排", value=True)
    clarify = st.toggle("歧义时反问用户（Human-in-the-loop）", value=True)
    budget_s = st.slider("单次提问时间预算（秒）", 20, 600, int(SETTINGS.request_budget_s), step=10,
                         help="超预算就带着已有结果返回；想更省时间就把它调小。"
                              "模型调用本身无法中断，所以颗粒度是「每个阶段之间」。")
    st.caption(f"单次模型调用超时 {SETTINGS.request_timeout:.0f}s ｜ 重试 {SETTINGS.max_retries} 次 ｜ "
               f"寒暄快速通道 {'开' if SETTINGS.fast_path else '关'}")
    if st.button("重建示例库", use_container_width=True):
        seed_database(SETTINGS.db_path)
        get_docs.clear()
        get_agent.clear()
        st.success("示例库已重建")

    st.divider()
    st.caption(f"后端 `{SETTINGS.llm_backend}` · 超时 {SETTINGS.query_timeout_s:.0f}s · "
               f"强制 LIMIT {SETTINGS.default_limit} · 自愈上限 {SETTINGS.max_repair_rounds}")

    with st.expander("📚 库表画像（Schema Linking 的输入）", expanded=False):
        for doc in get_docs():
            st.markdown(f"**{doc.name}** · {doc.row_count} 行")
            st.caption(doc.note)
            for col in doc.columns:
                flag = "🔑" if col.pk else ("🔗" if col.fk_table else "")
                samples = "、".join(col.samples[:4])
                st.text(f"  {flag} {col.name} {col.type} {samples}")

agent = get_agent(model, n_candidates, rerank, clarify, backend, cloud_model)

# ---------------------------------------------------------------- 主区域
st.title("🗄️ 自然语言转 SQL 智能查询 Agent")
st.caption("Schema Linking → N 候选自一致性投票 → 执行自愈 → AST 安全护栏 → 全链路 Trace")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_clarify" not in st.session_state:
    st.session_state.pending_clarify = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            payload = msg["payload"]
            st.markdown(payload.get("answer") or "")
            _cols = st.columns([3, 2])
            with _cols[0]:
                st.markdown("**生成的 SQL**")
                st.code(payload.get("sql") or "(无)", language="sql")
            with _cols[1]:
                cand = payload.get("candidates") or {}
                st.metric("投票一致度", f"{cand.get('agreement', 0):.0%}" if cand else "—")
                st.metric("结果行数", payload.get("row_count", 0))
                st.metric("耗时", f"{payload.get('latency_ms', 0) / 1000:.1f}s")
                _trace = payload.get("trace") or {}
                # 把「这次花了多少」显式展示出来 —— 云端按 token 计费，心里要有数
                _tokens = _trace.get("total_tokens", 0)
                _cost = _trace.get("cost", 0)
                st.metric("本次消耗", f"{_tokens} token" + (f"（≈¥{_cost:.4f}）" if _cost else "（本地免费）"))
            rows = payload.get("rows") or []
            columns = payload.get("columns") or []
            if rows:
                st.dataframe([dict(zip(columns, r)) for r in rows], use_container_width=True, height=260)
            with st.expander("🔍 候选与投票明细"):
                for c in cand.get("candidates", []):
                    st.markdown(f"`#{c['index']}` 执行={'成功' if c['ok'] else '失败'} 行数={c['rows']}")
                    st.code(c["sql"], language="sql")
                    if c.get("error"):
                        st.caption(f"失败原因：{c['error']}")
            with st.expander("🧭 调用链 Trace（每个节点的耗时与 token）"):
                st.text(payload.get("trace_text") or "")
                st.json(payload.get("trace") or {})
            if payload.get("rejected"):
                with st.expander("🛡️ 被护栏拦下 / 执行失败的候选"):
                    for r in payload["rejected"]:
                        st.markdown(f"`#{r['index']}` {r['reason']}")
                        st.code(r["sql"], language="sql")

prompt = st.chat_input("用中文问一个业务问题，例如：华东大区2024年每个月的成交额是多少？")
if prompt:
    question = prompt
    if st.session_state.pending_clarify:
        question = f"{st.session_state.pending_clarify['question']}（补充说明：{prompt}）"
        st.session_state.pending_clarify = None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 逐阶段流式显示：以前是"等全部算完才出画面"，看起来像卡死；
        # 现在每过一个阶段就写一行（含已用秒数），并且受总时间预算约束。
        # 想中断：用页面右上角的「停止」——因为每个阶段之间都会 yield，所以停止很快生效。
        STAGE_LABEL = {
            "start": "开始",
            "fast_path": "寒暄/非数据问题，直接回复（不调模型）",
            "guard_input": "输入检查",
            "link_schema": "召回相关表（Schema Linking）",
            "check_ambiguity": "判断问题是否有歧义",
            "clarified": "已收到你的补充说明",
            "generate": "生成候选 SQL",
            "execute": "护栏校验 + 执行 + 投票",
            "repair": "执行失败，自愈重写",
            "synthesize": "汇总结论",
            "aborted": "提前结束",
        }
        result = None
        with st.status("Agent 正在工作…", expanded=True) as status:
            t_start = time.time()
            for event in agent.iter_run(question, budget_s=budget_s):
                stage = event.get("stage", "")
                if stage == "done":
                    result = event["result"] or {}
                    break
                elapsed = time.time() - t_start
                line = f"`{elapsed:5.1f}s`  {STAGE_LABEL.get(stage, stage)}"
                extra = ""
                if stage == "link_schema":
                    extra = " → " + "、".join(event.get("tables", [])[:6])
                elif stage == "generate":
                    extra = f" → {len(event.get('candidates') or [])} 条"
                elif stage == "execute":
                    extra = f" → 一致度 {event.get('agreement')}，{event.get('row_count')} 行"
                elif stage == "aborted":
                    extra = f" → {event.get('reason')}"
                status.write(line + extra)
            used = time.time() - t_start
            state = "complete"
            if result and (result.get("error") or "").find("预算") >= 0:
                state = "error"
            status.update(label=f"完成，用时 {used:.1f} 秒（预算 {budget_s:.0f} 秒）", state=state)

        result = result or {}
        if result.get("needs_clarification") and result.get("clarify_question"):
            st.warning(f"问题有歧义，需要你补充：**{result['clarify_question']}**")
            st.session_state.pending_clarify = {"question": result["question"]}
        if result.get("error"):
            st.info(f"提醒：{result['error']}")
        payload = {
            "answer": result.get("answer", ""),
            "sql": result.get("sql", ""),
            "rows": result.get("rows", []),
            "columns": result.get("columns", []),
            "row_count": result.get("row_count", 0),
            "latency_ms": result.get("latency_ms", 0),
            "candidates": result.get("candidates", {}),
            "rejected": result.get("rejected", []),
            "trace": result.get("trace", {}),
            "trace_text": result.get("trace_text", ""),
        }
        st.session_state.messages.append({"role": "assistant", "payload": payload})
        st.rerun()
