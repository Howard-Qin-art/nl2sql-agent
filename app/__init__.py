"""NL2SQL Agent —— 自然语言转 SQL 的智能查询 Agent。

包结构：
    app.config        配置（环境变量 / .env）
    app.llm           LLM 双后端抽象（Ollama 原生 API / OpenAI 兼容 API）+ FakeLLM
    app.tracing       全链路 trace（每个节点的输入输出、耗时、token、成本）
    app.guardrails    sqlglot AST 安全护栏（只读白名单、单语句、强制 LIMIT）
    app.db.seed       确定性示例库生成（SQLite）
    app.db.schema     库表自省 -> 表/列文档（供 Schema Linking 与 Prompt 使用）
    app.db.executor   只读执行器（超时中断、行数上限、错误原文回传）
    app.schema_linker Schema Linking：BM25 + 向量 + RRF + LLM 重排
    app.candidates    N 候选生成 + 结果集规范化 + 自一致性投票
    app.agent.*       Agent 编排（LangGraph StateGraph / 框架无关核心）
    app.api           FastAPI 服务
"""

__version__ = "0.1.0"
