"""全局配置：环境变量优先，其次 `.env` 文件，最后是内置默认值。

设计原则：所有可调参数（候选数、自愈轮数、超时、护栏 LIMIT、模型名）都集中在这里，
方便跑评测时用环境变量做消融实验（ablation），例如：

    NL2SQL_N_CANDIDATES=1 python -m eval.run_eval     # 关掉自一致性投票
    NL2SQL_MAX_REPAIR_ROUNDS=0 python -m eval.run_eval  # 关掉执行自愈
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载器（不引入额外依赖）；已存在的环境变量不会被覆盖。"""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---------- LLM 后端 ----------
    llm_backend: str = "ollama"                  # ollama | openai | fake
    model: str = "qwen2.5:7b"
    ollama_base: str = "http://127.0.0.1:11434"
    # 本机 RTX 3060 + 驱动 546.30 与 Ollama 0.40.3 的 CUDA 内核不兼容，GPU 会崩；
    # 默认强制 CPU（num_gpu=0）。升级显卡驱动后可设 NL2SQL_OLLAMA_NUM_GPU=-1 用回 GPU。
    ollama_num_gpu: int = 0
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "30m"
    # ---------- 云端后端（NL2SQL_LLM_BACKEND=openai 时生效） ----------
    # 阿里云百炼（DashScope OpenAI 兼容模式）：base 形如
    #   https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    # 注意 key 与地域绑定；模型名如 qwen-plus / qwen-max。
    openai_base: str = "https://api.openai.com/v1"
    openai_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # 百炼文档：temperature 范围 [0,2) 且"不建议取值为 0"，云端给个下限
    min_temperature: float = 0.01

    # 生成温度：候选采样用高温（自一致性），确定性节点用 0
    temperature: float = 0.0
    sample_temperature: float = 0.7
    max_tokens: int = 512
    # 单次模型调用的超时与重试：原来是 300s × 3 次重试 = 最坏 15 分钟才失败，
    # CPU 推理下这体验太差，收紧到 120s / 1 次重试。
    request_timeout: float = 120.0
    max_retries: int = 1

    # ---------- 时间预算与打断（体验相关） ----------
    # 一次提问的**总时间预算**：超了就不再往下跑，带着已有结果返回并说明原因。
    # 这是"能强制打断"的具体实现：每个阶段开始前检查剩余预算。
    request_budget_s: float = 120.0
    # 寒暄/非数据问题快速通道：不打模型直接回（"你好"不该等 3 分钟）
    fast_path: bool = True

    # ---------- Agent 行为 ----------
    n_candidates: int = 3            # 自一致性采样条数
    max_repair_rounds: int = 2       # 执行报错后的自愈轮数上限
    top_k_tables: int = 6            # Schema Linking 保留的表数
    confidence_threshold: float = 0.55   # 低于此值触发澄清（Human-in-the-loop）

    # ---------- Schema Linking 的向量召回（可选） ----------
    # 留空 = 关闭（只用 BM25 + 值命中 + LLM 重排）。填模型名即启用，本机已 pull 好 bge-m3：
    #     NL2SQL_EMBED_MODEL=bge-m3
    # 本机 GPU 驱动与 Ollama 的 CUDA 内核不兼容，embedding 同样必须走 CPU（num_gpu=0），
    # 否则 llama-server 直接崩；CPU 下 bge-m3 一次 9 条文本约 0.3s，且有磁盘缓存。
    embed_model: str = ""
    embed_cache_dir: Path = field(default_factory=lambda: ROOT / "data" / "embed_cache")

    # ---------- 数据库 ----------
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "nl2sql.db")
    query_timeout_s: float = 10.0
    max_rows: int = 500              # 执行器最多返回多少行
    default_limit: int = 100         # 护栏强制补的 LIMIT

    # ---------- 记忆 ----------
    # local = 会话写 data/sessions/*.json + 偏好写 data/memory.db；redis = 两边都走 Redis
    memory_backend: str = "local"
    redis_url: str = "redis://127.0.0.1:6379/0"
    session_ttl_s: int = 7 * 24 * 3600        # Redis 下会话记忆的过期时间

    # ---------- Trace / 成本 ----------
    trace_dir: Path = field(default_factory=lambda: ROOT / "traces")
    trace_enabled: bool = True
    # 是否在 trace 里记录模型原始请求/响应（默认关：文件会明显变大，排查问题时再开）
    trace_llm: bool = False
    # 单位：元 / 千 token（云端后端用来估算成本；本地 Ollama 记为 0）
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        s = cls()
        s.llm_backend = _env_str("NL2SQL_LLM_BACKEND", s.llm_backend)
        s.model = _env_str("NL2SQL_MODEL", s.model)
        s.ollama_base = _env_str("NL2SQL_OLLAMA_BASE", s.ollama_base).rstrip("/")
        s.ollama_num_gpu = _env_int("NL2SQL_OLLAMA_NUM_GPU", s.ollama_num_gpu)
        s.ollama_num_ctx = _env_int("NL2SQL_OLLAMA_NUM_CTX", s.ollama_num_ctx)
        s.openai_base = _env_str("NL2SQL_OPENAI_BASE", _env_str("OPENAI_BASE_URL", s.openai_base)).rstrip("/")
        s.openai_key = _env_str("NL2SQL_OPENAI_KEY", _env_str("OPENAI_API_KEY", s.openai_key))
        s.openai_model = _env_str("NL2SQL_OPENAI_MODEL", s.openai_model)
        s.min_temperature = _env_float("NL2SQL_MIN_TEMPERATURE", s.min_temperature)
        s.temperature = _env_float("NL2SQL_TEMPERATURE", s.temperature)
        s.sample_temperature = _env_float("NL2SQL_SAMPLE_TEMPERATURE", s.sample_temperature)
        s.max_tokens = _env_int("NL2SQL_MAX_TOKENS", s.max_tokens)
        s.request_timeout = _env_float("NL2SQL_REQUEST_TIMEOUT", s.request_timeout)
        s.max_retries = _env_int("NL2SQL_MAX_RETRIES", s.max_retries)
        s.request_budget_s = _env_float("NL2SQL_REQUEST_BUDGET", s.request_budget_s)
        s.fast_path = _env_bool("NL2SQL_FAST_PATH", s.fast_path)
        s.n_candidates = _env_int("NL2SQL_N_CANDIDATES", s.n_candidates)
        s.max_repair_rounds = _env_int("NL2SQL_MAX_REPAIR_ROUNDS", s.max_repair_rounds)
        s.top_k_tables = _env_int("NL2SQL_TOP_K_TABLES", s.top_k_tables)
        s.confidence_threshold = _env_float("NL2SQL_CONFIDENCE_THRESHOLD", s.confidence_threshold)
        s.embed_model = _env_str("NL2SQL_EMBED_MODEL", s.embed_model)
        s.embed_cache_dir = Path(_env_str("NL2SQL_EMBED_CACHE", str(s.embed_cache_dir)))
        s.db_path = Path(_env_str("NL2SQL_DB", str(s.db_path)))
        s.query_timeout_s = _env_float("NL2SQL_QUERY_TIMEOUT", s.query_timeout_s)
        s.max_rows = _env_int("NL2SQL_MAX_ROWS", s.max_rows)
        s.default_limit = _env_int("NL2SQL_DEFAULT_LIMIT", s.default_limit)
        s.trace_dir = Path(_env_str("NL2SQL_TRACE_DIR", str(s.trace_dir)))
        s.trace_enabled = _env_bool("NL2SQL_TRACE", s.trace_enabled)
        s.trace_llm = _env_bool("NL2SQL_TRACE_LLM", s.trace_llm)
        s.memory_backend = _env_str("NL2SQL_MEMORY_BACKEND", s.memory_backend).lower()
        s.redis_url = _env_str("NL2SQL_REDIS_URL", s.redis_url)
        s.session_ttl_s = _env_int("NL2SQL_SESSION_TTL", s.session_ttl_s)
        if s.llm_backend == "openai":
            s.price_per_1k_prompt = _env_float("NL2SQL_PRICE_PROMPT", 0.001)
            s.price_per_1k_completion = _env_float("NL2SQL_PRICE_COMPLETION", 0.002)
        return s


SETTINGS = Settings.load()
