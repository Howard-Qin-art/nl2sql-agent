"""Schema Linking：从整库 schema 里挑出与问题相关的表/列。

库里只有 9 张表，但我按「上百张表也能用」的接口来设计，因为面试官一定会问
「你表只有 9 张，实际生产几百张怎么办」。四路召回 + RRF 融合：

    1) 词法召回 BM25（表级 / 列级两套索引）—— 中文用「字 + 二元组」切分，
       因为本机没装 jieba，而字级二元组对中文术语（华东、成交额、客单价）召回够用；
    2) 值命中：把问题里的字面值（华东 / app / completed）匹配到列的抽样值，
       再**沿外键反向扩展到事实表** —— 「华东」只出现在 region 表里，
       但真正要查的是 orders，这一步是纯词法召回救不回来的；
    3) 向量召回（可选）：装了 embedding 模型（如 ollama nomic-embed-text）就自动启用，
       没装则记录跳过原因，不阻断链路 —— 接口预留、可插拔；
    4) LLM 重排：把融合后的候选表连同业务说明丢给模型，让它挑最终 top_k，
       失败自动回退到 RRF 顺序（LLM 不是关键路径，挂了不影响可用性）。
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.db.schema import TableDoc

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文：单字 + 相邻二字组；英文/数字：整词。"""
    text = (text or "").lower()
    words = ASCII_WORD.findall(text)
    chars = CJK.findall(text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return words + chars + bigrams


class BM25:
    """标准 BM25（k1=1.5, b=0.75），纯 Python 实现，无外部依赖。"""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in corpus]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        df: Counter = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str, idx: int) -> float:
        q = Counter(tokenize(query))
        doc, dl = self.docs[idx], self.lengths[idx]
        total = 0.0
        for term, qf in q.items():
            tf = doc.get(term)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)) * min(qf, 3)
        return total

    def rank(self, query: str, top_n: int | None = None) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n] if top_n else scored


def rrf(rankings: Iterable[Sequence[Any]], k: int = 60) -> dict[Any, float]:
    """Reciprocal Rank Fusion：多路召回按排名融合，天然不需要跨路归一化分数。"""
    fused: dict[Any, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)


@dataclass
class LinkResult:
    tables: list[TableDoc]
    scores: dict[str, float] = field(default_factory=dict)
    value_hits: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]


class SchemaLinker:
    def __init__(self, docs: Sequence[TableDoc], *, llm: Any = None, top_k: int = 6,
                 embed_model: str | None = None, ollama_base: str = "http://127.0.0.1:11434",
                 use_llm_rerank: bool = True, candidate_pool: int = 12,
                 embed_cache_dir: "Path | None" = None, num_gpu: int = 0):
        self.docs = list(docs)
        self.by_name = {d.name: d for d in self.docs}
        self.llm = llm
        self.top_k = top_k
        self.use_llm_rerank = use_llm_rerank
        self.candidate_pool = candidate_pool
        self.embed_model = embed_model
        self.ollama_base = ollama_base
        self.embed_cache_dir = Path(embed_cache_dir) if embed_cache_dir else None
        # GPU 与 Ollama 的 CUDA 内核不兼容时 embedding 会直接崩，所以要能强制 CPU
        self.num_gpu = num_gpu
        self._embed_cache: dict[str, list[float]] = {}
        self.embed_skip_reason: str | None = None
        self.embed_stats: dict[str, Any] = {}

        # 表级 / 列级 BM25 索引
        self.table_bm25 = BM25([d.search_text() for d in self.docs])
        self.column_keys: list[tuple[str, str]] = []
        column_texts: list[str] = []
        for d in self.docs:
            for c in d.columns:
                self.column_keys.append((d.name, c.name))
                column_texts.append(f"{d.name} {c.name} {c.note} " + " ".join(c.samples[:8]))
        self.column_bm25 = BM25(column_texts)

        # 值 -> (表, 列) 反向索引，用于字面值命中
        self.value_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for d in self.docs:
            for c in d.columns:
                if c.is_categorical or (c.samples and c.distinct <= 25):
                    for sample in c.samples:
                        if sample and not _is_numeric(sample):
                            self.value_index[sample.lower()].append((d.name, c.name))

        # 外键反向扩展：(表, 列) -> 引用它的 (表, 列)  —— 列级精确匹配
        self.fk_reverse: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        # 表级反向扩展：表 -> 引用它的 (表, 列)
        # 为什么需要表级：值命中往往落在维表的**描述列**上（region.region_name='华东'），
        # 而事实表的外键指向的是维表的**主键列**（orders.region_id -> region.region_id），
        # 只在列级匹配就永远扩不到事实表，召回会退化。
        self.fk_reverse_table: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for d in self.docs:
            for c in d.columns:
                if c.fk_table:
                    self.fk_reverse[(c.fk_table, c.fk_column)].append((d.name, c.name))
                    self.fk_reverse_table[c.fk_table].append((d.name, c.name))

    # ---------------------------------------------------------- 各路召回
    def _lexical(self, question: str) -> tuple[list[str], list[tuple[str, str]]]:
        table_rank = [self.docs[i].name for i, _ in self.table_bm25.rank(question)]
        col_rank = [(self.column_keys[i][0], self.column_keys[i][1]) for i, _ in self.column_bm25.rank(question)]
        # 列得分回流到表：一张表命中多列 => 更相关
        col_table_scores: Counter = Counter()
        for pos, (tname, _cname) in enumerate(col_rank[:40]):
            col_table_scores[tname] += 1.0 / (1 + pos)
        table_rank = list(dict.fromkeys(
            [t for t in table_rank] + [t for t, _ in col_table_scores.most_common()]
        ))
        return table_rank, col_rank

    def _value_hits(self, question: str) -> tuple[list[str], list[str]]:
        """字面值命中列 -> 该表 + 引用该表的所有表（表级外键扩展）。"""
        q = question.lower()
        hit_tables: list[str] = []
        hits: list[str] = []
        for value, locations in self.value_index.items():
            if len(value) < 2 or value not in q:
                continue
            for tname, cname in locations:
                hits.append(f"{value} -> {tname}.{cname}")
                hit_tables.append(tname)
                expanded = list(self.fk_reverse.get((tname, cname), []))
                for ref_table, ref_col in self.fk_reverse_table.get(tname, []):
                    if (ref_table, ref_col) not in expanded:
                        expanded.append((ref_table, ref_col))
                for ref_table, ref_col in expanded:
                    hits.append(f"{value} -> {tname}.{cname} -FK-> {ref_table}.{ref_col}")
                    hit_tables.append(ref_table)
        return list(dict.fromkeys(hit_tables)), hits

    def _embedding_rankings(self, question: str) -> tuple[list[str], list[str]]:
        """可选向量召回（bge-m3 等）。返回 (表级排名, 由列级排名归并出的表排名)。

        工程要点：
            1. 批量调用 `/api/embed`（一次把 9 张表 + N 列发过去），比逐条快一个量级；
            2. **必须带 options.num_gpu=0**：本机 GPU 驱动与 Ollama 的 CUDA 内核不兼容，
               不带这个参数 llama-server 会直接崩（qwen 与 bge-m3 都一样）；
            3. 向量落磁盘缓存（按 模型+文本 哈希），第二次开始 0 成本；
            4. 任何异常都只记录原因、不阻断链路 —— 向量召回是增强项，不是关键路径。
        """
        if not self.embed_model:
            self.embed_skip_reason = "未配置 embedding 模型（NL2SQL_EMBED_MODEL），跳过向量召回"
            return [], []

        table_texts = [d.search_text() for d in self.docs]
        col_texts = [f"{t}.{c} {note}" for (t, c), note in
                     zip(self.column_keys, [self._col_note(t, c) for t, c in self.column_keys])] \
            if self.column_keys else []
        try:
            t0 = time.perf_counter()
            table_vecs = self._embed_cached(table_texts)
            col_vecs = self._embed_cached(col_texts) if col_texts else []
            qvec = self._embed_cached([question])[0]
            embed_ms = (time.perf_counter() - t0) * 1000

            table_scored = sorted(
                ((d.name, _cosine(qvec, vec)) for d, vec in zip(self.docs, table_vecs)),
                key=lambda x: -x[1])
            table_rank = [name for name, score in table_scored if score > 0]

            col_table_scores: Counter = Counter()
            if col_vecs:
                col_scored = sorted(
                    ((self.column_keys[i], _cosine(qvec, vec)) for i, vec in enumerate(col_vecs)),
                    key=lambda x: -x[1])
                for pos, ((tname, _cname), _score) in enumerate(col_scored[:40]):
                    col_table_scores[tname] += 1.0 / (1 + pos)
            col_table_rank = [t for t, _ in col_table_scores.most_common()]

            self.embed_stats = {
                "used": True,
                "model": self.embed_model,
                "dim": len(qvec),
                "texts": len(table_texts) + len(col_texts),
                "latency_ms": round(embed_ms, 1),
                "cache_size": len(self._embed_cache),
                "top_tables": [f"{n}:{s:.3f}" for n, s in table_scored[:4]],
            }
            return table_rank, col_table_rank
        except Exception as exc:  # embedding 不是关键路径
            if os.getenv("NL2SQL_DEBUG_EMBED", "").strip() in {"1", "true", "yes"}:
                import traceback

                traceback.print_exc()
            self.embed_skip_reason = f"向量召回不可用：{type(exc).__name__}: {str(exc)[:160]}"
            self.embed_stats = {"used": False, "reason": self.embed_skip_reason}
            return [], []

    def _col_note(self, table: str, column: str) -> str:
        for c in self.by_name[table].columns:
            if c.name == column:
                samples = " ".join(c.samples[:6])
                return f"{c.note} {samples}"
        return ""

    # ---------- 向量：磁盘缓存 + 批量调用 ----------
    def _embed_cached(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_embed_cache()
        missing = [t for t in texts if _embed_key(self.embed_model, t) not in self._embed_cache]
        if missing:
            vectors = _embed_batch(self.ollama_base, self.embed_model, missing, num_gpu=self.num_gpu)
            for text, vec in zip(missing, vectors):
                self._embed_cache[_embed_key(self.embed_model, text)] = vec
            self._save_embed_cache()
        return [self._embed_cache[_embed_key(self.embed_model, t)] for t in texts]

    def _cache_file(self) -> "Path | None":
        if not self.embed_cache_dir:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.embed_model or "embed")
        return self.embed_cache_dir / f"{safe}.json"

    def _load_embed_cache(self) -> None:
        path = self._cache_file()
        if not path or self._embed_cache or not path.exists():
            return
        try:
            import json

            self._embed_cache = {k: list(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        except Exception:
            self._embed_cache = {}

    def _save_embed_cache(self) -> None:
        path = self._cache_file()
        if not path:
            return
        try:
            import json

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._embed_cache), encoding="utf-8")
        except Exception:
            pass

    # ---------------------------------------------------------- 主流程
    def link(self, question: str) -> LinkResult:
        table_rank, col_rank = self._lexical(question)
        value_tables, value_hits = self._value_hits(question)
        embed_rank, embed_col_rank = self._embedding_rankings(question)

        col_table_rank = list(dict.fromkeys(t for t, _ in col_rank))
        fused = rrf([r for r in (table_rank, col_table_rank, value_tables, embed_rank, embed_col_rank) if r])
        # 值命中的表额外加权：字面值匹配是强信号
        for name in value_tables:
            fused[name] = fused.get(name, 0.0) + 0.05

        ordered = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        pool = [name for name, _ in ordered[: self.candidate_pool]] or [d.name for d in self.docs[: self.candidate_pool]]
        pool = [n for n in pool if n in self.by_name]

        rerank_debug: dict[str, Any] = {"used": False}
        final = pool[: self.top_k]
        if self.use_llm_rerank and self.llm is not None and len(pool) > 1:
            try:
                picked, reason = self._llm_rerank(question, pool)
                if picked:
                    valid = [p for p in picked if p in pool]
                    rest = [p for p in pool if p not in valid]
                    final = (valid + rest)[: self.top_k]
                    rerank_debug = {"used": True, "reason": reason, "picked": valid}
            except Exception as exc:
                rerank_debug = {"used": False, "error": f"{type(exc).__name__}: {exc}"}

        tables = [self.by_name[n] for n in final]
        # 保证被值命中/外键扩展涉及的表也在内（否则生成阶段看不到必要的维表）
        for extra in value_tables[:2] + [t for t in pool if t in value_tables]:
            if extra in self.by_name and extra not in [t.name for t in tables] and len(tables) < self.top_k + 2:
                tables.append(self.by_name[extra])

        return LinkResult(
            tables=tables,
            scores={name: round(score, 5) for name, score in ordered},
            value_hits=value_hits,
            debug={
                "lexical_tables": table_rank[:8],
                "lexical_columns": [f"{t}.{c}" for t, c in col_rank[:8]],
                "value_tables": value_tables,
                "embedding": dict(self.embed_stats) if self.embed_stats else
                {"used": False, "reason": self.embed_skip_reason},
                "embedding_tables": embed_rank[:8],
                "embedding_col_tables": embed_col_rank[:8],
                "pool": pool,
                "llm_rerank": rerank_debug,
            },
        )

    def _llm_rerank(self, question: str, pool: Sequence[str]) -> tuple[list[str], str]:
        from app.prompts import build_rerank_messages

        lines = []
        for name in pool:
            doc = self.by_name[name]
            cols = ", ".join(c.name for c in doc.columns[:10])
            lines.append(f"- {name}（{doc.row_count} 行）: {doc.note or ''} 列: {cols}")
        data, _ = self.llm.chat_json(build_rerank_messages(question, "\n".join(lines)), temperature=0.0, max_tokens=200)
        picked = data.get("tables") or []
        return [str(p).strip() for p in picked], str(data.get("reason", ""))[:200]


# ---------------------------------------------------------------- embedding 小工具
def _embed_key(model: str, text: str) -> str:
    import hashlib

    return hashlib.sha1(f"{model}\u0000{text}".encode("utf-8")).hexdigest()


def _embed_batch(base: str, model: str, texts: Sequence[str], *, num_gpu: int = 0,
                 timeout: float = 300.0) -> list[list[float]]:
    """批量取 embedding：优先 /api/embed（新端点，支持 batch），失败再退到 /api/embeddings 逐条。

    关键：本机 GPU 驱动与 Ollama 的 CUDA 内核不兼容（device kernel image is invalid），
    必须显式带 options.num_gpu=0 才会走 CPU，否则 llama-server 直接崩。
    """
    import requests

    options = {"num_gpu": num_gpu}
    try:
        resp = requests.post(f"{base}/api/embed",
                             json={"model": model, "input": list(texts), "options": options}, timeout=timeout)
        if resp.status_code == 200:
            vectors = resp.json().get("embeddings") or []
            if len(vectors) == len(texts):
                return [list(v) for v in vectors]
        last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"

    # 退路：老端点逐条
    out: list[list[float]] = []
    for text in texts:
        resp = requests.post(f"{base}/api/embeddings",
                             json={"model": model, "prompt": text, "options": options}, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"embedding 失败（新端点：{last_error}；老端点 HTTP {resp.status_code}: {resp.text[:160]}）")
        vec = resp.json().get("embedding")
        if not vec:
            raise RuntimeError("embedding 返回为空（模型是否已 pull？）")
        out.append(list(vec))
    return out


def _embed(base: str, model: str, text: str) -> list[float]:
    """单条兼容接口（保留给外部调用）。"""
    return _embed_batch(base, model, [text])[0]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def _is_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False
