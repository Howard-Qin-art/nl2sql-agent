# -*- coding: utf-8 -*-
"""长短期记忆。

分两层，解决的问题不一样：

**短期记忆（会话内）** —— 让多轮能接上："华东上月成交额" → "那华东呢"。
    存的是最近几轮的问答（问题 / SQL / 结论），拼进生成 Prompt，供指代消解。
    实现：按 session_id 落一个 JSON 文件，读的时候截断到最近 N 轮。

**长期记忆（跨会话）** —— 记住用户的偏好与口径，不必每轮重复交代。
    比如"成交额只算已完成订单""默认看最近 30 天""金额单位是万元"。
    实现：SQLite 一张表 + 关键词/字素重叠打分召回（不引向量库，数据量小）。
    recall 出来的条目拼进 Prompt 的【已知偏好】段。

**存储可切换**：默认落本地（会话写 JSON 文件、偏好写 SQLite）；
把 `NL2SQL_MEMORY_BACKEND=redis` 就换成 Redis —— 会话用 LIST + TTL（过期自动清），
偏好用 HASH + 命中计数。Redis 的优势是多实例共享会话、能设过期、内存快；
单机自用则本地更省事（不需要额外服务）。

设计上刻意保持"可读可删"：记忆是给人看的文本，不是隐向量，用户能 list / forget。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSION_DIR = ROOT / "data" / "sessions"
DEFAULT_MEMORY_DB = ROOT / "data" / "memory.db"

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    """中文按单字+二字组、英文数字按词切 —— 与检索层保持一致的口径。"""
    low = (text or "").lower()
    chars = _CJK.findall(low)
    return _ASCII.findall(low) + chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


# ---------------------------------------------------------------- 短期记忆
@dataclass
class Turn:
    question: str
    answer: str = ""
    sql: str = ""
    tables: list[str] | None = None
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "sql": self.sql,
                "tables": self.tables or [], "ts": self.ts or time.time()}

    def as_messages(self) -> list[dict[str, str]]:
        """转成 prompt 里用的问答对：问题一行、结论一行（不塞整段 SQL，太占 token）。"""
        out = [{"role": "user", "content": self.question}]
        if self.answer:
            out.append({"role": "assistant", "content": self.answer})
        return out


class SessionStore:
    """按 session_id 存的短期记忆（最近 N 轮问答）。"""

    def __init__(self, session_id: str, *, root: Path | None = None, max_turns: int = 4):
        self.session_id = session_id or "default"
        self.dir = Path(root or DEFAULT_SESSION_DIR)
        self.max_turns = max_turns
        self.path = self.dir / f"{self.session_id}.json"

    # ---------- 读写 ----------
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"id": self.session_id, "created_at": time.time(), "turns": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 文件被手工改坏时不要让主流程崩掉，当成空会话
            return {"id": self.session_id, "created_at": time.time(), "turns": []}

    def _save(self, data: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data["turns"] = data.get("turns", [])[-self.max_turns:]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 对外 ----------
    def turns(self) -> list[Turn]:
        raw = self._load().get("turns", [])
        return [Turn(**{k: t.get(k, "") if k != "tables" else t.get("tables") or []
                        for k in ("question", "answer", "sql", "tables", "ts")}) for t in raw]

    def history(self) -> list[dict[str, str]]:
        """最近几轮展开成 messages，直接喂给 Agent 的 history 参数。"""
        msgs: list[dict[str, str]] = []
        for t in self.turns():
            msgs.extend(t.as_messages())
        return msgs

    def append(self, question: str, answer: str = "", sql: str = "",
               tables: Sequence[str] | None = None) -> None:
        data = self._load()
        data.setdefault("turns", []).append(
            Turn(question=question, answer=answer, sql=sql,
                 tables=list(tables or []), ts=time.time()).to_dict())
        self._save(data)

    def clear(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------- 长期记忆
class LongTermMemory:
    """跨会话的事实/偏好（SQLite 一张表，纯文本 + 关键词召回）。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DEFAULT_MEMORY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,          -- preference / metric / entity
                    text TEXT NOT NULL,          -- 人可读的一句话
                    source TEXT DEFAULT 'user',  -- user / clarification / import
                    created_at REAL NOT NULL,
                    hits INTEGER DEFAULT 0
                )
            """)

    def remember(self, text: str, kind: str = "preference", source: str = "user") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("记忆内容不能为空")
        fid = uuid.uuid4().hex[:10]
        with self._conn() as conn:
            conn.execute("INSERT INTO facts (id, kind, text, source, created_at) VALUES (?,?,?,?,?)",
                         (fid, kind, text, source, time.time()))
        return fid

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, kind, text, source, created_at, hits FROM facts "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def forget(self, fact_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return cur.rowcount > 0

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[dict[str, Any]]:
        """按「查询词与记忆文本的 token 重叠率」排序；重叠太低的不返回，避免瞎塞上下文。"""
        q = set(_tokens(query))
        if not q:
            return []
        rows = self.list(limit=200)
        scored = []
        for row in rows:
            toks = set(_tokens(row["text"]))
            if not toks:
                continue
            overlap = len(q & toks) / len(q)
            if overlap >= min_score:
                scored.append((overlap, row))
        scored.sort(key=lambda x: (-x[0], -x[1]["created_at"]))
        picked = [row for _, row in scored[:top_k]]
        if picked:  # 命中计数，方便以后统计"哪些记忆真被用到"
            ids = [(row["id"],) for row in picked]
            with self._conn() as conn:
                conn.executemany("UPDATE facts SET hits = hits + 1 WHERE id = ?", ids)
            for row in picked:      # 返回值也同步 +1，避免调用方看到旧值
                row["hits"] = int(row.get("hits") or 0) + 1
        return picked


# ---------------------------------------------------------------- Redis 实现
def _redis_client(url: str | None = None):
    """惰性导入 redis 客户端：没装就给出可执行的提示，而不是抛 ImportError 堆栈。"""
    try:
        import redis  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "用了 Redis 记忆后端，但当前环境没有 redis 客户端：先 `pip install redis`；"
            "本机还需要一个 Redis 服务（例如 `docker run -d -p 6379:6379 redis:7-alpine`）"
        ) from exc
    return redis.Redis.from_url(url or "redis://127.0.0.1:6379/0", decode_responses=True)


class RedisSessionStore:
    """短期记忆放 Redis：LIST 存最近几轮，LTRIM 截断，EXPIRE 自动过期。"""

    def __init__(self, session_id: str, *, client=None, url: str | None = None,
                 max_turns: int = 4, ttl_s: int = 7 * 24 * 3600, prefix: str = "nl2sql:session"):
        self.session_id = session_id or "default"
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self.key = f"{prefix}:{self.session_id}"
        self.client = client or _redis_client(url)

    def _decode(self, raw: str) -> Turn:
        data = json.loads(raw)
        return Turn(question=data.get("question", ""), answer=data.get("answer", ""),
                    sql=data.get("sql", ""), tables=data.get("tables") or [],
                    ts=float(data.get("ts") or 0))

    def turns(self) -> list[Turn]:
        raw = self.client.lrange(self.key, -self.max_turns, -1) or []
        return [self._decode(r) for r in raw]

    def history(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        for t in self.turns():
            msgs.extend(t.as_messages())
        return msgs

    def append(self, question: str, answer: str = "", sql: str = "",
               tables: Sequence[str] | None = None) -> None:
        payload = json.dumps(Turn(question=question, answer=answer, sql=sql,
                                  tables=list(tables or []), ts=time.time()).to_dict(),
                             ensure_ascii=False)
        pipe = self.client.pipeline()
        pipe.rpush(self.key, payload)
        pipe.ltrim(self.key, -self.max_turns, -1)    # 只留最近 N 轮
        pipe.expire(self.key, self.ttl_s)            # 过期自动清，不用手工回收
        pipe.execute()

    def clear(self) -> None:
        self.client.delete(self.key)


class RedisLongTermMemory:
    """长期记忆放 Redis：HASH 存事实，另一个 HASH 记命中次数。"""

    def __init__(self, *, client=None, url: str | None = None, prefix: str = "nl2sql:memory"):
        self.prefix = prefix
        self.key = f"{prefix}:facts"
        self.hits_key = f"{prefix}:hits"
        self.client = client or _redis_client(url)

    def remember(self, text: str, kind: str = "preference", source: str = "user") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("记忆内容不能为空")
        fid = uuid.uuid4().hex[:10]
        payload = json.dumps({"kind": kind, "text": text, "source": source,
                              "created_at": time.time()}, ensure_ascii=False)
        self.client.hset(self.key, fid, payload)
        return fid

    def _all(self) -> list[dict[str, Any]]:
        raw = self.client.hgetall(self.key) or {}
        hits = self.client.hgetall(self.hits_key) or {}
        rows = []
        for fid, payload in raw.items():
            data = json.loads(payload)
            rows.append({"id": fid, "kind": data.get("kind", ""), "text": data.get("text", ""),
                         "source": data.get("source", ""), "created_at": data.get("created_at", 0),
                         "hits": int(hits.get(fid, 0))})
        rows.sort(key=lambda r: -r["created_at"])
        return rows

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._all()[:limit]

    def forget(self, fact_id: str) -> bool:
        removed = self.client.hdel(self.key, fact_id)
        self.client.hdel(self.hits_key, fact_id)
        return bool(removed)

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[dict[str, Any]]:
        q = set(_tokens(query))
        if not q:
            return []
        scored = []
        for row in self._all():
            toks = set(_tokens(row["text"]))
            if not toks:
                continue
            overlap = len(q & toks) / len(q)
            if overlap >= min_score:
                scored.append((overlap, row))
        scored.sort(key=lambda x: (-x[0], -x[1]["created_at"]))
        picked = [row for _, row in scored[:top_k]]
        if picked:
            pipe = self.client.pipeline()
            for row in picked:
                pipe.hincrby(self.hits_key, row["id"], 1)
                row["hits"] = int(row.get("hits") or 0) + 1
            pipe.execute()
        return picked


# ---------------------------------------------------------------- 工厂（按配置选后端）
def make_session_store(session_id: str, *, settings: Any = None, root: Path | None = None,
                       max_turns: int = 4):
    """按 NL2SQL_MEMORY_BACKEND 选实现：local（JSON 文件）/ redis。"""
    s = settings
    backend = getattr(s, "memory_backend", "local") if s is not None else "local"
    if backend == "redis":
        return RedisSessionStore(session_id, url=getattr(s, "redis_url", None),
                                 max_turns=max_turns,
                                 ttl_s=int(getattr(s, "session_ttl_s", 7 * 24 * 3600)))
    return SessionStore(session_id, root=root, max_turns=max_turns)


def make_long_term_memory(*, settings: Any = None, db_path: Path | None = None):
    """长期记忆同样按后端选：local（SQLite）/ redis。"""
    s = settings
    backend = getattr(s, "memory_backend", "local") if s is not None else "local"
    if backend == "redis":
        return RedisLongTermMemory(url=getattr(s, "redis_url", None))
    return LongTermMemory(db_path)


# ---------------------------------------------------------------- Prompt 注入
def build_memory_block(question: str, *, long_term: Any = None,
                       session: Any = None, top_k: int = 3) -> str:
    """把「历史对话」和「已知偏好」拼成一段 Prompt 片段；都没有就返回空串。"""
    parts: list[str] = []

    if session is not None:
        turns = session.turns()
        if turns:
            lines = ["【历史对话（仅供理解指代，不要照抄）】"]
            for t in turns:
                lines.append(f"问：{t.question}")
                if t.answer:
                    lines.append(f"答：{t.answer}")
            parts.append("\n".join(lines))

    if long_term is not None:
        facts = long_term.recall(question, top_k=top_k)
        if facts:
            lines = ["【已知偏好与口径（用户之前说过，优先遵守）】"]
            lines.extend(f"- {f['text']}" for f in facts)
            parts.append("\n".join(lines))

    return "\n\n".join(parts)
