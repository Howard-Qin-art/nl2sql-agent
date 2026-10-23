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


# ---------------------------------------------------------------- Prompt 注入
def build_memory_block(question: str, *, long_term: LongTermMemory | None = None,
                       session: SessionStore | None = None, top_k: int = 3) -> str:
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
