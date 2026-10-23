# -*- coding: utf-8 -*-
"""Redis 记忆后端的测试。

不需要真的 Redis 服务，也不依赖 redis 包：这里实现了一个最小的内存假客户端，
只覆盖我们用到的那几个命令（LIST / HASH / 管道 / 过期）。
真机联调另有一份脚本 `scripts/check_redis_memory.py`。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.memory import (RedisLongTermMemory, RedisSessionStore,  # noqa: E402
                         build_memory_block, make_long_term_memory, make_session_store)


# ---------------------------------------------------------------- 假 Redis
class _FakePipeline:
    def __init__(self, client: "_FakeRedis"):
        self.client = client
        self.ops: list[tuple] = []

    def rpush(self, k, v):
        self.ops.append(("rpush", k, v)); return self

    def ltrim(self, k, s, e):
        self.ops.append(("ltrim", k, s, e)); return self

    def expire(self, k, ttl):
        self.ops.append(("expire", k, ttl)); return self

    def hincrby(self, k, f, n=1):
        self.ops.append(("hincrby", k, f, n)); return self

    def execute(self):
        for op in self.ops:
            getattr(self.client, op[0])(*op[1:])
        self.ops.clear()
        return []


class _FakeRedis:
    """够用的内存实现：list 命令按切片语义、hash 命令按 dict。"""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttl: dict[str, int] = {}

    def pipeline(self):
        return _FakePipeline(self)

    # ---- list ----
    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lrange(self, key, start, end):
        lst = self.lists.get(key, [])
        n = len(lst)
        s = max(0, n + start) if start < 0 else start
        e = n - 1 if end < 0 and -end > n else (n + end if end < 0 else min(end, n - 1))
        return lst[s:e + 1] if e + 1 > s else []

    def ltrim(self, key, start, end):
        self.lists[key] = self.lrange(key, start, end)
        return True

    def expire(self, key, ttl):
        self.ttl[key] = ttl
        return True

    def delete(self, key):
        return 1 if self.lists.pop(key, None) is not None else 0

    # ---- hash ----
    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hdel(self, key, field):
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0

    def hincrby(self, key, field, n=1):
        bucket = self.hashes.setdefault(key, {})
        bucket[field] = str(int(bucket.get(field, 0)) + n)
        return int(bucket[field])


@pytest.fixture
def fake():
    return _FakeRedis()


# ---------------------------------------------------------------- 会话记忆
def test_redis_session_append_and_history(fake):
    s = RedisSessionStore("r1", client=fake, max_turns=4, ttl_s=3600)
    assert s.history() == []
    s.append("一共有多少个客户", answer="200 个客户", sql="SELECT ...", tables=["customer"])
    s.append("那上个月呢", answer="上月新增 12 个")
    hist = s.history()
    assert len(hist) == 4
    assert hist[0]["content"] == "一共有多少个客户"
    assert hist[2]["content"] == "那上个月呢"
    # TTL 被刷新（会话不会永久堆在 Redis 里）
    assert fake.ttl[s.key] == 3600


def test_redis_session_trims_to_max_turns(fake):
    s = RedisSessionStore("r2", client=fake, max_turns=2)
    for i in range(5):
        s.append(f"问题{i}", answer=f"答案{i}")
    assert len(s.turns()) == 2
    assert s.turns()[-1].question == "问题4"


def test_redis_session_clear(fake):
    s = RedisSessionStore("r3", client=fake)
    s.append("q", answer="a")
    s.clear()
    assert s.history() == []


# ---------------------------------------------------------------- 长期记忆
def test_redis_long_term_remember_list_forget(fake):
    mem = RedisLongTermMemory(client=fake)
    fid = mem.remember("成交额只算已完成订单", kind="metric")
    rows = mem.list()
    assert len(rows) == 1 and rows[0]["text"] == "成交额只算已完成订单"
    assert rows[0]["kind"] == "metric"
    assert mem.forget(fid) is True
    assert mem.list() == []
    assert mem.forget("不存在") is False


def test_redis_long_term_recall_and_hit_counter(fake):
    mem = RedisLongTermMemory(client=fake)
    mem.remember("成交额只算已完成订单")
    mem.remember("金额单位统一用万元")
    mem.remember("华东大区的数据排除测试账号")

    hits = mem.recall("华东大区成交额多少")
    assert hits and any("成交额" in h["text"] for h in hits)
    assert hits[0]["hits"] >= 1                     # 命中计数写回
    assert mem.recall("今天天气怎么样") == []          # 不相关不注入


# ---------------------------------------------------------------- 按配置选后端
def test_factories_pick_backend():
    local = dataclasses.replace(SETTINGS, memory_backend="local")
    redis_cfg = dataclasses.replace(SETTINGS, memory_backend="redis",
                                    redis_url="redis://127.0.0.1:6379/0")
    assert type(make_session_store("x", settings=local)).__name__ == "SessionStore"
    assert type(make_long_term_memory(settings=local)).__name__ == "LongTermMemory"
    # 没装 redis 客户端时应当给出可执行报错；装了就该真的换成 Redis 实现
    try:
        import redis  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError) as exc:
            make_session_store("x", settings=redis_cfg)
        assert "redis" in str(exc.value).lower()
    else:
        # 连接是惰性的，构造时不会去连服务器
        assert type(make_session_store("x", settings=redis_cfg)).__name__ == "RedisSessionStore"
        assert type(make_long_term_memory(settings=redis_cfg)).__name__ == "RedisLongTermMemory"


def test_memory_block_works_with_redis_stores(fake):
    session = RedisSessionStore("r4", client=fake)
    session.append("华东有多少客户", answer="120 个")
    mem = RedisLongTermMemory(client=fake)
    mem.remember("成交额只算已完成订单")
    block = build_memory_block("华东成交额", long_term=mem, session=session)
    assert "【历史对话" in block and "华东有多少客户" in block
    assert "【已知偏好" in block and "成交额只算已完成订单" in block
