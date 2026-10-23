# -*- coding: utf-8 -*-
"""Redis 记忆后端的真机联调脚本。

前置：
    1) 有一个 Redis 服务（任选其一）
       - Docker： docker run -d --name nl2sql-redis -p 6379:6379 redis:7-alpine
       - WSL：    sudo apt install redis-server && sudo service redis-server start
       - Memurai / Redis for Windows
    2) 装 Python 客户端： pip install redis

用法：
    python scripts/check_redis_memory.py
    python scripts/check_redis_memory.py --url redis://127.0.0.1:6379/0

它会做四件事并逐条打印结果：连通性 → 会话记忆（写入/读回/截断/TTL）→ 长期记忆（增删/召回/命中计数）
→ 清理测试键。任何一步失败都会给出可执行的下一步建议。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  -> {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="redis://127.0.0.1:6379/0")
    ap.add_argument("--keep", action="store_true", help="跑完不清理测试键（排查用）")
    args = ap.parse_args()

    try:
        import redis  # noqa: F401
    except ImportError:
        print("缺少 redis 客户端：先执行  pip install redis")
        return 2

    from app.memory import RedisLongTermMemory, RedisSessionStore, build_memory_block

    # ---- 1) 连通性 ----
    try:
        client = redis.Redis.from_url(args.url, decode_responses=True)
        pong = client.ping()
        check("连接 Redis", bool(pong), args.url)
    except Exception as exc:  # noqa: BLE001
        print(f"连不上 Redis：{type(exc).__name__}: {exc}")
        print("  先起一个服务，任选其一：")
        print("    docker run -d --name nl2sql-redis -p 6379:6379 redis:7-alpine")
        print("    wsl sudo apt install -y redis-server && wsl sudo service redis-server start")
        return 2

    run_id = f"check-{int(time.time())}"
    session_key_prefix = f"nl2sql:check:{run_id}"
    print(f"\n测试键前缀：{session_key_prefix}*")

    # ---- 2) 会话记忆 ----
    try:
        store = RedisSessionStore(run_id, client=client, max_turns=2, ttl_s=120,
                                  prefix=session_key_prefix)
        store.append("一共有多少个客户", answer="200 个客户")
        store.append("那已取消的订单呢", answer="201 笔")
        store.append("再问一个", answer="第三个答案")     # 应该被 LTRIM 挤掉最早那条
        turns = store.turns()
        check("写入并读回会话记忆", len(turns) == 2, f"{len(turns)} 轮")
        check("按 max_turns 截断", turns[-1].question == "再问一个" and
              all(t.question != "一共有多少个客户" for t in turns),
              str([t.question for t in turns]))
        ttl = client.ttl(store.key)
        check("设置了 TTL", 0 < ttl <= 120, f"ttl={ttl}s")
        hist = store.history()
        check("history() 转成问答对", len(hist) == 4, f"{len(hist)} 条消息")
    except Exception as exc:  # noqa: BLE001
        check("会话记忆", False, f"{type(exc).__name__}: {exc}")

    # ---- 3) 长期记忆 ----
    try:
        mem = RedisLongTermMemory(client=client, prefix=f"{session_key_prefix}:mem")
        fid1 = mem.remember("成交额只算已完成订单", kind="metric")
        mem.remember("华东大区的数据排除测试账号")
        mem.remember("金额单位统一用万元")
        rows = mem.list()
        check("写入并列出长期记忆", len(rows) == 3, f"{len(rows)} 条")
        hits = mem.recall("华东大区成交额多少")
        check("按相关度召回", bool(hits) and any("成交额" in h["text"] for h in hits),
              " / ".join(h["text"][:12] for h in hits))
        check("不相关的不召回", mem.recall("今天天气怎么样") == [])
        check("命中计数自增", (mem.recall("成交额")[0]["hits"] >= 2) if mem.recall("成交额") else False)
        check("删除记忆", mem.forget(fid1) is True and len(mem.list()) == 2)
    except Exception as exc:  # noqa: BLE001
        check("长期记忆", False, f"{type(exc).__name__}: {exc}")

    # ---- 4) 拼 Prompt 片段 ----
    try:
        store2 = RedisSessionStore(run_id + "-p", client=client, prefix=session_key_prefix)
        store2.append("华东有多少客户", answer="120 个")
        mem2 = RedisLongTermMemory(client=client, prefix=f"{session_key_prefix}:mem")
        block = build_memory_block("华东成交额", long_term=mem2, session=store2)
        check("拼出 Prompt 片段", "【历史对话" in block and "【已知偏好" in block,
              f"{len(block)} 字符")
    except Exception as exc:  # noqa: BLE001
        check("Prompt 片段", False, f"{type(exc).__name__}: {exc}")

    # ---- 清理 ----
    if not args.keep:
        for key in client.scan_iter(match=f"{session_key_prefix}*"):
            client.delete(key)
        print("已清理测试键")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
