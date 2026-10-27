# -*- coding: utf-8 -*-
"""把公开的 Text-to-SQL 数据集下到 E:\dddddd\nl2sql数据。

用法：
    python scripts/fetch_dataset.py bird_mini      # BIRD mini_dev（阿里云 OSS，约 800MB）
    python scripts/fetch_dataset.py spider_hf      # HF: HAL-9001/spider-databases（约 210MB）
    python scripts/fetch_dataset.py cspider        # CSpider 中文问题（GitHub，需能访问 raw）

特点：分块下载 + 断点续传 + 失败重试，避免大文件一次性拉断。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

BASE = Path(r"E:\dddddd\nl2sql数据")
UA = {"User-Agent": "Mozilla/5.0 (nl2sql-dataset-fetch)"}

SOURCES = {
    "bird_mini": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip",
        "dir": "bird_mini_dev",
        "file": "minidev.zip",
    },
    "spider_hf": {
        "url": "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip",
        "dir": "spider",
        "file": "spider_data.zip",
    },
}


def download(url: str, dest: Path, retries: int = 5) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    print(f"  [retry {attempt}] HTTP {r.status_code}")
                    time.sleep(2 * attempt)
                    continue
                total = r.headers.get("content-length")
                total = have + int(total) if total else None
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                with dest.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        if total and have % (20 * 1024 * 1024) < 1024 * 1024:
                            print(f"    {have / 1e6:7.1f} / {total / 1e6:7.1f} MB "
                                  f"({have / total * 100:5.1f}%)", flush=True)
            size = dest.stat().st_size
            if total and size < total * 0.99:
                print(f"  [retry {attempt}] 不完整 {size}/{total}，续传")
                continue
            print(f"  [OK] {dest.name}  {size / 1e6:.1f} MB")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [retry {attempt}] {type(exc).__name__}: {exc}")
            time.sleep(2 * attempt)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(SOURCES) + ["cspider"])
    ap.add_argument("--url", default=None, help="覆盖下载地址（CSpider 用）")
    args = ap.parse_args()

    if args.name == "cspider":
        # CSpider 的中文问题在 GitHub；先试几个可能的路径
        cands = [
            "https://raw.githubusercontent.com/taolusi/chisp/master/data/dev.json",
            "https://raw.githubusercontent.com/taolusi/chisp/main/data/dev.json",
            "https://raw.githubusercontent.com/taolusi/chisp/master/database/tables.json",
        ]
        for u in cands:
            dest = BASE / "cspider" / u.rsplit("/", 1)[-1].replace("?", "_")
            print(f"尝试 {u}")
            if download(u, dest, retries=1):
                print(f"  -> {dest}")
        return 0

    spec = SOURCES[args.name]
    url = args.url or spec["url"]
    dest = BASE / spec["dir"] / spec["file"]
    print(f"下载 {args.name} -> {dest}")
    ok = download(url, dest)
    if ok:
        print(f"完成：{dest}  {dest.stat().st_size / 1e6:.1f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
