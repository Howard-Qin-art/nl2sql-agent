"""确定性示例库生成（SQLite）。

为什么自己造数据：
    - 不能依赖联网下载数据集，评测必须可复现（固定随机种子）；
    - 数据要「有坑」：空值、下架商品、取消订单、退款、类目两级、跨年日期 ——
      这样窗口函数（每月 Top-N）、CTE、多表 Join、NULL 处理才有东西可问，
      也才能把 Agent 的错误暴露出来（而不是所有问题都是简单 SELECT）。
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE region (
    region_id    INTEGER PRIMARY KEY,
    region_name  TEXT    NOT NULL,
    manager      TEXT
);
CREATE TABLE customer (
    customer_id   INTEGER PRIMARY KEY,
    customer_name TEXT    NOT NULL,
    email         TEXT,
    city          TEXT,
    region_id     INTEGER REFERENCES region(region_id),
    signup_date   DATE    NOT NULL,
    level         TEXT    NOT NULL CHECK (level IN ('bronze','silver','gold','platinum')),
    is_active     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE supplier (
    supplier_id   INTEGER PRIMARY KEY,
    supplier_name TEXT    NOT NULL,
    city          TEXT,
    rating        REAL
);
CREATE TABLE category (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT    NOT NULL,
    parent_id     INTEGER REFERENCES category(category_id)
);
CREATE TABLE product (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT    NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES category(category_id),
    supplier_id  INTEGER NOT NULL REFERENCES supplier(supplier_id),
    list_price   REAL    NOT NULL,
    cost         REAL    NOT NULL,
    launched_on  DATE,
    status       TEXT    NOT NULL CHECK (status IN ('on_sale','off_shelf'))
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customer(customer_id),
    order_date   DATE    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('completed','cancelled','pending','refunded')),
    channel      TEXT    NOT NULL CHECK (channel IN ('app','web','offline','mini_program')),
    region_id    INTEGER NOT NULL REFERENCES region(region_id),
    total_amount REAL    NOT NULL DEFAULT 0
);
CREATE TABLE order_item (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES product(product_id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL    NOT NULL,
    discount   REAL    NOT NULL DEFAULT 0
);
CREATE TABLE payment (
    payment_id INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    paid_at    DATETIME,
    method     TEXT NOT NULL CHECK (method IN ('alipay','wechat','card','balance')),
    amount     REAL NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('success','failed','pending'))
);
CREATE TABLE refund (
    refund_id   INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    refund_date DATE    NOT NULL,
    amount      REAL    NOT NULL,
    reason      TEXT    NOT NULL CHECK (reason IN ('quality','wrong_item','late_delivery','no_reason'))
);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_item_order ON order_item(order_id);
CREATE INDEX idx_item_product ON order_item(product_id);
"""

REGIONS = ["华东", "华北", "华南", "西南", "西北", "东北", "华中", "海外"]
MANAGERS = ["张伟", "李娜", "王强", "刘洋", "陈静", "赵磊", "孙倩", "周涛"]
CITIES = ["上海", "北京", "广州", "成都", "西安", "沈阳", "武汉", "杭州", "深圳", "南京", "重庆", "苏州"]
SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜")
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "勇", "军", "杰", "娟", "涛", "明", "超", "秀兰", "霞", "平", "刚"]
CATEGORIES = [
    ("家用电器", ["冰箱", "洗衣机", "空调", "电视"]),
    ("数码产品", ["手机", "笔记本电脑", "平板", "耳机"]),
    ("服饰鞋包", ["男装", "女装", "运动鞋", "箱包"]),
    ("食品饮料", ["休闲零食", "茶饮", "粮油", "生鲜"]),
    ("家居建材", ["家具", "厨具", "灯具", "家纺"]),
    ("美妆个护", ["护肤", "彩妆", "洗护", "口腔护理"]),
]
PRODUCT_WORDS = ["智享", "轻颜", "云舒", "极简", "星爵", "初心", "本色", "悠享", "御风", "清野"]
REASONS = ["quality", "wrong_item", "late_delivery", "no_reason"]


def _rand_name(rng: random.Random) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def _rand_date(rng: random.Random, start: str, end: str) -> str:
    import datetime as dt

    a = dt.date.fromisoformat(start)
    b = dt.date.fromisoformat(end)
    return (a + dt.timedelta(days=rng.randint(0, (b - a).days))).isoformat()


def seed_database(db_path: Path, *, orders: int = 3000, customers: int = 200, products: int = 120,
                  rng_seed: int = 20240101, overwrite: bool = True) -> Path:
    """生成示例库；同样的种子 -> 同样的数据（评测可复现）。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and db_path.exists():
        db_path.unlink()

    rng = random.Random(rng_seed)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # 区域
    conn.executemany(
        "INSERT INTO region(region_id, region_name, manager) VALUES (?,?,?)",
        [(i + 1, name, MANAGERS[i]) for i, name in enumerate(REGIONS)],
    )

    # 客户：少量空 city / email，少量非活跃
    rows = []
    for cid in range(1, customers + 1):
        rows.append(
            (
                cid,
                _rand_name(rng),
                None if rng.random() < 0.05 else f"user{cid:04d}@example.com",
                None if rng.random() < 0.04 else rng.choice(CITIES),
                rng.randint(1, len(REGIONS)),
                _rand_date(rng, "2022-01-01", "2025-06-30"),
                rng.choices(["bronze", "silver", "gold", "platinum"], weights=[5, 3, 1.5, 0.5])[0],
                0 if rng.random() < 0.12 else 1,
            )
        )
    conn.executemany("INSERT INTO customer VALUES (?,?,?,?,?,?,?,?)", rows)

    # 供应商
    rows = [
        (sid, f"{rng.choice(PRODUCT_WORDS)}供应链{sid:02d}", rng.choice(CITIES), round(rng.uniform(3.0, 5.0), 1))
        for sid in range(1, 26)
    ]
    conn.executemany("INSERT INTO supplier VALUES (?,?,?,?)", rows)

    # 类目（两级）
    cat_id, cat_rows = 1, []
    for top_name, subs in CATEGORIES:
        parent = cat_id
        cat_rows.append((cat_id, top_name, None))
        cat_id += 1
        for sub in subs:
            cat_rows.append((cat_id, sub, parent))
            cat_id += 1
    conn.executemany("INSERT INTO category VALUES (?,?,?)", cat_rows)
    leaf_ids = [c[0] for c in cat_rows if c[2] is not None]

    # 商品：部分下架、部分无上市日期
    rows = []
    for pid in range(1, products + 1):
        cost = round(rng.uniform(20, 3000), 2)
        rows.append(
            (
                pid,
                f"{rng.choice(PRODUCT_WORDS)}{rng.randint(1, 999):03d}",
                rng.choice(leaf_ids),
                rng.randint(1, 25),
                round(cost * rng.uniform(1.25, 2.4), 2),
                cost,
                None if rng.random() < 0.06 else _rand_date(rng, "2021-01-01", "2025-06-30"),
                "off_shelf" if rng.random() < 0.15 else "on_sale",
            )
        )
    conn.executemany("INSERT INTO product VALUES (?,?,?,?,?,?,?,?)", rows)

    # 订单 + 明细 + 支付 + 退款
    order_rows, item_rows, pay_rows, refund_rows = [], [], [], []
    item_id = payment_id = refund_id = 1
    for oid in range(1, orders + 1):
        status = rng.choices(["completed", "cancelled", "pending", "refunded"], weights=[80, 6, 9, 5])[0]
        order_date = _rand_date(rng, "2024-01-01", "2025-12-31")
        total = 0.0
        for _ in range(rng.randint(1, 5)):
            qty = rng.randint(1, 6)
            price = round(rng.uniform(20, 3200), 2)
            discount = round(rng.choice([0, 0, 0, 0.05, 0.1, 0.2]), 2)
            total += qty * price * (1 - discount)
            item_rows.append((item_id, oid, rng.randint(1, products), qty, price, discount))
            item_id += 1
        order_rows.append((oid, rng.randint(1, customers), order_date, status, rng.choice(["app", "web", "offline", "mini_program"]),
                           rng.randint(1, len(REGIONS)), round(total, 2)))
        if status in ("completed", "refunded"):
            pay_status = "success"
        elif status == "pending":
            pay_status = rng.choice(["pending", "failed"])
        else:
            pay_status = "failed"
        pay_rows.append((payment_id, oid, f"{order_date} 1{rng.randint(0, 9)}:{rng.randint(0, 59):02d}:00",
                         rng.choice(["alipay", "wechat", "card", "balance"]), round(total, 2), pay_status))
        payment_id += 1
        if status == "refunded":
            refund_rows.append((refund_id, oid, _rand_date(rng, order_date, "2025-12-31"),
                                round(total * rng.choice([1.0, 1.0, 0.5]), 2), rng.choice(REASONS)))
            refund_id += 1

    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", order_rows)
    conn.executemany("INSERT INTO order_item VALUES (?,?,?,?,?,?)", item_rows)
    conn.executemany("INSERT INTO payment VALUES (?,?,?,?,?,?)", pay_rows)
    conn.executemany("INSERT INTO refund VALUES (?,?,?,?,?)", refund_rows)
    conn.commit()
    conn.close()
    return db_path


def table_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        conn.close()


if __name__ == "__main__":
    from app.config import SETTINGS

    path = seed_database(SETTINGS.db_path)
    print("已生成:", path)
    for name, count in table_counts(path).items():
        print(f"  {name:<12} {count:>7} 行")
