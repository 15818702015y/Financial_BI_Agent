"""
init_db.py
一次性初始化财务数据库，建表并插入模拟数据。
运行方式：python init_db.py
"""
import sqlite3
import os
import random
from datetime import datetime

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "financial.db")
os.makedirs(DB_DIR, exist_ok=True)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"删除旧数据库：{DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ========== 建表 ==========
cursor.execute("""
CREATE TABLE profit_statement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    region TEXT NOT NULL,
    product_line TEXT NOT NULL,
    revenue REAL NOT NULL,
    cost REAL NOT NULL,
    expense REAL NOT NULL,
    profit REAL NOT NULL
)
""")

# ========== 数据设计 ==========
regions = ["华南区", "华东区", "华北区", "华中区", "西南区"]
product_lines = ["SaaS", "硬件", "咨询服务", "运维服务"]
months = ["2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07", "2026-08"]

# 各大区/产品线的基础体量（万元）
base = {
    ("华南区", "SaaS"): 500, ("华南区", "硬件"): 300,
    ("华南区", "咨询服务"): 150, ("华南区", "运维服务"): 200,
    ("华东区", "SaaS"): 600, ("华东区", "硬件"): 400,
    ("华东区", "咨询服务"): 200, ("华东区", "运维服务"): 250,
    ("华北区", "SaaS"): 450, ("华北区", "硬件"): 350,
    ("华北区", "咨询服务"): 180, ("华北区", "运维服务"): 220,
    ("华中区", "SaaS"): 320, ("华中区", "硬件"): 220,
    ("华中区", "咨询服务"): 120, ("华中区", "运维服务"): 150,
    ("西南区", "SaaS"): 280, ("西南区", "硬件"): 180,
    ("西南区", "咨询服务"): 100, ("西南区", "运维服务"): 130,
}

random.seed(42)  # 固定随机种子，保证每次生成数据一致
data = []

for month in months:
    month_idx = months.index(month)
    # 自然增长趋势（每月 +2%）
    growth = 1 + 0.02 * month_idx

    for region in regions:
        for pl in product_lines:
            base_revenue = base[(region, pl)] * growth
            # 每月 ±5% 正常波动
            revenue = round(base_revenue * random.uniform(0.95, 1.05), 1)
            # 成本 = 收入的 60% ± 3%
            cost = round(revenue * random.uniform(0.57, 0.63), 1)
            # 费用 = 收入的 16% ± 2%
            expense = round(revenue * random.uniform(0.14, 0.18), 1)

            # ========== 异常场景植入 ==========
            # 1) 华南区 8 月：费用飙升（夏日大促 + 台风物流）
            if region == "华南区" and month == "2026-08":
                expense = round(expense * 1.25, 1)  # 费用 +25%
                revenue = round(revenue * 0.96, 1)  # 收入略降

            # 2) 华北区 8 月：SaaS 收入骤降（大客户流失）
            if region == "华北区" and month == "2026-08" and pl == "SaaS":
                revenue = round(revenue * 0.44, 1)  # 收入腰斩
                cost = round(cost * 0.55, 1)       # 成本同步下降

            # 3) 华东区：稳定增长（不做任何异常处理，作为对照组）

            # 利润 = 收入 - 成本 - 费用
            profit = round(revenue - cost - expense, 1)

            data.append((month, region, pl, revenue, cost, expense, profit))

cursor.executemany("""
INSERT INTO profit_statement
(month, region, product_line, revenue, cost, expense, profit)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", data)

conn.commit()

# ========== 验证 ==========
print(f"\n✅ 数据库初始化完成！")
print(f"📁 路径：{os.path.abspath(DB_PATH)}")
print(f"📊 共插入 {len(data)} 条记录\n")

print("=" * 70)
print("数据概览：")
print("=" * 70)
cursor.execute("SELECT COUNT(DISTINCT month), COUNT(DISTINCT region), COUNT(DISTINCT product_line) FROM profit_statement")
m, r, p = cursor.fetchone()
print(f"  月份数：{m}  大区数：{r}  产品线数：{p}")

print("\n华南区月度利润（异常场景）：")
cursor.execute("""
SELECT month, SUM(profit) as total_profit, SUM(expense) as total_expense
FROM profit_statement WHERE region='华南区'
GROUP BY month ORDER BY month
""")
for row in cursor.fetchall():
    print(f"  {row[0]}  利润：{row[1]:>7.1f}万  费用：{row[2]:>7.1f}万")

print("\n华北区 SaaS 月度收入（异常场景）：")
cursor.execute("""
SELECT month, revenue FROM profit_statement
WHERE region='华北区' AND product_line='SaaS'
ORDER BY month
""")
for row in cursor.fetchall():
    print(f"  {row[0]}  收入：{row[1]:>7.1f}万")

conn.close()
print("\n🎉 一切就绪，可以开始 RAG 检索和归因分析！")