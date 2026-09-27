"""
sql_agent.py
SQL 子图：生成 SQL -> 执行 SQL。
"""
from langchain_community.utilities import SQLDatabase
from src.state import AgentState
from src.llm import get_llm


def sql_generation_node(state: AgentState) -> dict:
    print("\n[SQL Generation] 正在生成 SQL...")
    llm = get_llm()
    db = SQLDatabase.from_uri("sqlite:///data/financial.db")

    # 拿到表结构
    table_info = db.get_table_info()

    # 取用户最新提问
    question = state["messages"][-1].content

    # 手动拼 Prompt
    prompt = f"""你是一个SQLite数据库专家。根据下面的表结构，将用户的自然语言问题转换为一条SQL查询语句。

    表结构：
    {table_info}

    字段说明：
    - month: 月份（如 '2026-07'、'2026-08'）
    - region: 大区（华南区 / 华东区 / 华北区）
    - product_line: 产品线（SaaS / 硬件）
    - revenue: 收入，cost: 成本，expense: 费用，profit: 利润（单位：万元）

    规则：
    1. 只输出一条SQL语句，不要输出任何解释。
    2. 不要用 markdown 代码块包裹。
    3. **如果用户问的是"某个月"或"某几个月"的指标，必须按月份分组（GROUP BY month），分别罗列每个月的值，不要 SUM 求和。**
    4. **只有当用户明确说"累计 / 总和 / 合计 / 一共"时，才用 SUM 聚合。**
    5. **如果用户问"最近几个月"，用 ORDER BY month DESC LIMIT N 取最近的 N 个月，再按月份分组展示。**
    6. 输出结果按月份升序排列，方便阅读。

    用户问题：{question}

    SQL："""

    response = llm.invoke(prompt)
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()

    print(f"[SQL Generation] 生成 SQL：{sql}")
    return {"generated_sql": sql}


def sql_execution_node(state: AgentState) -> dict:
    print("\n[SQL Execution] 正在执行 SQL...")
    db = SQLDatabase.from_uri("sqlite:///data/financial.db")
    sql = state["generated_sql"]

    try:
        result = db.run(sql)
        print(f"[SQL Execution] 查询结果：\n{result}")
        return {"query_result": str(result), "sql_error": ""}
    except Exception as e:
        print(f"[SQL Execution] 报错：{e}")
        return {"sql_error": str(e), "sql_retry_count": state.get("sql_retry_count", 0) + 1}


def should_retry_sql(state: AgentState) -> str:
    """条件边：判断 SQL 是否需要重试"""
    if state.get("sql_error") and state.get("sql_retry_count", 0) < 3:
        print("[Conditional Edge] SQL 报错，触发自我纠错...")
        return "retry"
    print("[Conditional Edge] SQL 执行成功，继续下一步。")
    return "continue"



from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm

def sql_answer_node(state: AgentState) -> dict:
    """把 SQL 查询结果翻译成自然语言回答"""
    print("\n[SQL Answer] 正在生成自然语言回答...")

    llm = get_llm()
    question = state["messages"][-1].content
    query_result = state.get("query_result", "")

    prompt = f"""你是一个财务数据助手。请根据下面的 SQL 查询结果，用简洁的中文回答用户的问题。

    【用户问题】
    {question}

    【SQL 查询结果】
    {query_result}

    要求：
    1. 如果结果里包含多个维度（多个月份 / 多个区域 / 多个产品线），**请逐条罗列，不要合并成一个数字**。
    2. 如果结果里只有一个数字，就直接给出。
    3. 数字带单位（万元）。
    4. 不要解释 SQL，直接给结论。
    """

    response = llm.invoke([
        SystemMessage(content="你是一位专业的财务数据助手。"),
        HumanMessage(content=prompt),
    ])

    print(f"[SQL Answer] 回答：{response.content}")
    return {"final_answer": response.content}