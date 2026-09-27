"""
attribution.py
归因分析节点：结合数据 + 业务背景，生成最终归因报告。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm
from src.state import AgentState
import ast
import pandas as pd
import plotly.express as px
import plotly.io as pio

ATTRIBUTION_PROMPT = """你是一个资深财务分析师。请根据以下信息，生成一份归因分析报告。

【数据摘要】
{analysis_summary}

【业务背景】
{business_context}

【用户问题】
{question}

要求：
1. 用Markdown格式输出，包含：结论、原因分析、改进建议。
2. 归因必须结合业务背景，不能凭空猜测。
3. 语言简洁专业，适合管理层阅读。
"""

def data_analysis_node(state: AgentState) -> dict:
    print("\n[Data Analysis] 正在用 Pandas 计算关键指标...")

    import ast
    import pandas as pd
    from src.state import AgentState

    def data_analysis_node(state: AgentState) -> dict:
        print("\n[Data Analysis] 正在用 Pandas 计算关键指标...")

        # 1. 拿到 SQL 查询结果字符串
        raw_result = state.get("query_result", "[]")

        try:
            # 2. 把字符串解析成 Python 列表
            # 因为 SQL 返回的格式是 [('SaaS', '2026-08', 480.0, ...), ...]
            data = ast.literal_eval(raw_result)
        except Exception as e:
            print(f"[Data Analysis] 解析 SQL 结果失败：{e}")
            return {"analysis_summary": f"数据解析失败：{e}"}

        if not data:
            return {"analysis_summary": "未查询到相关数据。"}

        # 3. 转成 DataFrame
        # 根据我们 SQL 查询的字段顺序，手动指定列名
        # 注意：如果你的 SQL 字段变了，这里的列名也要跟着变
        columns = ["product_line", "month", "revenue", "cost", "expense", "profit",
                   "profit_change", "revenue_change", "cost_change", "expense_change"]

        # 如果 SQL 返回的字段数不对，做个兜底
        if len(data[0]) != len(columns):
            # 尝试根据实际长度截取列名
            columns = columns[:len(data[0])]

        df = pd.DataFrame(data, columns=columns)

        # 4. 动态计算关键指标
        total_profit_change = df["profit_change"].sum()
        total_expense_change = df["expense_change"].sum()
        total_revenue_change = df["revenue_change"].sum()

        # 计算环比百分比（基期为上月，上月 = 本月 - 变化量）
        # 避免除以 0
        last_month_profit = df["profit"].sum() - total_profit_change
        last_month_expense = df["expense"].sum() - total_expense_change

        profit_pct = (total_profit_change / last_month_profit * 100) if last_month_profit != 0 else 0
        expense_pct = (total_expense_change / last_month_expense * 100) if last_month_expense != 0 else 0

        # 找出利润下滑最严重的产品线
        worst_line = df.loc[df["profit_change"].idxmin()]
        worst_line_name = worst_line["product_line"]
        worst_line_change = worst_line["profit_change"]

        # 5. 拼接成一段自然语言摘要
        summary = (
            f"本期利润环比变化 {profit_pct:.1f}%，"
            f"费用环比变化 {expense_pct:.1f}%，"
            f"收入环比变化 {total_revenue_change:.1f} 万元。"
            f"其中利润下滑最严重的产品线是【{worst_line_name}】，"
            f"利润减少 {abs(worst_line_change):.1f} 万元。"
        )

        print(f"[Data Analysis] 分析摘要：{summary}")
        return {"analysis_summary": summary}


def chart_node(state: AgentState) -> dict:
    print("\n[Chart] 正在生成图表...")
    raw = state.get("query_result", "[]")
    print(f"[Chart] query_result 原始：{raw[:200]}...")

    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        print(f"[Chart] ❌ 解析 query_result 失败：{e}")
        return {"chart_json": ""}

    if not data:
        print("[Chart] ❌ 数据为空")
        return {"chart_json": ""}

    try:
        # 自适应列名：SQL 返回几个字段，就用几个
        n_cols = len(data[0])
        print(f"[Chart] 每行字段数：{n_cols}")

        # 用最常见的财务字段顺序
        column_pool = ["month", "product_line", "revenue", "cost",
                       "expense", "profit", "profit_change",
                       "revenue_change", "cost_change", "expense_change"]

        if n_cols <= len(column_pool):
            columns = column_pool[:n_cols]
        else:
            columns = [f"col_{i}" for i in range(n_cols)]

        print(f"[Chart] 使用列名：{columns}")

        df = pd.DataFrame(data, columns=columns)
        print(f"[Chart] DataFrame 形状：{df.shape}")
        print(f"[Chart] DataFrame 内容：\n{df}")

        # 自适应选 X 轴和 Y 轴
        # X 轴优先 product_line，否则用第 2 列
        if "product_line" in df.columns:
            x_col = "product_line"
        elif "month" in df.columns:
            x_col = "month"
        else:
            x_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Y 轴优先 profit，否则用最后一列
        if "profit" in df.columns:
            y_col = "profit"
        else:
            y_col = df.columns[-1]

        print(f"[Chart] X 轴：{x_col}，Y 轴：{y_col}")

        # 用月份做颜色分组（如果有 month 字段）
        color_col = "month" if "month" in df.columns and x_col != "month" else None

        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            barmode="group" if color_col else "relative",
            title="各产品线利润对比（万元）",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="-apple-system, PingFang SC", size=12),
        )

        chart_json = pio.to_json(fig)
        print(f"[Chart] ✅ 图表生成成功，JSON 长度：{len(chart_json)}")
        return {"chart_json": chart_json}

    except Exception as e:
        import traceback
        print(f"[Chart] ❌ 图表生成失败：{e}")
        traceback.print_exc()
        return {"chart_json": ""}
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="-apple-system, PingFang SC", size=12),
    )
    return {"chart_json": pio.to_json(fig)}

def attribution_generation_node(state: AgentState) -> dict:
    print("\n[Attribution Generation] 正在生成归因报告...")
    llm = get_llm()

    question = state["messages"][-1].content
    prompt = ATTRIBUTION_PROMPT.format(
        analysis_summary=state.get("analysis_summary", "无"),
        business_context=state.get("business_context", "无"),
        question=question,
    )

    response = llm.invoke([
        SystemMessage(content="你是一位资深财务分析师。"),
        HumanMessage(content=prompt),
    ])

    answer = response.content
    print(f"[Attribution Generation] 报告生成完毕。")
    return {"final_answer": answer}