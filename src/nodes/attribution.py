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
import traceback


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
4. 数据摘要里的数字，就是用户所问区域的数字，直接引用，不要声称"无法从全公司拆分"。
5. 不要编造数据摘要里没有的数字。
"""


def data_analysis_node(state: AgentState) -> dict:
    print("\n[Data Analysis] 正在用 Pandas 计算关键指标...")

    raw = state.get("query_result", "[]")
    print(f"[Data Analysis] query_result 原始：{raw[:150]}...")

    # ===== 从用户问题里提取区域名 =====
    question = state["messages"][-1].content
    region = ""
    for r in ["华南区", "华东区", "华北区", "华中区", "西南区"]:
        if r in question:
            region = r
            break
    if not region:
        region = "所选区域"
    print(f"[Data Analysis] 识别区域：{region}")

    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        print(f"[Data Analysis] ❌ 解析失败：{e}")
        return {"analysis_summary": f"数据解析失败：{e}"}

    if not data:
        print("[Data Analysis] ❌ 数据为空")
        return {"analysis_summary": "未查询到相关数据。"}

    try:
        n_cols = len(data[0])
        print(f"[Data Analysis] 每行字段数：{n_cols}")

        # SQL 返回的字段顺序（按你实际 SQL 的输出）
        column_pool = ["month", "product_line", "revenue", "cost",
                       "expense", "profit", "profit_change",
                       "revenue_change", "cost_change", "expense_change"]
        columns = column_pool[:n_cols] if n_cols <= len(column_pool) else [f"col_{i}" for i in range(n_cols)]
        print(f"[Data Analysis] 使用列名：{columns}")

        df = pd.DataFrame(data, columns=columns)
        print(f"[Data Analysis] DataFrame 形状：{df.shape}")

        # ===== 按月份分组汇总 =====
        if "month" in df.columns and "profit" in df.columns:
            agg_dict = {"profit": "sum"}
            for col in ["revenue", "cost", "expense"]:
                if col in df.columns:
                    agg_dict[col] = "sum"

            monthly = df.groupby("month").agg(agg_dict).reset_index()
            monthly = monthly.sort_values("month")

            latest = monthly.iloc[-1]
            prev = monthly.iloc[-2] if len(monthly) >= 2 else None

            parts = []
            parts.append(
                f"{region}最新月份（{latest['month']}）汇总："
                f"利润 {latest['profit']:.1f} 万元"
            )
            if "revenue" in latest:
                parts[-1] += f"，收入 {latest['revenue']:.1f} 万元"
            if "expense" in latest:
                parts[-1] += f"，费用 {latest['expense']:.1f} 万元"
            parts[-1] += "。"

            if prev is not None:
                profit_change = latest["profit"] - prev["profit"]
                profit_pct = (profit_change / prev["profit"] * 100) if prev["profit"] != 0 else 0
                parts.append(
                    f"{region}环比上月（{prev['month']}）：利润变化 {profit_change:+.1f} 万元（{profit_pct:+.1f}%）。"
                )
                if "expense" in latest and "expense" in prev:
                    exp_change = latest["expense"] - prev["expense"]
                    exp_pct = (exp_change / prev["expense"] * 100) if prev["expense"] != 0 else 0
                    parts.append(
                        f"费用变化 {exp_change:+.1f} 万元（{exp_pct:+.1f}%）。"
                    )

            # 分产品线明细
            if "product_line" in df.columns:
                latest_df = df[df["month"] == latest["month"]]
                line_parts = []
                for _, row in latest_df.iterrows():
                    line_parts.append(f"{row['product_line']}利润 {row['profit']:.1f} 万元")
                if line_parts:
                    parts.append("分产品线：" + "，".join(line_parts) + "。")

            summary = " ".join(parts)
        else:
            summary = f"数据行数：{len(df)}。"
            if "profit" in df.columns:
                summary += f"利润合计 {df['profit'].sum():.1f} 万元。"

        print(f"[Data Analysis] ✅ 分析摘要：{summary}")
        return {"analysis_summary": summary}

    except Exception as e:
        print(f"[Data Analysis] ❌ 分析失败：{e}")
        traceback.print_exc()
        return {"analysis_summary": f"分析失败：{e}"}


def chart_node(state: AgentState) -> dict:
    print("\n[Chart] 正在生成图表...")
    raw = state.get("query_result", "[]")

    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        print(f"[Chart] ❌ 解析失败：{e}")
        return {"chart_json": ""}

    if not data:
        return {"chart_json": ""}

    try:
        n_cols = len(data[0])
        column_pool = ["month", "product_line", "revenue", "cost",
                       "expense", "profit", "profit_change",
                       "revenue_change", "cost_change", "expense_change"]
        columns = column_pool[:n_cols] if n_cols <= len(column_pool) else [f"col_{i}" for i in range(n_cols)]

        df = pd.DataFrame(data, columns=columns)

        # X 轴：优先 product_line
        x_col = "product_line" if "product_line" in df.columns else (
            "month" if "month" in df.columns else df.columns[0]
        )
        # Y 轴：优先 profit
        y_col = "profit" if "profit" in df.columns else df.columns[-1]

        # 颜色：按月份分组
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
        print(f"[Chart] ❌ 图表生成失败：{e}")
        traceback.print_exc()
        return {"chart_json": ""}


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

    # 如果原问题包含多个意图，加个提示
    if state.get("has_multi_intent"):
        answer += "\n\n---\n\n> 💡 您的问题中包含多个子问题，本次只回答了核心归因部分。如需了解其他问题，请单独提问。"

    print(f"[Attribution Generation] 报告生成完毕。")
    return {"final_answer": answer}