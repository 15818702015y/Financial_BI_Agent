"""
state.py
定义 LangGraph 的全局状态（State）。
所有节点通过读写这个 State 来传递数据。
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ===== 对话历史（LangGraph 内置的消息累加器）=====
    messages: Annotated[list[BaseMessage], add_messages]

    # ===== 路由意图 =====
    # 取值：'查数据' | '问制度' | '做归因'
    intent: Literal["查数据", "问制度", "做归因"]

    # ===== SQL 相关 =====
    generated_sql: str          # LLM 生成的 SQL
    sql_error: str              # SQL 执行报错信息（用于自我纠错）
    sql_retry_count: int        # SQL 重试次数（防死循环）
    query_result: str           # SQL 执行结果（转成字符串）

    # ===== 数据分析相关 =====
    analysis_summary: str       # Pandas 计算出的关键指标摘要

    # ===== RAG 相关 =====
    business_context: str       # 从知识库检索到的业务背景

    # ===== 最终输出 =====
    final_answer: str           # 最终返回给用户的回答
    chart_json: str             # Plotly Figure 的 JSON 字符串