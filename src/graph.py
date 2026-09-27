"""
graph.py
定义 LangGraph 的主图与子图，并编译成可执行的应用。
"""
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.nodes.router import router_node
from src.nodes.rag_agent import rag_retrieval_node
from src.nodes.rag_only_agent import rag_only_node
from src.nodes.sql_agent import sql_generation_node, sql_execution_node, should_retry_sql, sql_answer_node
from src.nodes.attribution import (
    data_analysis_node, attribution_generation_node, chart_node
)

def build_sql_subgraph():
    """构建'查数据'子图：生成 SQL -> 执行 SQL -> (可选)自纠错"""
    subgraph = StateGraph(AgentState)
    subgraph.add_node("sql_generation", sql_generation_node)
    subgraph.add_node("sql_execution", sql_execution_node)
    subgraph.add_edge(START, "sql_generation")
    subgraph.add_edge("sql_generation", "sql_execution")
    # 新增回答生成节点
    subgraph.add_node("sql_answer", sql_answer_node)
    subgraph.add_conditional_edges(
        "sql_execution",
        should_retry_sql,
        {"retry": "sql_generation", "continue": "sql_answer"},
    )
    subgraph.add_edge("sql_answer", END)
    return subgraph.compile()


def build_attribution_subgraph():
    """构建'做归因'子图：SQL -> 分析 -> 图表 -> RAG -> 报告"""
    subgraph = StateGraph(AgentState)

    # 1. 注册所有节点（必须先注册，才能连边）
    subgraph.add_node("sql_generation", sql_generation_node)
    subgraph.add_node("sql_execution", sql_execution_node)
    subgraph.add_node("data_analysis", data_analysis_node)
    subgraph.add_node("chart", chart_node)                      # ← 新增
    subgraph.add_node("rag_retrieval", rag_retrieval_node)
    subgraph.add_node("attribution_generation", attribution_generation_node)

    # 2. 连接节点
    subgraph.add_edge(START, "sql_generation")
    subgraph.add_edge("sql_generation", "sql_execution")

    # SQL 执行后：报错重试 / 成功继续
    subgraph.add_conditional_edges(
        "sql_execution",
        should_retry_sql,
        {"retry": "sql_generation", "continue": "data_analysis"},
    )

    subgraph.add_edge("data_analysis", "chart")                 # ← 新增
    subgraph.add_edge("chart", "rag_retrieval")                 # ← 新增
    subgraph.add_edge("rag_retrieval", "attribution_generation")
    subgraph.add_edge("attribution_generation", END)

    return subgraph.compile()


def build_main_graph():
    main = StateGraph(AgentState)
    main.add_node("router", router_node)

    # 把三个子图挂上来
    sql_subgraph = build_sql_subgraph()
    attribution_subgraph = build_attribution_subgraph()
    main.add_node("sql_subgraph", sql_subgraph)
    main.add_node("attribution_subgraph", attribution_subgraph)
    main.add_node("rag_only", rag_only_node)
    main.add_edge(START, "router")
    main.add_conditional_edges("router", lambda s: s["intent"], {
        "查数据": "sql_subgraph",
        "问制度": "rag_only",
        "做归因": "attribution_subgraph",
    })
    main.add_edge("sql_subgraph", END)
    main.add_edge("rag_only", END)
    main.add_edge("attribution_subgraph", END)

    return main.compile()