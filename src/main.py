"""
main.py
项目主入口：编译图并运行测试。
"""
from langchain_core.messages import HumanMessage
from src.graph import build_main_graph


def main():
    graph = build_main_graph()

    # 模拟用户提问
    user_query = "华南区 8 月发生了什么？"
    print("=" * 60)
    print(f"用户提问：{user_query}")
    print("=" * 60)

    result = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "sql_retry_count": 0,
    })

    print("\n" + "=" * 60)
    print("最终输出：")
    print("=" * 60)
    if result.get("final_answer"):
        print(result["final_answer"])
    elif result.get("query_result"):
        print("查询结果：", result["query_result"])

if __name__ == "__main__":
    main()