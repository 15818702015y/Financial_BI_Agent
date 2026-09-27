
"""
router.py
路由节点：用 LLM 判断用户意图。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm

ROUTER_PROMPT = """你是一个财务智能助手的路由分发器。
请判断用户的问题属于以下哪一类：

- 查数据：询问具体的财务数字（如"上个月华南区收入多少"、"8月利润是多少"）。
- 问制度：询问财务制度、报销标准、流程规范、业务背景、行业情况、市场动态、客户信息等非结构化知识（如"差旅报销标准"、"Q3行业情况如何"、"华南区8月发生了什么"、"北方能源集团合同情况"）。
- 做归因：明确询问"为什么"某指标变化，需要分析原因（如"为什么华南区利润下降"）。

判断原则：
1. 只要问题不是"要具体数字"也不是"问为什么变化"，就归到"问制度"。
2. 优先考虑"问制度"，因为它覆盖面最广。

只输出一个词：查数据 / 问制度 / 做归因。不要输出任何其他内容。"""


def router_node(state: AgentState) -> dict:
    print("\n[Router] 正在识别用户意图...")
    llm = get_llm()

    # 取最后一条用户消息
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_message),
    ])

    intent = response.content.strip()
    # 兜底：如果 LLM 输出了奇怪的东西，默认走"查数据"
    if intent not in ["查数据", "问制度", "做归因"]:
        intent = "查数据"

    print(f"[Router] 识别意图：{intent}")
    return {"intent": intent}