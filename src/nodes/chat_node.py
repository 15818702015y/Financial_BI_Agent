"""
chat_node.py
闲聊节点：处理问候、能力询问、与财务无关的问题。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm

CHAT_PROMPT = """你是智能财务BI助手，专注于财务数据分析、制度问答、归因分析三类任务。

用户现在问了一个与财务无关或属于问候类的问题，请友好回应，并引导用户回到财务话题。

要求：
1. 简洁回应，不超过两句话。
2. 主动介绍你能做什么（查数据、问制度、做归因）。
3. 不要编造能力（比如你无法查天气、讲笑话）。

示例回应：
- "你好！我是财务BI助手，可以帮你查数据、问制度、做归因分析。有什么财务问题需要帮忙吗？"
- "我主要擅长财务数据分析，比如查利润、问报销标准、分析利润波动原因。试试问我'华南区8月利润是多少'？"
"""


def chat_node(state: AgentState) -> dict:
    print("\n[Chat] 正在处理闲聊...")
    llm = get_llm()

    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=CHAT_PROMPT),
        HumanMessage(content=last_message),
    ])

    answer = response.content.strip()
    print(f"[Chat] 回应完毕。")
    return {"final_answer": answer}