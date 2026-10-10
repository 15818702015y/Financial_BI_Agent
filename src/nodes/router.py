
"""
router.py
路由节点：用 LLM 判断用户意图。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm

ROUTER_PROMPT = """你是一个财务智能助手的路由分发器。
请判断用户的问题属于以下哪一类：

【查数据】
用户想要具体的财务数字，不涉及原因分析。
示例：上个月华南区收入多少、8月利润是多少。

【问制度】
用户询问制度规则、业务背景、行业情况、市场动态、客户信息等非结构化知识。
示例：差旅报销标准、Q3行业情况如何、北方能源集团合同情况。

【做归因】
用户询问"为什么"某指标变化，或同时要求数字与原因。
示例：为什么华南区利润下降、华北区收入骤降的原因。

【闲聊】
问候、能力询问、与财务完全无关的问题。
示例：你好、你能做什么、今天天气如何。

【复合意图处理规则】
如果用户一句话里包含多个独立问题（如"为什么利润下降，顺便告诉我差旅制度"），只识别最靠前、最主要的意图，忽略其他问题。

判断原则：
1. 只要问题里包含"为什么"或"原因"，优先归为【做归因】。
2. 只要问题明确要具体数字且不涉及原因，归为【查数据】。
3. 涉及制度、背景、事实性描述的，归为【问制度】。
4. 问候、能力询问、与财务无关的，归为【闲聊】。
5. 拿不准时，归为【闲聊】。

只输出一个词：查数据 / 问制度 / 做归因 / 闲聊。不要输出任何其他内容。"""


VALID_INTENTS = ["查数据", "问制度", "做归因", "闲聊"]


def router_node(state: AgentState) -> dict:
    print("\n[Router] 正在识别用户意图...")
    llm = get_llm()

    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_message),
    ])

    intent = response.content.strip()

    # 兜底：如果 LLM 输出意外内容，默认走"闲聊"
    if intent not in VALID_INTENTS:
        print(f"[Router] 输出异常：{intent}，兜底为'闲聊'")
        intent = "闲聊"

    # 检测复合意图：如果原问题里同时出现多个意图的关键词，加提示
    last_msg = state["messages"][-1].content
    has_why = any(k in last_msg for k in ["为什么", "原因", "怎么下降", "为何"])
    has_policy = any(k in last_msg for k in ["制度", "规定", "标准", "流程", "政策"])
    has_number = any(k in last_msg for k in ["多少", "几个", "几万", "金额", "数字"])
    has_multi = sum([has_why, has_policy, has_number]) >= 2

    if has_multi:
        print(f"[Router] ⚠️ 检测到复合意图，只处理主意图：{intent}")

    print(f"[Router] 识别意图：{intent}")
    return {"intent": intent, "has_multi_intent": has_multi}