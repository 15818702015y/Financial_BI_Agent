"""
rag_only_agent.py
纯 RAG 节点：用户问制度时，直接检索知识库并生成回答。
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm

load_dotenv(override=True)
CHROMA_DIR = "data/chroma_db"


def rag_only_node(state: AgentState) -> dict:
    print("\n[问制度] 正在检索制度文档...")

    # 1. 检索知识库
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    question = state["messages"][-1].content
    query = f"业务背景 行业情况 制度 {question}"
    results = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([r.page_content for r in results])

    # 2. 用 LLM 基于检索结果生成回答
    llm = get_llm()
    prompt = f"""你是一位专业的财务分析师。请根据以下检索到的知识库文档，回答用户的问题。

    【检索到的文档】
    {context}

    【用户问题】
    {question}

    回答要求：
    1. **专业、严谨、结构化**。不要使用"我觉得"、"感觉"、"就这些"等口语化表达。
    2. **只基于检索到的文档内容作答**，不要编造文档中没有的信息。
    3. 如果文档内容不足以回答问题，要明确说明"根据现有资料，无法回答"，而不是强行编造。
    4. 如果涉及多个要点，**用 Markdown 列表或小标题组织**，条理清晰。
    5. 引用具体的**数据、条款、时间**，不要泛泛而谈。
    6. 结尾不要加"希望对你有帮助"、"如果还有其他问题"等客套话。
    7. 如果问题涉及财务数据，请在回答末尾注明**数据来源**（如"来源：2026_Q3 市场简报"）。
    """
    response = llm.invoke([
        SystemMessage(content="你是一位耐心、专业的财务同事，喜欢用口语化的方式帮助员工解决问题。"),
        HumanMessage(content=prompt),
    ])

    print(f"[问制度] 回答生成完毕。")
    return {"final_answer": response.content}