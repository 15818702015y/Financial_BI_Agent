import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.state import AgentState

load_dotenv(override=True)

CHROMA_DIR = "data/chroma_db"

def get_embeddings():
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
    )

def rag_retrieval_node(state: AgentState) -> dict:
    print("\n[RAG Retrieval] 正在检索业务背景...")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

    raw = state.get("analysis_summary") or state["messages"][-1].content
    user_question = state["messages"][-1].content

    # ===== 1. 提取关键词：区域 + 月份 =====
    regions = ["华南区", "华东区", "华北区", "华中区", "西南区"]
    target_region = None
    for r in regions:
        if r in user_question:
            target_region = r
            break

    # ===== 2. 拿全部"业务背景"类 chunk =====
    all_docs = vectorstore.get(include=['metadatas', 'documents'])
    business_chunks = []
    for i, m in enumerate(all_docs['metadatas']):
        if m.get('category') == '业务背景':
            business_chunks.append({
                'content': all_docs['documents'][i],
                'source': m.get('source', ''),
            })

    print(f"[RAG] 业务背景类 chunk 总数：{len(business_chunks)}")

    # ===== 3. 优先过滤：包含目标区域的 chunk =====
    if target_region:
        region_chunks = [c for c in business_chunks if target_region in c['content']]
        print(f"[RAG] 命中 '{target_region}' 的 chunk：{len(region_chunks)}")
    else:
        region_chunks = business_chunks

    # ===== 4. 在候选里做语义排序 =====
    if region_chunks:
        # 用 region_chunks 里的内容构造临时检索上下文
        from langchain_core.documents import Document
        temp_docs = [
            Document(page_content=c['content'], metadata={'source': c['source']})
            for c in region_chunks
        ]
        # 用 Chroma 已有的 embedding 做相似度排序
        query = f"{target_region or ''} 经营 利润 原因 {raw}"
        # 直接在候选里按字符串匹配排序（简单可靠）
        # 优先包含"8月"或目标月份的
        priority = [d for d in temp_docs if ('8月' in d.page_content or '2026-08' in d.page_content)]
        other = [d for d in temp_docs if d not in priority]
        final = (priority + other)[:5]
    else:
        # 没命中区域，退回普通语义检索
        final = vectorstore.similarity_search(f"业务背景 {raw}", k=5)

    # ===== 5. 调试输出 =====
    for i, r in enumerate(final):
        src = r.metadata.get('source', '?')
        print(f"[RAG-{i}] {src}")
        print(f"        {r.page_content[:80]}...")

    context = "\n\n---\n\n".join([r.page_content for r in final])
    print(f"[RAG Retrieval] 检索到 {len(final)} 个业务背景片段")
    return {"business_context": context}