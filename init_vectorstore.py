"""
init_vectorstore.py
作用：把 data/knowledge_base/ 下的 .md 文档向量化，存入 ChromaDB。
运行方式：python init_vectorstore.py
运行后会在 data/ 下生成 chroma_db/ 目录。
"""
import os
import glob
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

KB_DIR = "data/knowledge_base"
CHROMA_DIR = "data/chroma_db"

# 1. 读取所有 md 文件
docs = []
for filepath in glob.glob(os.path.join(KB_DIR, "**", "*.md"), recursive=True):
    loader = TextLoader(filepath, encoding="utf-8")
    docs.extend(loader.load())
    print(f"  读取：{filepath}")

# 2. 切分文档
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os

all_chunks = []
for doc in docs:
    filepath = doc.metadata.get("source", "")
    filename = os.path.basename(filepath)
    content = doc.page_content

    if "费用报销规则" in filepath:
        # ===== 制度手册：按 ### 三级标题切，每条 Q&A 独立成 chunk =====
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("###", "question")],
            strip_headers=False,
        )
        splits = markdown_splitter.split_text(content)
        # 补上 source 和 category metadata
        for s in splits:
            s.metadata["source"] = filepath
            s.metadata["category"] = "制度手册"
            s.metadata["filename"] = filename
        all_chunks.extend(splits)
        print(f"  [制度] {filename} → {len(splits)} chunks")
    else:
        # ===== 业务背景：按 ## 二级标题切，保证每个区域独立成 chunk =====
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")],
            strip_headers=False,
        )
        splits = md_splitter.split_text(content)

        # 如果按 ## 切完还只有 1 个，说明文档没有 ## 标题，改用按段落切
        if len(splits) <= 1:
            para_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=20,
                separators=["\n\n", "\n", "。", " "],
            )
            splits = para_splitter.create_documents([content])

        # 补上 metadata
        for s in splits:
            s.metadata["source"] = filepath
            s.metadata["category"] = "业务背景"
            s.metadata["filename"] = filename
        all_chunks.extend(splits)
        print(f"  [业务] {filename} → {len(splits)} chunks")

chunks = all_chunks
print(f"\n切分为 {len(chunks)} 个 chunk")

# 3. 向量化并存入 ChromaDB
load_dotenv(override=True)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
print(f"✅ 向量库已保存到 {CHROMA_DIR}")

# 4. 测试检索
results = vectorstore.similarity_search("华南区 利润 下降 原因", k=2)
print("\n检索测试：")
for r in results:
    print(f"  - {r.page_content[:80]}...")